# main.py
import os
import torch
import numpy as np
from config.config import Config
from models.component import Component
from data.dataloaders import get_dataset_tasks, get_task_loader, get_test_loader, get_permuted_mnist_tasks
from training.trainer import train_component, handle_memory_overflow, create_new_component, save_model_state
from training.expansion import check_expansion_mmd
from utils.memory import MemoryBuffer
from utils.logging import configure_logging, save_args_to_file
from utils.visualization import plot_mse_similarity_matrix, save_component_samples_as_png
from utils.testing import test_components  # 导入测试函数
from utils.data_tracker import DataTracker  # 导入数据跟踪器
from torch.utils.tensorboard import SummaryWriter
import logging


def main():
    # 解析配置参数
    config = Config()
    args = config.parse_args()
    
    # 为不同的数据集设置合适的输入通道数
    if args.dataset.lower() in ["cifar10", "cifar100"]:
        args.input_channels = 3
        args.img_size = 32
        # CIFAR100有100个类别，所以任务数应该是20
        if args.dataset.lower() == "cifar100":
            args.num_tasks = 20
    elif args.dataset.lower() == "mnist":
        args.input_channels = 1
        args.img_size = 32
    elif args.dataset.lower() == "permuted_mnist":
        args.input_channels = 1
        args.img_size = 32
    
    # 生成日志目录
    time_dir = config.generate_log_dir(args, args.dataset)
    
    # 配置日志
    configure_logging(args, args.dataset, time_dir)
    save_args_to_file(args, time_dir)
    writer = SummaryWriter(log_dir=time_dir)
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 初始化内存和组件
    memory = MemoryBuffer(size=args.memory_size, input_channels=args.input_channels)  # 传递输入通道数
    components = []
    component = create_new_component(args, components)
    
    # 初始化数据跟踪器
    data_tracker = DataTracker(save_dir=time_dir)
    
    # 获取数据
    if args.dataset.lower() == "permuted_mnist":
        task_datasets, p_test_loaders = get_permuted_mnist_tasks(
            num_tasks=args.num_tasks,
            fraction=args.dataset_fraction,
            batch_size=args.batch_size
        )
    else:
        task_datasets, p_test_loaders = get_dataset_tasks(
            dataset_name=args.dataset,
            num_tasks=args.num_tasks,
            fraction=args.dataset_fraction,
            batch_size=args.batch_size
        )
    
    expansion_count = 0  # 记录扩展次数
    global_step = 0

    for task_id, task_data in enumerate(task_datasets):
        digits = task_id * 2
        logging.info(f"开始训练任务 {task_id + 1}，包含数字 {digits} 和 {digits + 1}")
        print(f"开始训练任务 {task_id + 1}，包含数字 {digits} 和 {digits + 1}")
        
        task_loader = get_task_loader(task_data, batch_size=args.batch_size)

        for batch_idx, (data, labels) in enumerate(task_loader):
            data = data.to(device)
            labels = labels.to(device)
            flat_data = data

            # 更新数据流大小
            batch_size = data.size(0)
            data_tracker.update_data_flow(batch_size)

            signal = memory.add_samples(flat_data.cpu().numpy(), labels.cpu().numpy())

            # 添加内存状态日志
            current_memory_size = len(memory.buffer)
            max_memory_size = memory.size
            print(f"[Batch {batch_idx}] 内存状态: {current_memory_size}/{max_memory_size} 样本")
            logging.info(f"[Batch {batch_idx}] 内存状态: {current_memory_size}/{max_memory_size} 样本")

            if signal:  # 如果内存未满
                memory_samples, memory_labels = memory.get_samples()
                memory_samples = memory_samples.float().to(device)
                memory_labels = memory_labels.long().to(device)
                train_component(components[-1], memory_samples, memory_labels, args.batch_size, args.n_epochs)
            else:  # 如果内存已满
                memory_samples, memory_labels = memory.get_samples()
                should_expand, current_min_mmd = check_expansion_mmd(
                    components, memory_samples, threshold=args.threshold, n_steps=args.n_steps
                )

                # 添加MMD检查结果日志
                print(
                    f"[Batch {batch_idx}] 内存已满! 最小MMD值: {current_min_mmd:.6f}，扩展决策: {'是' if should_expand else '否'}")
                logging.info(
                    f"[Batch {batch_idx}] 内存已满! 最小MMD值: {current_min_mmd:.6f}，扩展决策: {'是' if should_expand else '否'}")

                if should_expand:
                    expansion_count += 1
                    # 记录扩展次数
                    data_tracker.increment_expansion()
                    # 添加扩展通知
                    print(
                        f"!! 组件扩展 #{expansion_count} 在任务 {task_id + 1}，批次 {batch_idx} (MMD={current_min_mmd:.6f}) !!")
                    logging.info(
                        f"!! 组件扩展 #{expansion_count} 在任务 {task_id + 1}，批次 {batch_idx} (MMD={current_min_mmd:.6f}) !!")

                    component = create_new_component(args, components)
                    memory.buffer = []
                    memory.buffer = list(zip(flat_data.cpu().numpy().tolist(), labels.cpu().numpy().tolist()))

                    # 记录组件数量变化
                    print(f"当前组件数量: {len(components)} (新增组件 #{len(components) - 1})")
                    logging.info(f"当前组件数量: {len(components)} (新增组件 #{len(components) - 1})")

                    if current_min_mmd is not None:
                        writer.add_scalar("Min_MSE/Value", current_min_mmd, global_step)
                        global_step += 1
                else:
                    # 添加内存优化通知
                    print(f"[Batch {batch_idx}] 无需扩展，执行内存优化处理...")
                    logging.info(f"[Batch {batch_idx}] 无需扩展，执行内存优化处理...")

                    component = handle_memory_overflow(
                        memory, components, component, flat_data, labels, device, time_dir,
                        strategy=args.strategy,
                        input_channels=args.input_channels
                    )
                    memory.print_statistics()
                    memory_samples, memory_labels = memory.get_samples()
                    memory_samples = memory_samples.float().to(device)
                    memory_labels = memory_labels.long().to(device)

                    # 添加内存优化后训练通知
                    print(f"[Batch {batch_idx}] 内存优化后训练组件 #{len(components)}")
                    logging.info(f"[Batch {batch_idx}] 内存优化后训练组件 #{len(components)}")

                    train_component(components[-1], memory_samples, memory_labels, args.batch_size, args.n_epochs)

                # 记录当前状态
                data_tracker.record_state(task_id=task_id, batch_idx=batch_idx)

        # 在每个任务结束时保存模型
        task_model_name = f"task_{task_id + 1}_model.pth"
        save_model_state(components, time_dir, task_model_name)

        # 添加任务完成统计
        print(f"===== 任务 {task_id + 1} 完成 =====")
        print(f"总扩展次数: {expansion_count}，当前组件数量: {len(components)}")
        print("===========================")
        logging.info(f"===== 任务 {task_id + 1} 完成 =====")
        logging.info(f"总扩展次数: {expansion_count}，当前组件数量: {len(components)}")
        logging.info("===========================")

    # 保存模型
    save_model_state(components, time_dir, "model_before_delete.pth")
    
    # 保存可视化结果
    plot_mse_similarity_matrix(components, time_dir, "div_after_mse_similarity_matrix.png")
    
    # 保存删除后生成的图片
    model_path_after_delete = os.path.join(time_dir, "model_after_delete.pth")
    save_model_state(components, time_dir, "model_after_delete.pth")

    # 测试组件性能
    test_components(components, p_test_loaders, device, args)

    # 保存数据跟踪记录
    data_tracker.save_records()

    logging.info("训练完成。")


if __name__ == "__main__":
    main()