# training/trainer.py
import torch
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter
import os
# 修复相对导入问题
from utils.memory import MemoryBuffer
from models.component import Component  # 修复导入路径


def train_component(component, memory_samples, memory_labels, batch_size, n_epochs=10):
    """
    训练一个组件，包括VAE和分类器
    """
    total_samples = memory_samples.size(0)
    num_batches = (total_samples + batch_size - 1) // batch_size

    print(f"总样本数: {total_samples}, 总批次数: {num_batches}")
    logging.info(f"总样本数: {total_samples}, 总批次数: {num_batches}")

    for epoch in range(n_epochs):
        component.vae.update_p(epoch, n_epochs)
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_data = memory_samples[start_idx:end_idx]
            batch_labels = memory_labels[start_idx:end_idx]

            # 训练VAE
            vae_loss = component.train_vae(batch_data)

            # 训练分类器
            classifier_loss = component.train_classifier(batch_data, batch_labels)

        if (epoch + 1) % 9 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}] Batch {batch_num + 1}/{num_batches} "
                  f"VAE Loss: {vae_loss:.4f} Classifier Loss: {classifier_loss:.4f}")
            logging.info(f"Epoch [{epoch + 1}/{n_epochs}] Batch {batch_num + 1}/{num_batches} "
                         f"VAE Loss: {vae_loss:.4f} Classifier Loss: {classifier_loss:.4f}")


def handle_memory_overflow(memory, components, component, flat_data, labels, device, time_dir,
                          strategy="diversity", input_channels=1):
    """
    高效全批量处理的内存溢出解决方案
    """
    if strategy == "sliding_window":
        print("内存已满，执行滑动窗口策略")
        logging.warning("内存已满，执行滑动窗口策略")
        num_new_samples = len(flat_data)

        # 使用滑动窗口策略更新内存
        memory.buffer = memory.buffer[num_new_samples:] + list(zip(flat_data.tolist(), labels.tolist()))
        return component
    else:
        print("内存已满，执行多样性样本选择")
        logging.warning("内存已满，执行多样性样本选择")

        # 获取并组合当前内存中的样本与新样本
        current_samples, current_labels = memory.get_samples()
        
        # 确保current_samples具有正确的通道数
        if current_samples.size(1) != input_channels:
            if input_channels == 3 and current_samples.size(1) == 1:
                # 单通道转三通道
                current_samples = current_samples.repeat(1, 3, 1, 1)
            elif input_channels == 1 and current_samples.size(1) == 3:
                # 三通道转单通道
                current_samples = current_samples.mean(dim=1, keepdim=True)
        
        combined_samples = torch.cat([current_samples, flat_data], dim=0).to(device)
        combined_labels = torch.cat([current_labels, labels], dim=0).to(device)
        num_samples = len(combined_samples)

        # 提取分类专家
        classifiers = [comp.classifier for comp in components]

        # 如果没有专家则使用滑动窗口
        if len(classifiers) == 0:
            print("无可用专家，回退到滑动窗口策略")
            num_new_samples = len(flat_data)
            memory.buffer = memory.buffer[num_new_samples:] + list(zip(flat_data.tolist(), labels.tolist()))
            return component

        # 高效计算所有分数
        with torch.no_grad():
            # ================== 1. 分类不确定性计算 ==================
            L_c_scores = torch.zeros(num_samples, device=device)

            # 准备批次输入(所有样本一次性处理)
            all_outputs = []
            for classifier in classifiers:
                classifier.eval()
                outputs = classifier(combined_samples)  # [N, C]
                all_outputs.append(outputs)

            # 并行计算所有专家模型的损失
            for i, classifier_out in enumerate(all_outputs):
                # 计算每个样本的交叉熵损失
                loss = nn.CrossEntropyLoss(reduction='none')(classifier_out, combined_labels)
                L_c_scores += loss

            # 平均各专家模型的损失
            L_c_scores /= len(classifiers)

        # ================== 2. 分数归一化与选择 ==================
        # 归一化分数
        L_c_scores_norm = (L_c_scores - L_c_scores.min()) / (L_c_scores.max() - L_c_scores.min() + 1e-8)

        # 计算综合分数
        S_scores = L_c_scores_norm

        # 选择分数最高的样本
        _, sorted_indices = torch.sort(S_scores, descending=True)
        selected_indices = sorted_indices[:memory.size]

        # 更新内存
        selected_samples = combined_samples[selected_indices].cpu()
        selected_labels = combined_labels[selected_indices].cpu()
        memory.buffer = list(zip(selected_samples.tolist(), selected_labels.tolist()))

        return component


def create_new_component(args, components):
    """
    创建一个新的组件并添加到组件列表中
    """
    # 根据数据集设置类别数
    if args.dataset.lower() == "cifar100":
        num_classes = 100  # CIFAR100有100个类别
    else:
        num_classes = 10  # 其他数据集默认10个类别
    
    component = Component(
        gan_z_dim=128,
        learning_rate=0.0001,
        beta1=0.5,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        input_channels=args.input_channels,  # 使用超参数指定的输入通道数
        num_classes=num_classes  # 添加类别数参数
    )
    components.append(component)
    print(f"已创建新组件。当前组件数量: {len(components)}")
    logging.info(f"已创建新组件。当前组件数量: {len(components)}")
    return component


def save_model_state(components, model_path, model_filename="model.pth"):
    """
    保存模型状态到指定路径

    :param components: 组件列表
    :param model_path: 保存模型的目录
    :param model_filename: 模型文件名，默认为 "model.pth"
    """
    # 确保目录存在
    os.makedirs(model_path, exist_ok=True)

    # 直接保存模型到指定路径
    model_filepath = os.path.join(model_path, model_filename)
    torch.save({
        f'component_{i + 1}': {
            'vae': comp.vae.state_dict(),
            'classifier': comp.classifier.state_dict()
        }
        for i, comp in enumerate(components)
    }, model_filepath)

    logging.info(f"模型已保存到 {model_filepath}")