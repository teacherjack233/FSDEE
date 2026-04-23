# config/config.py
import argparse
from datetime import datetime
import os


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="VAE-GAN PyTorch 实现")

    def parse_args(self):
        self.parser.add_argument("--batch_size", type=int, default=64, help="内存缓冲区大小")
        self.parser.add_argument("--gpu", type=str, default='0', help="使用的GPU编号")
        self.parser.add_argument("--n_epochs", type=int, default=20, help="每个组件的训练轮数")
        self.parser.add_argument("--strategy", type=str, default="sliding_window",
                                 choices=["sliding_window", "diversity"],
                                 help="样本选择策略，'sliding_window' 或 'diversity'")
        self.parser.add_argument("--threshold", type=float, default=0.15,
                                 help="扩展组件时的阈值")
        self.parser.add_argument("--n_steps", type=int, default=8, help="时间步数")
        self.parser.add_argument("--dataset_fraction", type=float, default=0.5,
                                 help="数据集采样比例")
        self.parser.add_argument("--num_tasks", type=int, default=5, help="任务数量")
        self.parser.add_argument("--num_samples", type=int, default=64, help="采样数量")
        self.parser.add_argument("--save_dir", type=str, default="results", help="保存目录")
        self.parser.add_argument("--dataset", type=str, default="mnist",
                                 choices=["mnist", "cifar10", "cifar100", "permuted_mnist"],
                                 help="选择数据集: mnist, cifar10, cifar100, permuted_mnist")
        self.parser.add_argument("--input_channels", type=int, default=1,
                                 help="输入数据通道数，MNIST为1，CIFAR10/CIFAR100为3")
        self.parser.add_argument("--img_size", type=int, default=32,
                                 help="输入图像尺寸")
        self.parser.add_argument("--memory_size", type=int, default=200,
                                 help="内存缓冲区大小")

        args = self.parser.parse_args()
        return args

    def generate_log_dir(self, args, dataset_name):
        """生成日志目录"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        param_dir = (
            f"{dataset_name}_"
            f"bs{args.batch_size}_"
            f"ep{args.n_epochs}_"
            f"{args.strategy}_"
            f"th{args.threshold:.3f}_"
            f"gpu{args.gpu}"
        )

        day_dir = os.path.join(args.save_dir, current_date)
        os.makedirs(day_dir, exist_ok=True)

        time_dir = os.path.join(day_dir, f"{current_time}_{param_dir}")
        os.makedirs(time_dir, exist_ok=True)

        return time_dir