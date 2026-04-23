# utils/memory.py
import random
import torch
import numpy as np
from torch.utils.data import Subset


class MemoryBuffer:
    """
    内存缓冲区，用于存储样本及其标签
    """

    def __init__(self, size=1000, input_channels=1):
        self.size = size
        self.buffer = []
        self.input_channels = input_channels  # 记录输入通道数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_samples(self, samples, labels):
        # 确保样本具有正确的通道数
        processed_samples = []
        for sample in samples:
            sample_tensor = torch.tensor(sample, dtype=torch.float32)
            if len(sample_tensor.shape) == 3:  # [C, H, W]
                if sample_tensor.shape[0] != self.input_channels:
                    if self.input_channels == 3 and sample_tensor.shape[0] == 1:
                        # 单通道转三通道
                        sample_tensor = sample_tensor.repeat(3, 1, 1)
                    elif self.input_channels == 1 and sample_tensor.shape[0] == 3:
                        # 三通道转单通道
                        sample_tensor = sample_tensor.mean(dim=0, keepdim=True)
            elif len(sample_tensor.shape) == 2:  # [H, W] - 单通道
                sample_tensor = sample_tensor.unsqueeze(0)  # 添加通道维度
                if self.input_channels == 3:
                    sample_tensor = sample_tensor.repeat(3, 1, 1)  # 单通道转三通道
            
            processed_samples.append(sample_tensor.numpy())
        
        combined = list(zip(processed_samples, labels))
        if len(self.buffer) + len(combined) > self.size:
            return False
        self.buffer.extend(combined)
        return True

    def get_samples(self, shuffle=True):
        """
        获取缓冲区中的所有样本
        :param shuffle: 是否打乱样本顺序 (默认为True)
        :return: (样本张量, 标签张量)
        """
        if len(self.buffer) == 0:
            # 根据输入通道数动态调整维度
            return (
                torch.empty(0, self.input_channels, 32, 32, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device)
            )

        # 如果需要打乱顺序
        if shuffle:
            shuffled_buffer = self.buffer[:]
            random.shuffle(shuffled_buffer)
            samples, labels = zip(*shuffled_buffer)
        else:
            samples, labels = zip(*self.buffer)

        # 转换为tensor
        samples_array = np.array(samples)
        samples = torch.tensor(samples_array, dtype=torch.float32, device=self.device)
        
        # 确保数据形状正确
        if len(samples.shape) == 4:  # [N, C, H, W] - 正确形状
            # 确保通道数正确
            if samples.shape[1] != self.input_channels:
                if self.input_channels == 3 and samples.shape[1] == 1:
                    # 单通道转三通道
                    samples = samples.repeat(1, 3, 1, 1)
                elif self.input_channels == 1 and samples.shape[1] == 3:
                    # 三通道转单通道
                    samples = samples.mean(dim=1, keepdim=True)
        elif len(samples.shape) == 3:  # [N, H, W] - 单通道情况
            samples = samples.unsqueeze(1)  # 添加通道维度 -> [N, 1, H, W]
            if self.input_channels == 3:
                # 单通道转三通道
                samples = samples.repeat(1, 3, 1, 1)
        
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        return samples, labels

    def update_samples(self, new_samples, new_labels):
        combined = list(zip(new_samples, new_labels))
        self.buffer = combined[:self.size]

    def __len__(self):
        return len(self.buffer)

    def get_class_distribution(self):
        """获取类别分布统计"""
        if not self.buffer:
            return {}
        _, labels = zip(*self.buffer)
        return dict(zip(*np.unique(labels, return_counts=True)))

    def get_statistics(self):
        """
        返回内存统计摘要
        """
        class_dist = self.get_class_distribution()
        total = len(self)

        stats = {
            "total_samples": total,
            "class_distribution": class_dist,
            "num_classes": len(class_dist),
            "min_samples": min(class_dist.values()) if class_dist else 0,
            "max_samples": max(class_dist.values()) if class_dist else 0,
            "diversity_index": len(class_dist) / total if total > 0 else 0
        }
        return stats

    def print_statistics(self):
        """
        打印内存统计摘要
        """
        stats = self.get_statistics()
        print("\n内存统计摘要:")
        print(f"总样本数: {stats['total_samples']}")
        print(f"类别数量: {stats['num_classes']}")
        print(f"最小类别样本数: {stats['min_samples']}")
        print(f"最大类别样本数: {stats['max_samples']}")
        print(f"多样性指数: {stats['diversity_index']:.4f}")
        if stats['class_distribution']:
            print("详细分布:")
            for cls, count in sorted(stats['class_distribution'].items()):
                print(f"  类别 {cls}: {count} 样本")


class PermutedMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset, perm):
        """
        base_dataset: 通常是 torchvision.datasets.MNIST 或 Subset(MNIST, indices_in_full)
        perm: 长度为 32*32 的 torch.LongTensor 或 torch.Tensor 索引
        """
        self.dataset = base_dataset
        self.perm = perm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]       # img: [1,32,32]
        img = img.view(-1)[self.perm]         # 展平并按 perm 重排（长度 1024）
        img = img.view(1, 32, 32)             # reshape 回 (1,32,32)
        return img, target