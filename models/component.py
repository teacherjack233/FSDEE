# models/component.py
import torch
import torch.nn as nn
import torch.optim as optim
from .vae_esvae import SpikingESVAE
from .classifier import MNISTClassifier


class Component:
    """
    组件类，包含VAE和分类器及其优化器
    """
    def __init__(self, 
                 gan_z_dim=128,
                 learning_rate=0.003, 
                 beta1=0.5, 
                 batch_size=512,
                 n_steps=8,
                 input_channels=1,
                 num_classes=10):  # 新增num_classes参数
        self.n_steps = n_steps
        self.input_channels = input_channels  # 记录输入通道数
        self.num_classes = num_classes  # 记录类别数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化VAE，根据输入通道数调整
        self.vae = SpikingESVAE(in_channels=input_channels, n_steps=n_steps).to(self.device)
        params = list(self.vae.named_parameters())
        sample_layer_lr_times = 10
        param_group = [
            {'params': [p for n, p in params if 'sample_layer' in n], 'weight_decay': 0.001,
             'lr': learning_rate * sample_layer_lr_times},
            {'params': [p for n, p in params if 'sample_layer' not in n], 'weight_decay': 0.001, 'lr': learning_rate},
        ]
        self.vae_optimizer = torch.optim.AdamW(param_group,
                                              lr=learning_rate,
                                              betas=(0.9, 0.999),
                                              weight_decay=0.001)

        # 初始化分类器，根据输入通道数和类别数调整
        self.classifier = MNISTClassifier(input_channels=input_channels, num_classes=num_classes).to(self.device)
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), 
                                             lr=learning_rate*10, 
                                             betas=(beta1, 0.999))

        self.batch_size = batch_size
        self.z_dim = gan_z_dim

    def train_vae(self, x):
        """
        训练VAE
        """
        # 根据输入通道数调整形状
        if self.input_channels == 1:
            # 确保x是单通道图像
            if x.shape[1] == 3:  # 如果意外收到3通道数据，转为单通道
                x = x.mean(dim=1, keepdim=True)  # 平均RGB通道为灰度
            x = x.reshape([x.size(0), 1, 32, 32])
        elif self.input_channels == 3:
            # 确保x是3通道图像
            if x.shape[1] == 1:  # 如果意外收到单通道数据，复制到3通道
                x = x.repeat(1, 3, 1, 1)
            x = x.reshape([x.size(0), 3, 32, 32])
        else:
            # 对于其他通道数，需要相应调整形状
            x = x.reshape([x.size(0), self.input_channels, 32, 32])
            
        spike_input = x.unsqueeze(-1).repeat(1, 1, 1, 1, self.n_steps)
        
        x_recon, q_z, p_z, sampled_z = self.vae(spike_input, scheduled=True)
        loss = self.vae.loss_function_mmd(x, x_recon, q_z, p_z)
        self.vae_optimizer.zero_grad()
        loss["loss"].backward()
        self.vae_optimizer.step()
        return loss["loss"].item()

    def get_mmd_loss(self, x):
        return self.vae.get_check_mmd_loss(x)

    def train_classifier(self, x, labels):
        """
        训练分类器
        """
        self.classifier.train()
        # 根据输入通道数调整形状
        if self.input_channels == 1:
            # 确保x是单通道图像
            if x.shape[1] == 3:  # 如果意外收到3通道数据，转为单通道
                x = x.mean(dim=1, keepdim=True)  # 平均RGB通道为灰度
            x = x.reshape([x.size(0), 1, 32, 32])
        elif self.input_channels == 3:
            # 确保x是3通道图像
            if x.shape[1] == 1:  # 如果意外收到单通道数据，复制到3通道
                x = x.repeat(1, 3, 1, 1)
            x = x.reshape([x.size(0), 3, 32, 32])
        else:
            # 对于其他通道数，需要相应调整形状
            x = x.reshape([x.size(0), self.input_channels, 32, 32])
            
        outputs = self.classifier(x)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.classifier_optimizer.step()
        return loss.item()

    def test_classifier(self, x):
        """
        测试分类器
        """
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(x)
            predicted = torch.argmax(outputs, dim=1)
        return predicted

    def to(self, device):
        """移动模型到指定设备"""
        self.vae = self.vae.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device
        return self

    def eval(self):
        """设置为评估模式"""
        self.vae.eval()
        self.classifier.eval()

    def train(self):
        """设置为训练模式"""
        self.vae.train()
        self.classifier.train()