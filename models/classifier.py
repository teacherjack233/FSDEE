# models/classifier.py
import torch
import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(Classifier, self).__init__()
        # 创建ResNet18模型
        self.model = models.resnet18(pretrained=False)
        
        # 修改第一个卷积层以适应不同的输入通道数
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            input_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # 修改最后的全连接层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class MNISTClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(MNISTClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 根据输入通道数和图像大小动态计算全连接层的输入维度
        feature_map_size = 64 * 8 * 8  # 假设经过两次池化后变成8x8
        self.classifier = nn.Sequential(
            nn.Linear(feature_map_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x