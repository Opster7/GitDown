import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from sdenoise import marginal_prob_std, marginal_prob_std_fn, device
from Unet import ScoreNet
# from Unet_deeper import ScoreNet

class Classifier(nn.Module):
    def __init__(self, score_net, num_classes):
        super(Classifier, self).__init__()
        self.score_net = score_net  # 使用基于U-Net的生成式模型作为特征提取器
        self.fc = nn.Linear(in_features=3, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, t):
        # 使用基于U-Net的生成式模型提取特征
        features = self.score_net(x, t)
        # 全局平均池化或展平特征图
        features = torch.mean(features, dim=[2, 3])
        # 通过全连接层得到分类结果
        logits = self.fc(features)
        # 使用Softmax激活函数输出类别概率
        probabilities = self.softmax(logits)

        return probabilities
#
# cifar100
# class Classifier(nn.Module):
#     def __init__(self, score_net, num_classes):
#         super(Classifier, self).__init__()
#         self.score_net = score_net
#         self.feature_layers = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc = nn.Linear(in_features=128, out_features=num_classes)
#
#     def forward(self, x, t):
#         features = self.score_net(x, t)
#         features = self.feature_layers(features)
#         features = torch.mean(features, dim=[2, 3])
#         logits = self.fc(features)
#         return logits  # 注意，我们返回logits，而不是概率

# Unet_Deeper
# class Classifier(nn.Module):
#     def __init__(self, score_net, num_classes):
#         super(Classifier, self).__init__()
#         self.score_net = score_net
#         self.fc = nn.Linear(in_features=1024, out_features=num_classes)  # 确保这里的输入特征数与 ScoreNet 的输出匹配
#
#     def forward(self, x, t):
#         features = self.score_net(x, t)
#         features = torch.mean(features, dim=[2, 3])  # Global average pooling
#         logits = self.fc(features)
#         return logits

# Domainnet
# class Classifier(nn.Module):
#     def __init__(self, score_net, num_classes):
#         super(Classifier, self).__init__()
#         self.score_net = score_net  # U-Net-based generative model as feature extractor
#         self.fc = nn.Linear(in_features=1024, out_features=num_classes)  # Adjust the in_features accordingly
#
#     def forward(self, x, t):
#         # Extract features using U-Net-based generative model
#         features = self.score_net(x, t)
#         # Global average pooling to flatten the feature maps
#         features = torch.mean(features, dim=[2, 3])
#         # Obtain classification logits
#         logits = self.fc(features)
#
#         # No softmax here if you are using nn.CrossEntropyLoss
#         return logits
