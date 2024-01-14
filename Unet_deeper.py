import torch
import torch.nn as nn
import numpy as np

# 时间傅里叶编码
class TimeEncoding(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat((torch.sin(x_proj), torch.cos(x_proj)), dim=-1)

# 扩维MLP
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]

# 基于Unet的时间依赖的分数估计模型
class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[128, 256, 512, 1024], embed_dim=1024):
        super().__init__()
        self.embed = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))

        # Unet的编码器部分
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])  # 输出通道数与 conv1 层相同
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])  # 输出通道数与 conv2 层相同
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])  # 输出通道数与 conv3 层相同
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])  # 输出通道数与 conv4 层相同
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Unet解码器
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])  # 输出通道数与 tconv4 层相同
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] * 2, channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])  # 输出通道数与 tconv3 层相同
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] * 2, channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])  # 输出通道数与 tconv2 层相同
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        # Example modification in ScoreNet
        # Adjusting the last convolutional layer of ScoreNet to output 1024 channels
        self.tconv1 = nn.ConvTranspose2d(channels[0] * 2, 1024, 3, stride=1, padding=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # 解码器部分
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat((h, h3), dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat((h, h2), dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat((h, h1), dim=1))
        h = h / self.marginal_prob_std(t)[:, None, None, None]

        return h
