
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEncoding(nn.Module):
    """时间傅里叶编码, SDE更一般形式，有限时间不包括,高斯随机特征编码"""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat((torch.sin(x_proj), torch.cos(x_proj)), dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshape outputs to feature maps"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """基于Unet的时间依赖的分数估计模型"""

    def __init__(self, marginal_prob_std, channels=[64, 128, 256, 512], embed_dim=256):
        """ Initialize a time-dependant score-based network
        Args:
        Parameters
        ----------
        marginal_prob_std: A function that takes time t and gives the standard deviation of the perturbation kernel_p{x0}(x(t)|x(0))
        channels: The number of channels for feature maps of each resolution
        embed_dim: The dimensionality of Gaussian random features embeddings
        """
        super().__init__()
        self.embed = nn.Sequential(TimeEncoding(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.embed_linear = nn.Linear(embed_dim, embed_dim)

        # Encoding part
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding part
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=3, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.act = nn.ReLU(inplace=True)
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 10, 3, stride=1)

        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        time_embed = self.embed(t)
        time_embed = self.embed_linear(time_embed)

        # Encoding part
        h1 = self.conv1(x)
        h1 += self.dense1(time_embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(time_embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(time_embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(time_embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding part
        h = self.tconv4(h4)
        h += self.dense5(time_embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        h = self.tconv3(torch.cat((h, h3), dim=1))
        h += self.dense6(time_embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        h = self.tconv2(torch.cat((h, h2), dim=1))
        h += self.dense7(time_embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        h = self.tconv1(torch.cat((h, h1), dim=1))
        h = h.mean(dim=(2, 3))
        h = F.softmax(h, dim=1)
        return h
