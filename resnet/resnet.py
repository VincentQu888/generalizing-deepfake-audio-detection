import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks=3, channels=32):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=channels, kernel_size=3, padding=1 
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList(
            [ResNetBlock(channels) for _ in range(blocks)]
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
        self.fc1 = nn.Linear(channels, channels // 2)
        self.fc2 = nn.Linear(channels // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        conv_out = F.relu(x)

        for block in self.blocks:
            conv_out = block(conv_out)

        out = self.global_pool(conv_out) # [B, C, 1, 1]
        out = out.view(out.size(0), -1) # flatten to [B, C]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out