import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pad_flag=False, downsample=False):
        super().__init__()
        self.pad_flag = pad_flag
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 if pad_flag else 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1 if pad_flag else 0)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if not self.pad_flag:
            self.natural_pool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.downsample_flag = downsample
        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pad_flag:
            identity = x
        else:
            identity = self.natural_pool(self.natural_pool(x))
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.gelu(out)
        if self.downsample_flag:
            residual = self.maxpool(self.downsample_conv(residual))
            out = self.maxpool(out)
        return out

class A2Cv2_0312(nn.Module):
    def __init__(self, in_channels=14, out_channels=512, num_layers=10, num_action=5):
        super().__init__()
        layers = []
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        for il in range(num_layers // 2): # changed to use ResidualBlock, so num_layers is roughly halved
            # downsample = (il%3==0 and il>0) # Keep the same downsample logic at the 5th layer
            if il < 1:
                pad_flag=True
            else:
                pad_flag=False
            downsample = False
            layers.append(ResidualBlock(out_channels, out_channels, pad_flag=pad_flag, downsample=downsample))
            
        self.encoder = nn.Sequential(*layers)
        self.gelu = nn.GELU()

        self.fc = nn.Linear(12, 128)
        self.bn_fc = nn.BatchNorm1d(128)  # 添加批归一化

        # 特征融合
        self.fc2 = nn.Linear(out_channels + 128, out_channels//2)
        self.bn_fc2 = nn.BatchNorm1d(out_channels//2)  # 添加批归一化
        self.fc3 = nn.Linear(out_channels//2, out_channels//2)
        self.dropout = nn.Dropout(0.1)

        # act layer
        self.act_layer = nn.Linear(out_channels//2, num_action)
        # value layer
        self.value_layer = nn.Linear(out_channels//2, 1)

    def forward(self, x, x1):
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(3).squeeze(2)
        x1 = self.gelu(self.bn_fc(self.fc(x1)))

        cbn = torch.cat([x, x1], dim=1)
        y = self.gelu(self.bn_fc2(self.fc2(cbn)))
        y = self.dropout(y)
        y = self.gelu(self.fc3(y))
        
        act = self.act_layer(y)
        val = self.value_layer(y)
        return act, val
