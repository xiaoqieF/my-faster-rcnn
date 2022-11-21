import torch
from torch import nn

class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU6, self).__init__(
            # when in_channel == out_channel == groups, it becomes Depthwise Convolution
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    """
    1x1 conv -> expand num of channels  (relu6)
    3x3 depthwise conv                  (relu6)
    1x1 conv -> squeeze num of channels (linear)
    """
    def __init__(self, in_channel, out_channel, stride, expand_ratio) -> None:
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channel, hidden_channel, kernel_size=1))
        
        layers.extend([
            # 3x3 depthwise convolution
            ConvBNReLU6(hidden_channel, hidden_channel, kernel_size=3, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, weights_path=None) -> None:
        super().__init__()
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # expand_ratio, output_channels, block_repeat_times, stride(only first block)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLU6(3, input_channel, kernel_size=3, stride=2))
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                # only first block stride is s
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, out_channel=c, stride=stride, expand_ratio=t))
                input_channel = c
        features.append(ConvBNReLU6(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        if weights_path is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            self.load_state_dict(torch.load(weights_path))
    
    def forward(self, x):
        x = self.features(x)
        x = self.avepool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x