import torch
from torch import nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        in_channels: channel of input feature map
        out_channels: channel of middle feature map
        outputs channels of Bottleneck is expansion * out_channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, 
            kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super().__init__()
        self.in_channels = 64
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # only first block of layer use downsample
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channel, downsample=downsample, stride=stride))
        self.in_channels = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def resnet50(weight_path="", trainable_layers=3):
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3], include_top=False)

    if weight_path != "":
        print(resnet_backbone.load_state_dict(torch.load(weight_path), strict=False))
    
    assert 0 <= trainable_layers <= 5

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # freeze layers
    for name, parameter in resnet_backbone.named_parameters():
        # 只训练在layers_to_train列表中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    resnet_backbone.out_channels = 2048

    return resnet_backbone

def resnet50_fpn(weight_path="", trainable_layers=4):
    resnet_backbone = resnet50(weight_path, trainable_layers)
    returned_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3',
    }
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, returned_layers, in_channels_list, out_channels)