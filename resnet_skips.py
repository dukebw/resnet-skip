# Copyright 2018 Brendan Duke.
#
# This file is part of Resnet Skips.
#
# Resnet Skips is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Resnet Skips is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Resnet Skips. If not, see <http://www.gnu.org/licenses/>.

import math

import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18
from torchvision.models.resnet import model_urls


class ResNet18Canned(torch.nn.Module):
    """Directly use the resnet18 from torchvision.models as encoder.
    """

    def __init__(self, num_classes=2):
        torch.nn.Module.__init__(self)

        self.enc = resnet18(pretrained=True)
        del self.enc.fc
        del self.enc.avgpool

        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.dec4 = torch.nn.Sequential(
            # NOTE(brendan): 2*64 since we will concatenate with the output of
            # layer1.
            torch.nn.ConvTranspose2d(in_channels=2*64,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        )

        self.dec5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32,
                                     out_channels=16,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
        )

        self.logits = torch.nn.Conv2d(16, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.enc.conv1(x)
        x = self.enc.bn1(x)
        x = self.enc.relu(x)
        x = self.enc.maxpool(x)

        x = self.enc.layer1(x)
        x_skip = x

        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        x = self.enc.layer4(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        x = torch.cat([x_skip, x], dim=1)

        x = self.dec4(x)
        x = self.dec5(x)

        return self.logits(x)


# NOTE(brendan): Below is copy pasted from torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18CopyPasta(torch.nn.Module):
    def __init__(self, num_classes=2):
        torch.nn.Module.__init__(self)

        self.inplanes = 64
        layers = [2, 2, 2, 2]
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512,
                                     out_channels=256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.dec4 = torch.nn.Sequential(
            # NOTE(brendan): 2*64 since we will concatenate with the output of
            # layer1.
            torch.nn.ConvTranspose2d(in_channels=2*64,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        )

        self.dec5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32,
                                     out_channels=16,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
        )

        self.logits = torch.nn.Conv2d(16, num_classes, 3, padding=1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x_skip = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        x = torch.cat([x_skip, x], dim=1)

        x = self.dec4(x)
        x = self.dec5(x)

        return self.logits(x)


def resnet_skips():
    rando_input = torch.randn([1, 3, 224, 224]).cuda()
    model = ResNet18Canned().cuda()

    # NOTE(brendan): this is just to print the output shapes of the layer1-4
    # and dec 1-5 layers. You can ignore it.
    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            module.register_forward_hook(lambda m, inp, out: print(out.shape))

    y = model(rando_input)

    print()
    model = ResNet18CopyPasta().cuda()
    # NOTE(brendan): this will load the matching layers, but only because they
    # are named the exact same. Be careful with strict=False!
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            module.register_forward_hook(lambda m, inp, out: print(out.shape))

    y = model(rando_input)


if __name__ == '__main__':
    resnet_skips()
