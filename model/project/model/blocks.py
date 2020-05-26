import torch.nn as nn
from model import utils


class PreActivationBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d):
        super(PreActivationBlock, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.stride = stride

        self.sequential = nn.Sequential(
            norm_layer(inplanes),
            nn.ReLU(inplace=True),
            utils.conv3x3(inplanes, planes, stride),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            utils.conv3x3(planes, planes)
        )

    def forward(self, x):
        return self.sequential(x)


class InvertedBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, expand_ratio=6):
        super(InvertedBlock, self).__init__()
        self.expanded_dim = inplanes * expand_ratio

        self.sequential = nn.Sequential(
            utils.conv1x1(inplanes, self.expanded_dim),
            norm_layer(self.expanded_dim),
            nn.ReLU6(inplace=True),

            utils.conv3x3(self.expanded_dim,  self.expanded_dim, stride, groups=self.expanded_dim),
            norm_layer(self.expanded_dim),
            nn.ReLU6(inplace=True),

            utils.conv1x1(self.expanded_dim, planes),
            norm_layer(planes)
        )

    def forward(self, x):
        return self.sequential(x)


class SqueezeExcitationLayer(nn.Module, utils.Identifier):

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sequential(x)
        return x * out


class SqueezeExcitationBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, norm_layer=nn.InstanceNorm2d, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()

        self.sequential = nn.Sequential(
            ConvBlock(inplanes, planes, norm_layer),
            SqueezeExcitationLayer(planes, reduction)
        )

    def forward(self, x):
        return self.sequential(x)


class EfficientNetBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.InstanceNorm2d, reduction=16, expand_ratio=6):
        super(EfficientNetBlock, self).__init__()

        self.sequential = nn.Sequential(
            InvertedBlock(inplanes, planes, stride, norm_layer, expand_ratio=expand_ratio),
            SqueezeExcitationLayer(planes, reduction),
        )

    def forward(self, x):
        return self.sequential(x)

class ConvBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes, norm_layer=None):
        super(ConvBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, padding=1, bias=norm_layer is not None))
        if norm_layer is not None:
            layers.append(norm_layer(planes))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(planes, planes, 3, padding=1, bias=norm_layer is not None))
        if norm_layer is not None:
            layers.append(norm_layer(planes))
        layers.append(nn.ReLU(True))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

class UpConvBlock(nn.Module, utils.Identifier):

    def __init__(self, inplanes, planes):
        super(UpConvBlock, self).__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(inplanes, planes, 2, 2, output_padding=1, bias=True))
        layers.append(nn.ReLU(True))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)