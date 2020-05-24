import torch.nn as nn
from model import utils
import itertools
import functools
import torch


class Unet(nn.Module, utils.Identifier):

    def __init__(self, block, inplanes, in_dim, out_dim, depth=4, norm_layer=None):
        super(Unet, self).__init__()

        self.depth = depth
        self.block = block
        self._norm_layer = norm_layer
        self.inplanes = inplanes
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.first_layer = block(in_dim, inplanes, norm_layer=norm_layer)
        self.down_layers = nn.ModuleList([self._make_down_layer(block, inplanes*2**i) for i in range(depth)])
        self.downsample = nn.MaxPool2d(2, 2)
        mid_planes = inplanes*2**depth
        self.mid_layer = self._make_mid_layer(block, mid_planes)
        self.up_layers = nn.ModuleList([self._make_up_layer(block, inplanes*2**i) for i in range(depth)])
        self.lateral_layers = nn.ModuleList([self._make_lateral_layer(block, inplanes*2**i) for i in range(depth)])
        self.out_layer = nn.Conv2d(inplanes, out_dim, 1)
           
    def _make_down_layer(self, block, planes):
        return block(planes, planes*2, norm_layer=self._norm_layer)

    def _make_mid_layer(self, block, planes):
        return block(planes, planes, norm_layer=self._norm_layer)

    def _make_up_layer(self, block, planes):
        return nn.ConvTranspose2d(planes*2, planes, 2, 2)
    
    def _make_lateral_layer(self, block, planes):
        return block(planes*2, planes, norm_layer=self._norm_layer)

    def forward(self, x):
        x = self.first_layer(x)
        
        outputs = [x]
        for l in self.down_layers:
            x = self.downsample(x)
            x = l(x)
            outputs.append(x)
    
        x = self.mid_layer(x)

        for i in range(self.depth-1, -1, -1):
            x = self.up_layers[i](x)
            x = torch.cat((x, outputs[i]), 1)
            x = self.lateral_layers[i](x)

        x = self.out_layer(x) 
        return x