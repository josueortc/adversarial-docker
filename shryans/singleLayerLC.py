import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding=1, bias=False, conv_init = True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.conv_init = conv_init
        if conv_init == True:
            useless = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False) ## init a std layer which is useless
            self.weight = torch.stack([useless.weight.data for i in range((output_size[0])*(output_size[1]))], axis=2) ##  stacking it with respect to size
            self.weight = self.weight.reshape(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
            self.weight = nn.Parameter(self.weight)
        else:
            self.weight = nn.Parameter(
                torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)) ## In std model you init like this
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0])
            )
            self.bias = torch.stack([self.bias for i in range(output_size[1])],axis=3)
        else:
            self.register_parameter('bias', None)
        self.padding = padding


        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out



class Model():
    def __init__(self, in_planes, planes, stride=1, input_size=32):
        # super(ResNet, self).__init__()
        self.in_planes = 64

        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) ## we can change this to locllyconnected
        self.lc1 = LocallyConnected2D(in_planes, planes, int(input_size/stride), 3, stride, padding=1, conv_init=True)
        self.bn1 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.lc1(x)))
        return out


def returnModel():
    return Model(64, 64)