import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import glob
import numpy as np
import imageio
import itertools
import foolbox as fb
import foolbox.ext.native as fbn
import time
import torchvision


    
class Netsreb(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3):
        super(Netsreb, self).__init__()
        self.kernel = kernel
        self.channels = channels
        self.conv = nn.Conv2d(3,self.channels,self.kernel,1) 
        self.pad = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.linear = nn.Linear(self.channels*(32 - self.kernel + 1 + self.pad*2)*(32 - self.kernel + self.pad*2 + 1),10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.nn.functional.pad(x,(self.pad,self.pad,self.pad,self.pad),'circular')
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

class Net3sreb(nn.Module):
    def __init__(self, classes=10, kernel = 3, channels=3):
        super(Net3sreb, self).__init__()
        self.kernel = kernel
        self.channels = channels
        
        self.conv1 = nn.Conv2d(3,self.channels,self.kernel,1) 
        self.pad1 = math.ceil((32 - (32 - self.kernel + 1))/2)
        self.output1 = int((32 - self.kernel + 1 + self.pad1*2))
        
        self.conv2 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad2 = math.ceil((self.output1 - (self.output1 - self.kernel + 1))/2)
        self.output2 = int((self.output1 - self.kernel + 1 + self.pad2*2))
        
        self.conv3 = nn.Conv2d(self.channels, self.channels, self.kernel,1) 
        self.pad3 = math.ceil((self.output2 - (self.output2 - self.kernel + 1))/2)
        self.output3 = int((self.output2 - self.kernel + 1 + self.pad3*2))
        
        self.linear = nn.Linear(self.channels*(self.output3 - self.kernel + 1 + self.pad3*2)*(self.output3 - self.kernel + self.pad3*2 + 1),10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.nn.functional.pad(x,(self.pad1,self.pad1,self.pad1,self.pad1),'circular')
        x = self.conv1(x)
        x = torch.nn.functional.pad(x,(self.pad2,self.pad2,self.pad2,self.pad2),'circular')
        x = self.conv2(x)
        x = torch.nn.functional.pad(x,(self.pad3,self.pad3,self.pad3,self.pad3),'circular')
        x = self.conv3(x)
        x = self.flat(x)
        x = self.linear(x)
        return x
    
    
    
    
    
    
    
    
    

class Net(nn.Module):
    def __init__(self, classes=10,prop=0.2):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3,32,3,1)
        self.linear = nn.Linear(30*30*32,10)
        self.flat = nn.Flatten()
        self.prop = prop

    def forward(self, x):
        x = self.conv(x)
        x, x1 = torch.split(x, [x.shape[1] - int(x.shape[1]*self.prop),int(x.shape[1] *self.prop)], dim=1)
        x1 = torch.nn.functional.relu_(x1)
        x = torch.cat([x,x1],axis=1)
        x = self.flat(x)
        x = self.linear(x)
        return x

    
class Net4(nn.Module):
    def __init__(self, classes=10,prop=0.2):
        super(Net4, self).__init__()
        self.conv = nn.Conv2d(3,3,32,1)
        self.linear = nn.Linear(33*33*3,10)
        self.flat = nn.Flatten()
        self.prop = prop

    def forward(self, x):
        x = torch.nn.functional.pad(x,(16,16,16,16),'circular')
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

class Net5(nn.Module):
    def __init__(self, classes=10,prop=0.2):
        super(Net5, self).__init__()
        self.conv = nn.Conv2d(3,3,32,1)
        self.linear = nn.Linear(33*33*3,10)
        self.flat = nn.Flatten()
        self.prop = prop

    def forward(self, x):
        x = torch.nn.functional.pad(x,(16,16,16,16),'circular')
        x = self.conv(x)
        x = torch.nn.functional.relu_(x)
        x = self.flat(x)
        x = self.linear(x)
        return x


class Net3(nn.Module):
    def __init__(self, classes=10,prop=0.2):
        super(Net3, self).__init__()
        self.conv = nn.Conv2d(3,32,3,1,padding=1)
        self.conv1 = nn.Conv2d(32,32,3,1,padding=1)
        self.linear = nn.Linear(30*30*32,10)
        self.flat = nn.Flatten()
        self.prop = prop

    def forward(self, x):
        x1 = self.conv(x)
        x = self.conv1(x1)
        x = torch.nn.functional.relu_(x)
        x = x1 + x
        x = self.flat(x)
        x = self.linear(x)
        return x
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        #torch.nn.init.xavier_normal_(self.weight)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
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

class Net2(nn.Module):
    def __init__(self, classes=10, kernel_size=7):
        super(Net2, self).__init__()
        self.features = int((32*30*30)/((32 - kernel_size + 1) *(32 - kernel_size + 1)))
        self.lc = LocallyConnected2d(3,self.features,32 - kernel_size +1, kernel_size,1,bias=True)
        self.linear = nn.Linear((32 - kernel_size + 1)*(32 - kernel_size + 1)*self.features,10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.lc(x)
        x = torch.nn.functional.relu_(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])


'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

def return_model(name='resner18'):
    if name == 'linear':
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,10))
    elif name == 'simplelinear':
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
    elif name == 'simple':
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1), torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
    elif name == 'fc1000':
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,1000),torch.nn.ReLU(), torch.nn.Linear(1000,10))
    elif name == 'fclinear':
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,1000),torch.nn.ReLU(), torch.nn.Linear(1000,10))
    elif name == 'resnet':
        model = ResNet18()
    elif name == 'vgg':
        model = vgg19_bn()
    elif name == 'lc':
        model = nn.Sequential(LocallyConnected2d(3,32,30,3,1,bias=True),nn.ReLU(), nn.Flatten(), nn.Linear(30*30*32,10))
    elif name == 'lck7':
        model = Net2(kernel_size=7)
    elif name == 'lck9':
        model = Net2(kernel_size=9)
    elif name == 'lck15':
        model = Net2(kernel_size=15)
    elif name == 'convresnet':
        model = Net3()
    elif name == 'circular': # which is the linear version...
        model = Net4()
    elif name == 'circularlinear': # which is the nonlinear version??
        model = Net5()
    elif name == 'simplea10':
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1), torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]*=10.0
            model[0].bias[:]*=10.0
            model[3].weight[:]/=10.0
    elif name == 'simplea3': # was simplehigha
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1), torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]*=3.0
            model[0].bias[:]*=3.0
            model[3].weight[:]/=3.0
    elif name == 'simpleapoint33': # was simplelowa
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1), torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]/=3.0
            model[0].bias[:]/=3.0
            model[3].weight[:]*=3.0
    elif name == 'simpleapoint1':
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1), torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]/=10.0
            model[0].bias[:]/=10.0
            model[3].weight[:]*=10.0
    elif name == 'simplelineara10':
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]*=10.0
            model[0].bias[:]*=10.0
            model[2].weight[:]/=10.0
    elif name == 'simplelineara3': 
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]*=3.0
            model[0].bias[:]*=3.0
            model[2].weight[:]/=3.0
    elif name == 'simplelinearapoint33': 
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]/=3.0
            model[0].bias[:]/=3.0
            model[2].weight[:]*=3.0
    elif name == 'simplelinearapoint1':
        model = torch.nn.Sequential(torch.nn.Conv2d(3,32,3,1),torch.nn.Flatten(),torch.nn.Linear(32*30*30,10))
        with torch.no_grad():
            model[0].weight[:]/=10.0
            model[0].bias[:]/=10.0
            model[2].weight[:]*=10.0
    elif name == 'lca10':
        model = nn.Sequential(LocallyConnected2d(3,32,30,3,1,bias=True),nn.ReLU(), nn.Flatten(), nn.Linear(30*30*32,10))
        with torch.no_grad():
            model[0].weight[:]*=10.0
            model[0].bias[:]*=10.0
            model[3].weight[:]/=10.0
    elif name == 'lcapoint1':
        model = nn.Sequential(LocallyConnected2d(3,32,30,3,1,bias=True),nn.ReLU(), nn.Flatten(), nn.Linear(30*30*32,10))
        with torch.no_grad():
            model[0].weight[:]/=10.0
            model[0].bias[:]/=10.0
            model[3].weight[:]*=10.0
    elif name == 'lclineara10':
        model = nn.Sequential(LocallyConnected2d(3,32,30,3,1,bias=True), nn.Flatten(), nn.Linear(30*30*32,10))
        with torch.no_grad():
            model[0].weight[:]*=10.0
            model[0].bias[:]*=10.0
            model[2].weight[:]/=10.0
    elif name == 'lclinearapoint1':
        model = nn.Sequential(LocallyConnected2d(3,32,30,3,1,bias=True), nn.Flatten(), nn.Linear(30*30*32,10))
        with torch.no_grad():
            model[0].weight[:]/=10.0
            model[0].bias[:]/=10.0
            model[2].weight[:]*=10.0
    elif name == 'lclinear':
        model = nn.Sequential(LocallyConnected2d(3,32,30,3,1,bias=True), nn.Flatten(), nn.Linear(30*30*32,10))
        
    # New sreb stuff
    elif name == 'fclinearl3':
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32,3*32*32),nn.Linear(3*32*32,3*32*32), nn.Linear(3*32*32,3*32*32), nn.Linear(3*32*32,10))
    elif name == 'fclinearl1':
        model = nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,3*32*32), nn.Linear(3*32*32,10))
    elif name == 'convlinearl1k3c3':
        model = Netsreb(kernel=3,channels=3)
    elif name == 'convlinearl3k3c3':
        model = Net3sreb(kernel=3,channels=3)
    elif name == 'convlinearl1k11c3':
        model = Netsreb(kernel=11,channels=3)
    elif name == 'convlinearl3k11c3':
        model = Net3sreb(kernel=11, channels=3)
    elif name == 'convlinearl1k32c3':
        model = Netsreb(kernel=32, channels=3)
    elif name == 'convlinearl3k32c3':
        model = Net3sreb(kernel=32, channels=3)
    elif name == 'convlinearl1k3c8':
        model = Netsreb(kernel=3,channels=8)
    elif name == 'convlinearl1k3c32':
        model = Netsreb(kernel=3, channels=32)
    elif name == 'convlinearl3k3c8':
        model = Net3sreb(kernel=3, channels=8)
    elif name == 'convlinearl3k3c32':
        model = Net3sreb(kernel=3, channels=32)
    return model



posible_models = ['fclinearl3', 'fclinearl1','convlinearl1k3c3', 'convlinearl3k3c3','convlinearl1k11c3','convlinearl3k11c3','convlinearl1k32c3',
                 'convlinearl3k32c3','convlinearl1k3c8','convlinearl1k3c32','convlinearl3k3c8','convlinearl3k3c32']
