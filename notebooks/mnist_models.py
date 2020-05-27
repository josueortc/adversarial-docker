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



class Net(nn.Module):
    def __init__(self, classes=2):
        super(Net, self).__init__()
        #self.core = nn.Sequential(nn.Conv2d(1,32,3,1), nn.ReLU(), nn.Conv2d(32,64,3,1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(9216,128), 
        #                          nn.ReLU(), nn.Linear(128,2))
        self.core = nn.Sequential(nn.Flatten(), nn.Linear(784,classes))
        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.core(x)
        output = F.log_softmax(x, dim=1)
        return output



def return_model(name='full'):
    if name == 'linear':
        model = Net()
    elif name == 'simplelinear':
        model = torch.nn.Sequential(torch.nn.Conv2d(1,32,3,1),torch.nn.Flatten(),torch.nn.Linear(26*26*32,10))
    elif name == 'simple':
        model = torch.nn.Sequential(torch.nn.Conv2d(1,32,3,1), torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(32*26*26,10))
    elif name == 'fc1000':
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(1*28*28,1000),torch.nn.ReLU(), torch.nn.Linear(1000,10))
    elif name == 'fclinear':
        model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(1*28*28,1000), torch.nn.Linear(1000,10))
    elif model == 'full':
        model = nn.Sequential(nn.Conv2d(1,32,3,1), nn.ReLU(), nn.Conv2d(32,64,3,1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(9216,128), nn.ReLU(), nn.Linear(128,10))
    return model