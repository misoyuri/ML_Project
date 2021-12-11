import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class CNN19(nn.Module):
    """
        Simple CNN Clssifier
    """
    def __init__(self, num_classes=7):
        super().__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        self.avg_pool = nn.AvgPool2d(3)
        
        self.fc = nn.Sequential(
            nn.Linear(192, 32),
        )
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        
        return x