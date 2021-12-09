import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class ResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.in_dim = 64
        self.mid_dim = 16
        self.out_dim = 64
        
        # 1 48 48
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.in_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 64 24 24
        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.mid_dim, kernel_size=1, bias=False),
            nn.ReLU(), 
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.mid_dim, self.in_dim, kernel_size=1, bias=False)
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 64 12 12
        self.block2 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.mid_dim, kernel_size=1, bias=False),
            nn.ReLU(), 
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.mid_dim, self.in_dim, kernel_size=1, bias=False)
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 64 3 3
        self.avg_pool = nn.AvgPool2d(3)
        
        self.ReLU = nn.ReLU()
        # 64 1 1
        self.classifier = nn.Sequential(
            nn.Linear(64, 32)   
        )

        self.pool = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.pool(self.block1(output) + self.shortcut(output))
        output = self.pool(self.block2(output) + self.shortcut(output))
        output = self.avg_pool(output)
        
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        output = F.log_softmax(output , dim=1)
        
        return output