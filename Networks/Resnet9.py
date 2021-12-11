from __future__ import print_function
import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super().__init__()
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.converter = transforms.ToTensor()
        
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):          # 1 48 48
        
        out = self.conv1(xb)        # 64 48 48
        out = self.conv2(out)       # 128 24 24
        out = self.res1(out) + out  # 128 24 24
        
        out = self.conv3(out)       # 256 12 128
        out = self.conv4(out)       # 512 6 6
        out = self.res2(out) + out  # 512 6 6
        
        out = self.classifier(out)  # 7
        return F.softmax(out , dim=1)
    
    def predict(self, images):
        x = self.converter(images)
        x = x.reshape((1, 1, 48, 48)).to(self.DEVICE)
        y_hat = self.forward(x)
        return y_hat
