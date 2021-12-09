from __future__ import print_function
import torch
import torchvision            
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import time
import math
import random as rd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import datetime
import os
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt
from PIL import Image

print(torch.__version__)

class MyDataset():
    def __init__(self, path, made_transforms = None):
        self.path = path
        self.img_set, self.label_set, self.length, self.classes_number = self.read_data()
        self.transforms = made_transforms

    def read_data(self):
        data_X = []
        data_Y = []
        
        for label, name in enumerate(os.listdir(self.path)):
            target_dir = os.path.join(self.path, name)
            
            for files in os.listdir(target_dir):
                file_ = os.path.join(target_dir, files)
                img = Image.open(file_)
                
                if img != None:
                    data_X.append(file_)
                    data_Y.append(label)
                
                
        return data_X, data_Y, len(data_X), len(name)

    def __getitem__(self, index):
        img_ = Image.open(self.img_set[index])

        if self.transforms != None:
            img_ = self.transforms(img_)

        return img_, self.label_set[index]

    def __len__(self):
        return self.length


class SimpleCNN(nn.Module):
    """
        Simple CNN Clssifier
    """
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Sequential(
            #1 48 48
            nn.Conv2d(1, 32, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #32 24 24
            nn.Conv2d(32, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #64 12 12
            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #128 6 6
            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            #256 3 3
            # nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
            # nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
        )
        #512 3 3

        self.avg_pool = nn.AvgPool2d(3)
        #512 1 1
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), 
            nn.Linear(128, 7)   
        )
    
    def forward(self, x):
        features = self.conv(x)
        x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x , dim=1)
        
        return x

class ResNet(nn.Module):
    s

class SimpleFC(nn.Module):
    """
        Simple CNN Clssifier
    """
    def __init__(self, num_classes=7):
        super(SimpleFC, self).__init__()

        #1 48 48
        self.classifier = nn.Sequential(
            nn.Linear(48*48, 128), 
            nn.Linear(128, 7)   
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x , dim=1)
        
        return x

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # self.modelC = modelC

        self.fc1 = nn.Linear(input, 7)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        # out3 = self.modelC(x)

        out = out1 + out2

        out = out / 2
        return torch.softmax(x, dim=1)            
        
    
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


trainset_path = "./Dataset/train"
testset_path = "./Dataset/test"

transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)), ])
batch_size = 16

custom_dataset_train = MyDataset(trainset_path, made_transforms = transform)
custom_dataset_test = MyDataset(testset_path, made_transforms = transform)

print("train: ", custom_dataset_train.length)
print("test : ", custom_dataset_test.length)

train_loader = DataLoader(dataset=custom_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=custom_dataset_test, batch_size=1000, shuffle=False, drop_last=False)

MODEL_NAME = 'DNN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("MODEL_NAME = {}, DEVICE = {}".format(MODEL_NAME, DEVICE))

sCNN = SimpleCNN().to(DEVICE)
sFC = SimpleFC().to(DEVICE)
model = MyEnsemble(sFC, sCNN, 32).to(DEVICE)
criterion = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=0.00001)

all_losses = []
all_acc = []
i, l = custom_dataset_train[0]

print(type(i))
print(i.shape, l)

max_epoch = 1000   
step = 0             

plot_every = 200
total_loss = 0 
start = time.time()

for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(train_loader):

        x, y = images.to(DEVICE), labels.to(DEVICE) # (N, 1, 28, 28), (N, )
        
        y_hat = model(x) # (N, 10)  
        loss = criterion(y_hat, y)    
        total_loss += loss.item()
          
        optim.zero_grad()           
        loss.backward()              
        optim.step()                        

        if step % 500 == 0:
            print('Epoch({}): {}/{}, Step: {}, Loss: {}'.format(timeSince(start), epoch, max_epoch, step, loss.item()))
        
        if (step + 1) % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            plt.figure()
            plt.plot(all_losses)
            plt.savefig("./losses.png")
            plt.cla()
            total_loss = 0
        

        if step % 1000 == 0:
            save_file_name = "./weights/weights_" + str(step) + ".pth"
            torch.save(model.state_dict(), save_file_name)
            
            model.eval()
            acc = 0.
            with torch.no_grad():  
                for idx, (images, labels) in enumerate(test_loader):
                    x, y = images.to(DEVICE), labels.to(DEVICE) # (N, 1, 28, 28), (N, )
                    y_hat = model(x) # (N, 10)
                    loss = criterion(y_hat, y)
                    _, indices = torch.max(y_hat, dim=-1)     
                    
                    acc += torch.sum(indices == y).item() 
            
            all_acc.append(acc/len(custom_dataset_test)*100)                                                             
            print('*'*20, 'Test', '*'*20)
            print('Step: {}, Loss: {}, Accuracy: {} %'.format(step, loss.item(), acc/len(custom_dataset_test)*100))
            print('*'*46)
            
            plt.figure()
            plt.plot(all_acc)
            plt.savefig("./acc_test.png")
            plt.cla()
            model.train()
        step += 1
        
        
