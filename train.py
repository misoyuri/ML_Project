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

from Networks import CNN, FC, Resnet, Ensemble

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

sCNN = CNN.SimpleCNN().to(DEVICE)
sFC = FC.SimpleFC().to(DEVICE)
model = Resnet.ResNet().to(DEVICE)
# model = Ensemble.MyEnsemble(sFC, sCNN, 32).to(DEVICE)

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
        
        
