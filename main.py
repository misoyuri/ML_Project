import numpy as np
import cv2
import os
from PIL import Image
from Networks.Resnet9 import ResNet9

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from Networks import *

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


class MyDataset():
    def __init__(self, path, made_transforms=None):
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


MODEL_NAME = 'CNN'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("MODEL_NAME = {}, DEVICE = {}".format(MODEL_NAME, DEVICE))



face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

model = Resnet9.ResNet9().to(DEVICE)
model.load_state_dict(torch.load('./best_weights./trial3./resnet9_best.pth'))


# model = CNN_63acc.CNN().to(DEVICE)
# model.load_state_dict(torch.load('./weights/weights_150000.pth'))

# model = CNN_61acc.CNN().to(DEVICE)
# model.load_state_dict(torch.load('./best_weights/trial1/weights_best.pth'))

# model = CNN_yoon.CNN().to(DEVICE)
# model.load_state_dict(torch.load('./weights_4000.pth'))


model.eval()

cap = cv2.VideoCapture("input2.mp4")
idx = 0

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
delay = round(1000/fps)
out = cv2.VideoWriter('output2.mp4', fourcc, fps, (w, h))


while True:
    ret, output = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)

    frame = output.copy()
    cv2.rectangle(output, (10, 0), (220, 220), (112, 112, 112), -1)

    cv2.addWeighted(output, 0.5, frame, 1 - .5, 0, frame)



    if len(num_faces) == 0:
        for i in range(len(emotion_dict)):
            emotion_analysis = str(emotion_dict[i]) + ': 0'
            cv2.putText(frame, emotion_analysis, (20, 20 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 255), 1, cv2.LINE_AA)
            # R, G, B = 255, 0, 204

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = cv2.resize(gray_frame[y:y + h, x:x + w], (48, 48))

        emotion_prediction = model.predict(roi_gray_frame)
        _, emotion_idx = torch.max(emotion_prediction, dim=-1)
        emotion_idx = int(emotion_idx.detach().cpu())

        emotion_prediction = emotion_prediction.detach().cpu().numpy().flatten()

        emotion_analysis = ''
        for i in range(len(emotion_prediction)):
            emotion_analysis = str(emotion_dict[i]) + ': ' + '{0:0.3f}'.format(emotion_prediction[i])
            cv2.putText(frame, emotion_analysis, (20, 20 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, emotion_dict[emotion_idx], (x -20 , y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # cv2.imshow('Video', frame)
    # cv2.imwrite(str(idx)+".jpg", frame)
    out.write(frame) 
    idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        break
