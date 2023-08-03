import numpy as np
import os
import cv2
import random
import torch
import torchvision
from torchvision import models
from torchvision import datasets, transforms
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.transforms.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import json
from PIL import Image
import matplotlib.pyplot as plt
import csv


transforms = T.Compose([
    T.Resize([256, 256]),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomDataSet(Dataset):
    def __init__(self, root):
        self.root = root
        self.image_paths = os.listdir(root)
        self.transform = transforms

    def __getitem__(self, index):
        image_path = self.root + '/' + self.image_paths[index]
        data = Image.open(image_path).convert('RGB')
        data = self.transform(data)
        filename = self.image_paths[index]
        return data, filename
    
    def __len__(self):
        return len(self.image_paths)


def load(data_path):
    dataset = datasets.ImageFolder(data_path, transform = transforms)
    return dataset
    
def split(trainset):
    data_len=len(trainset)
    train_num=int(data_len*4/5)
    valid_num=int(data_len/5)
    if (train_num + valid_num) < data_len:
        valid_num = valid_num + 1
    generator = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(trainset, [train_num, valid_num], generator)
    train_load = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    valid_load = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=8)
    #print(len(train_set))
    #print(valid_set[500][1])
    return train_load, valid_load

def train(model, device, train_loader, valid_loader, optimizer, epoch):
    epoch_time = list(range(1, (epoch+1)))
    loss_t_list = [0] * epoch
    loss_v_list = [0] * epoch
    acc_t_list = [0] * epoch
    acc_v_list = [0] * epoch
    model.train()   # for Dropout, Batch Normalization, etc.
    train_len = len(train_loader.dataset)
    valid_len = len(valid_loader.dataset)
    total_correct = 0
    for i in range(epoch):
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # forward
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_correct, train_loss = valid(model, device, train_loader)
        valid_correct, valid_loss = valid(model, device, valid_loader)
        print('epoch',(i+1), ':\n', '    train correct = ', (train_correct*100/train_len), '%, valid correct = ', (valid_correct*100/valid_len),
             '%\n     train loss = ', train_loss, ', valid loss = ', valid_loss)
        loss_t_list[i] = train_loss
        loss_v_list[i] = valid_loss
        acc_t_list[i] = train_correct/train_len
        acc_v_list[i] = valid_correct/valid_len
        if (valid_correct) > total_correct:
            total_correct = valid_correct
            torch.save(model.state_dict(), 'model_weights.pth')
    fig1 = plt.figure()
    plt.title("loss")
    plt.plot(epoch_time, loss_t_list, label='train')
    plt.plot(epoch_time, loss_v_list, label='valid')
    plt.legend(
        loc='best',
        fontsize=10,
        facecolor='#ccc',
        edgecolor='#000',)
    fig1.savefig('loss.png')
    fig2 = plt.figure()
    plt.title("accurancy")
    plt.plot(epoch_time, acc_t_list, label='train')
    plt.plot(epoch_time, acc_v_list, label='valid')
    plt.legend(
        loc='best',
        fontsize=10,
        facecolor='#ccc',
        edgecolor='#000',)
    fig2.savefig('accurancy.png')
    plt.close()

        
def valid(model, device, test_loader):
    model.eval()    # equals to model.train(False)
    correct = 0
    total_loss = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.no_grad():
            # forward    
            outputs = model(inputs)
            
            # evaluation, e.g.
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
    return correct, total_loss

def module(grad):
    device = torch.device('cuda')

    model = torchvision.models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, progress=True)
    for param in model.parameters():
        param.requires_grad = grad
    model.fc = nn.Linear(2048, 12)
    model = model.to(device, non_blocking=True)

    #optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    return model, device, optimizer

def run(train_path, epoch):
    trainset = load(train_path)
    class_index = trainset.class_to_idx
    train_loader, valid_loader = split(trainset)
    model, device, optimizer = module(True)
    train(model, device, train_loader, valid_loader, optimizer, epoch)
    #train2(model, device, train_loader, optimizer, epoch)
    return class_index

def test(test_path, class_idx):
    testset=CustomDataSet(test_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

    model, device, optimizer = module(False)
    weights = torch.load('model_weights.pth', map_location='cpu')
    model.load_state_dict(weights)
    model.eval()

    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file', 'species'])
        for inputs, image_names in testloader:
            inputs = inputs.to(device, non_blocking=True)

            with torch.no_grad():
                # forward    
                outputs = model(inputs)
                # evaluation, e.g.
                #pred = torch.max(outputs, 1)
                pred = outputs.argmax(dim=1, keepdim=True)
                pred = pred.to('cpu').tolist()
                pred_len = len(pred)
                for i in range(pred_len):
                    name = image_names[i]
                    idx = pred[i][0]
                    classify=list(class_idx.keys())[list(class_idx.values()).index(idx)]
                    writer.writerow([name, classify])
    
