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
import matplotlib
import resnet50_model as RN50


epoch = 50
train_path = "/home/JHlin/seed_class/plant-seedlings-classification/train"
test_path = "/home/JHlin/seed_class/plant-seedlings-classification/test"
class_idx = RN50.run(train_path, epoch)
RN50.test(test_path, class_idx)

