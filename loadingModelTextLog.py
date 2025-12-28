import glob
import os
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.utils.data.dataloader
from torchvision import transforms
from PIL import Image
import pandas as pd #read csv in pandas
import re
from torchvision.io import decode_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import roc_curve, auc
import numpy as np
import torchvision.models as models

#100 EPOCHS
print('100 Epochs')
with open('log.txt', 'r') as f:
    data = f.readlines()

maxCancerousAccuracy = 0
maxEpochNumber = 0

for i in range(0, len(data), 7):
    if(i>(len(data)-7)):
        break
    
    epochNumber = re.findall(r'\d+\.\d+|\d+', data[i])
    #print('Epoch #', epochNumber[0])
    
    cancerousAccuracy = re.findall(r'\d+\.\d+|\d+', data[i+6])
    #print('Cancerous Accuracy: ', cancerousAccuracy[len(cancerousAccuracy)-1])
    
    if i==0:
        maxCancerousAccuracy = cancerousAccuracy[0]
        maxEpochNumber = epochNumber[0]
    else:
        if cancerousAccuracy[0]>maxCancerousAccuracy:
            maxCancerousAccuracy = cancerousAccuracy[0]
            maxEpochNumber = epochNumber[0]

print('\nBEST PERFORMING: \nEPOCH #', maxEpochNumber, '\nCancerous Accuracy: ', maxCancerousAccuracy, '\n')


#1000 EPOCHS
print('100 Epochs')
with open('newCSVresnet50#2.txt', 'r') as f:
    data = f.readlines()

maxCancerousAccuracy = 0
maxEpochNumber = 0

for i in range(0, len(data), 7):
    if(i>(len(data)-7)):
        break
    
    epochNumber = re.findall(r'\d+\.\d+|\d+', data[i])
    #print('Epoch #', epochNumber[0])
    
    cancerousAccuracy = re.findall(r'\d+\.\d+|\d+', data[i+6])
    #print('Cancerous Accuracy: ', cancerousAccuracy[len(cancerousAccuracy)-1])
    
    if i==0:
        maxCancerousAccuracy = cancerousAccuracy[0]
        maxEpochNumber = epochNumber[0]
    else:
        if cancerousAccuracy[0]>maxCancerousAccuracy:
            maxCancerousAccuracy = cancerousAccuracy[0]
            maxEpochNumber = epochNumber[0]

print('\nBEST PERFORMING: \nEPOCH #', maxEpochNumber, '\nCancerous Accuracy: ', maxCancerousAccuracy, '\n')




