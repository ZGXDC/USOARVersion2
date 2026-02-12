#Python File to calculate mean and standard deviation of a dataset
import glob
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.utils.data.dataloader
from torchvision import transforms, datasets
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
import random
from torchvision.transforms import v2

class rsnaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.file = annotations_file
        if annotations_file == 'train_split.csv' or annotations_file=='val_split.csv':
            self.img_labels = pd.read_csv(annotations_file).to_dict(orient='records') #change this
        elif annotations_file=='CMMD_clinicaldata_updated.xlsx':
            self.img_labels = pd.read_excel(annotations_file).to_dict(orient='records')
        
        #get cancer key and see value put them into a list --> make 2 lists of dictionary objects
        self.img_labelCancer = []
        self.cancerCount = 0
        self.img_labelNoncancerous = []
        self.nonCancerousCount = 0
        self.cancerPercent = 0.5
        self.nonCancerousPercent = 0.5
        
        #Splitting labels into positive and negative 
        #Count number of positive and negative labels
        if annotations_file == 'train_split.csv' or annotations_file=='val_split.csv':
            for label in self.img_labels:
                if label['cancer'] == 1:
                    self.img_labelCancer.append(label)
                    self.cancerCount+=1
                else:
                    self.img_labelNoncancerous.append(label)
                    self.nonCancerousCount+=1
        elif annotations_file=='CMMD_clinicaldata_updated.xlsx':
            for label in self.img_labels:
                if label['classification']=='Malignant':
                    self.img_labelCancer.append(label)
                    self.cancerCount+=1
                else:
                    self.img_labelNoncancerous.append(label)
                    self.nonCancerousCount+=1
              
        #self.img_labels is a dictionary with keys/values compatible to the csv file. Ex: patient_id: 1234
        # for label in self.img_labels:
        #     print(label)
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if self.file == 'train_split.csv' or self.file=='val_split.csv':
            img_path = os.path.join(self.img_dir, str(self.img_labels[idx]['patient_id']) + "_" + str(self.img_labels[idx]['image_id'])+".png")
            image = decode_image(img_path)/255.0
            label =  self.img_labels[idx]['cancer']
        elif self.file=='CMMD_clinicaldata_updated.xlsx':
            img_path = os.path.join(self.img_dir, str(self.img_labels[idx]['ID1']) + '-'+str(self.img_labels[idx]['ImageID'])+'.png')
            image = decode_image(img_path)/255.0
            if self.img_labels[idx]['classification']=='Malignant':
                label = 1
            else:
                label = 0
        if self.transform:
            image = self.transform(image)     
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor(label)

class meanAndStd:
    #create a dataloader with minimal transforms to determine unbiased mean and standard deviation
    def __init__(self, annotationFile, imagePath):
        self.annotationFile = annotationFile
        self.imagePath = imagePath
        self.transform = transforms.Grayscale()
        self.dataset = rsnaDataset('train_split.csv', 'train_images', self.transform)
        self.dataLoader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=10)
        
    def calculateMeanAndStd(self):
        mean = 0.0
        std = 0.0
        numPixels = 0
        
        for images, *_ in self.dataLoader:
            batchSize, channels, height, width = images.shape
            numPixels += batchSize * height * width
            mean+=images.sum()
            std += (images**2).sum()
        
        mean /= numPixels
        std = (std/numPixels - mean ** 2) ** 0.5
        
        return mean.item(), std.item()