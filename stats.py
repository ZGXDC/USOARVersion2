#Necessary Modules
import glob
import os
import matplotlib.pyplot as plt
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
import random
from torchvision.transforms import v2
from meanAndStd import meanAndStd

class rsnaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.file = annotations_file
        if annotations_file == 'train_split.csv' or annotations_file=='val_split.csv':
            self.img_labels = pd.read_csv(annotations_file).to_dict(orient='records') #change this
        elif annotations_file=='CMMD_clinicaldata_updated.xlsx':
            self.img_labels = pd.read_excel(annotations_file).to_dict(orient='records')
              
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
    
    def countPosAndNegSamples(self):
           #get cancer key and see value put them into a list --> make 2 lists of dictionary objects
        cancerCount = 0
        nonCancerousCount = 0
        
        #Splitting labels into positive and negative 
        #Count number of positive and negative labels
        if self.file == 'train_split.csv' or self.file=='val_split.csv':
            for label in self.img_labels:
                if label['cancer'] == 1:
                    cancerCount+=1
                else:
                    nonCancerousCount+=1
                    
        elif self.file=='CMMD_clinicaldata_updated.xlsx':
            for label in self.img_labels:
                if label['classification']=='Malignant':
                    cancerCount+=1
                else:
                    nonCancerousCount+=1
        print("CANCEROUS COUNT: ", cancerCount)
        print("NONCANCEROUS COUNT: ", nonCancerousCount)

    def biRADS(self):
        score0Count = 0
        score1Count = 0
        score2Count = 0
        
        score0AndCancer = 0
        score2AndCancer = 0
        
        for label in self.img_labels:
            if label['BIRADS']==0.0:
                score0Count+=1
                if label['cancer']==1:
                    score0AndCancer+=1
            elif label['BIRADS']==1.0:
                score1Count+=1
            elif label['BIRADS']==2.0:
                score2Count+=1
                if label['cancer']==1:
                    score2AndCancer+=1
        
        print('\n\nScore 0: ', score0Count)
        print('Score 0 and cancer: ', score0AndCancer)
        print('Score 1: ', score1Count)
        print('Score 2: ', score2Count)
        print('Score 2 and cancer: ', score2AndCancer)
        
    def density(self):
        aCount = 0
        aAndCancer= 0
        difNegativeA = 0
        
        bCount = 0
        bAndCancer = 0
        difNegativeB = 0
        
        cCount = 0
        cAndCancer = 0
        difNegativeC = 0
        
        dCount = 0
        dAndCancer = 0
        difNegativeD = 0
        
        for label in self.img_labels:
            if label['density']=='A':
                aCount +=1
                if label['cancer']==1:
                    aAndCancer+=1
                if label['difficult_negative_case']==True:
                    difNegativeA+=1
                
            elif label['density']=='B':
                bCount +=1
                if label['cancer']==1:
                    bAndCancer+=1
                if label['difficult_negative_case']==True:
                    difNegativeB+=1
                    
            elif label['density']=='C':
                cCount+=1
                if label['cancer']==1:
                    cAndCancer+=1
                if label['difficult_negative_case']==True:
                    difNegativeC+=1
                
            elif label['density']=='D':
                dCount+=1
                if label['cancer']==1:
                    dAndCancer+=1
                if label['difficult_negative_case']==True:
                    difNegativeD+=1
                    
        print('\n\nA: ', aCount)
        print('A Cancerous: ', aAndCancer)
        print('Difficult Negative A: ', difNegativeA)
        
        print('\nB: ', bCount)
        print('B Cancerous: ', bAndCancer)
        print('Difficult Negative B: ', difNegativeB)
        
        print('\nC: ', cCount)
        print('C Cancerous: ', cAndCancer)
        print('Difficult Negative C: ', difNegativeC)
        
        print('\nD: ', dCount)
        print('D Cancerous: ', dAndCancer)
        print('Difficult Negative D: ', difNegativeD)
       
    def difficultNegative(self):
        difficultNegativeCases = 0
        
        for label in self.img_labels:
            if label['difficult_negative_case']==True:
                difficultNegativeCases+=1
                
        print('\n\nDifficult Negative Cases: ', difficultNegativeCases)
         
    def abnormality(self):
        calicificationCount = 0
        calcificationCancer = 0
        
        massCount = 0
        massCountCancer = 0
        
        bothCount = 0
        bothCancer = 0
        
        for label in self.img_labels:
            if label['abnoramlity']=='calcification':
                calicificationCount+=1
                if label['classification']=='Malignant':
                    calcificationCancer+=1
                    
            elif label['abnoramlity']=='mass':
                massCount+=1
                if label['classification']=='Malignant':
                    massCountCancer+=1
                    
            elif label['abnoramlity']=='both':
                bothCount+=1
                if label['classification']=='Malignant':
                    bothCancer+=1
        
        print('\nCalcification: ', calicificationCount)
        print('Calcification and Cancer: ', calcificationCancer)
        
        print('\nMass: ', massCount)
        print('Mass and Cancer: ', massCountCancer)
        
        print('\nBoth: ', bothCount)
        print('Both and Cancer: ', bothCancer)
             
    def subtype(self):
        luminalACount = 0
        luminalACancer = 0
        
        luminalBCount = 0
        luminalBCancer = 0
        
        her2Count = 0
        her2Cancer = 0
        
        tripleNeg = 0
        tripNegCancer = 0
        
        for label in self.img_labels:
            if label['subtype']=='Luminal A':
                luminalACount+=1
                if label['classification']=='Malignant':
                    luminalACancer+=1
            elif label['subtype']=='Luminal B':
                luminalBCount+=1
                if label['classification']=='Malignant':
                    luminalBCancer+=1
            elif label['subtype']=='HER2-enriched':
                her2Count+=1
                if label['classification']=='Malignant':
                    her2Cancer+=1
            elif label['subtype']=='triple negative':
                tripleNeg+=1
                if label['classification']=='Malignant':
                    tripNegCancer+=1
                    
        print('\n\nLuminal A: ', luminalACount)
        print('Luminal A Cancer: ', luminalACancer)
        
        print('\nLuminal B: ', luminalBCount)
        print('Luminal B Count: ', luminalBCancer)
        
        print('\nHER2: ', her2Count)
        print('HER2 Caner: ', her2Cancer)
        
        print('\nTriple Negative: ', tripleNeg)
        print('Triple Negative Cancer: ', tripNegCancer)
        
        
CMMDDataset = rsnaDataset('CMMD_clinicaldata_updated.xlsx', '/home/zgxdc/USOAR/ConvertedCMMD', transform=None, target_transform=None)
rsnaDataset = rsnaDataset('train_split.csv', '/home/zgxdc/USOAR/train_images', transform = None, target_transform=None)

#Calculating Cancerous/Noncancerous Samples
CMMDDataset.countPosAndNegSamples()
rsnaDataset.countPosAndNegSamples()

#BIRADS
rsnaDataset.biRADS()

#Density
rsnaDataset.density()

#Difficult Negative Case
rsnaDataset.difficultNegative()

#CMMD Abnormality Count
CMMDDataset.abnormality()

CMMDDataset.subtype()