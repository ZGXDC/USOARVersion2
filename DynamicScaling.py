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

#---------------------------------------------------------------------------------------------------------
#Custom pytorch Dataset
class rsnaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.file = annotations_file
        
        if annotations_file == 'train_split.csv' or annotations_file=='val_split.csv':
            self.img_labels = pd.read_csv(annotations_file).to_dict(orient='records') #change this
        elif annotations_file=='CMMD_clinicaldata_revision.xlsx':
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
        elif annotations_file=='CMMD_clinicaldata_revision.xlsx':
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
        elif self.file=='CMMD_clinicaldata_revision.xlsx':
            numImages = self.img_labels[idx]['number']
            imgNum = random.randint(1,numImages)
            img_path = os.path.join(self.img_dir, str(self.img_labels[idx]['ID1']) + '-1-'+str(imgNum)+'.png')
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
    
    #make function called increase negative samples
    def increaseSamples(self, count):
        newImgLabelObject = [] # this should be a list of dictionary objects
        
        if count != 0 and self.cancerPercent<0.75:
            self.cancerPercent += 0.005
            self.nonCancerousPercent -=0.005
        
        print(len(self.img_labels))
        
        cancerAmount = int(len(self.img_labels) * self.cancerPercent)
        nonCancerousAmount = int(len(self.img_labels) * self.nonCancerousPercent)
        
        print('Cancer Percent', self.cancerPercent)
        print('Noncancerous Percent', self.nonCancerousPercent)
        print('Cancer Amount', cancerAmount)
        print('Noncancerous Amount', nonCancerousAmount)
        
        randomGen = 0
        for i in range(cancerAmount):
            randomGen = random.randint(0, len(self.img_labelCancer)-1)
            newImgLabelObject.append(self.img_labelCancer[randomGen])
            
        for i in range(nonCancerousAmount):
            randomGen = random.randint(0, len(self.img_labelNoncancerous)-1)
            newImgLabelObject.append(self.img_labelNoncancerous[randomGen])
        
        random.shuffle(newImgLabelObject)
        self.img_labels = newImgLabelObject
    
    def decreaseSamples(self, count):
        newImgLabelObject = [] # this should be a list of dictionary objects
        
        if count != 0 and self.cancerPercent>0.50:
            self.cancerPercent -= 0.005
            self.nonCancerousPercent +=0.005
        
        print(len(self.img_labels))
        
        cancerAmount = int(len(self.img_labels) * self.cancerPercent)
        nonCancerousAmount = int(len(self.img_labels) * self.nonCancerousPercent)
        
        print('Cancer Percent', self.cancerPercent)
        print('Noncancerous Percent', self.nonCancerousPercent)
        print('Cancer Amount', cancerAmount)
        print('Noncancerous Amount', nonCancerousAmount)
        
        randomGen = 0
        for i in range(cancerAmount):
            randomGen = random.randint(0, len(self.img_labelCancer)-1)
            newImgLabelObject.append(self.img_labelCancer[randomGen])
            
        for i in range(nonCancerousAmount):
            randomGen = random.randint(0, len(self.img_labelNoncancerous)-1)
            newImgLabelObject.append(self.img_labelNoncancerous[randomGen])
        
        random.shuffle(newImgLabelObject)
        self.img_labels = newImgLabelObject
        
    #can also go in an see how many images it's showing from the labels and make 50% of that cancer and 50% noncancerous
#---------------------------------------------------------------------------------------------------------------

#Method to load the PNGS into the code from the directory
def loadPNG(dirPath):
    #1) Define path to PNG images
    path = dirPath

    #2) Use glob function to load all png from path into list of file names
    imageFiles = glob.glob(os.path.join(path, "**/*.png"), recursive = True)

    #3) Iterate through each file in imageFiles, read, append to list
    imageObjects = []
    for file in imageFiles:
        try:
            img = Image.open(file)
            imageObjects.append(img)
        except Exception as e:
            print("ERROR", e)
   
    return imageObjects

#---------------------------------------------------------------------------------------------------------------
#Method to evaluate the model's performance
def eval(testloader, net):
    correct = 0
    total = 0
   
    outs = []
    lbls = []
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            outs.append(outputs.cpu())
            lbls.append(labels.cpu())

            # the class with the highest energy is what we choose as prediction
            predicted = outputs > .5
            predicted = predicted.to(torch.float32)
            total += labels.size(0)
            correct +=(predicted.squeeze()==labels.cuda()).sum().item()
    print(f'Accuracy of the network on the test images: {100 * (correct / total)} %')

    outs = torch.cat(outs)
    lbls = torch.cat(lbls)

    fpr, tpr, thresholds = roc_curve(lbls.numpy(), outs.numpy())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    cancer = 0.0
    non_cancer = 0.0
    cancer_right = 0.0
    non_cancer_right = 0.0

    for x in range(len(outs)):
        if lbls[x] == 1:
            cancer += 1
            if outs[x] >= .5:
                cancer_right += 1
        else:
            non_cancer += 1
            if outs[x] < .5:
                non_cancer_right += 1

    print("Non Cancerous Accuracy:", non_cancer_right / non_cancer)
    print("Cancerous Accuracy:", cancer_right / cancer)
    
#-----------------------------------------------------------------------
#MAIN
# 1) Load PNGS from directory into list of PIL Image Objects
trainImagesPNG = loadPNG('/home/zgxdc/USOAR/train_images')

#2) Load/Normalize training/test datasets
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomAffine(0, (0.05,0), scale=None, shear=None),
    transforms.RandomRotation(20),
    transforms.ElasticTransform(alpha=3.0, sigma=0.5),
    transforms.v2.GaussianNoise(mean =0.0, sigma = 0.01),
    transforms.Resize((512, 512)),
    transforms.Normalize((0.1390), (0.2200))
])

batch_size = 32

trainDataset = rsnaDataset('CMMD_clinicaldata_revision.xlsx', '/home/zgxdc/USOAR/ConvertedCMMD', transform = transform)
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=10)

#/mnt/network/sgrieggs/rsna/train_balanced.csv

#can make a separate file for getting mean and standard deviation/method

# #calling increase samples early to get a 50/50 dataset
increaseSamplesCount = 0
trainDataset.increaseSamples(increaseSamplesCount)

#calling decrease samples to get 75/25 dataset (remember to change percents in dataset class)
# decreaseSamplesCount = 0
# trainDataset.decreaseSamples(decreaseSamplesCount)

testDataset = rsnaDataset('val_split.csv', '/home/zgxdc/USOAR/train_images', transform = transform)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=10)

#3) Defining the Convolutional Neural Network 
#PATH is the current best performing saved epoch to load in
PATH = "/home/zgxdc/rsnaPaths/resnet50NewCSVEpoch84.pth"
directoryPath = '/home/zgxdc/rsnaPaths'

#NEW RESNET 50 MODEL
net = models.resnet50(pretrained=True)

#Modify Resnet50 for Grayscale images ---------------------------------------------
firstLayer = net.conv1
originalWeights = firstLayer.weight.data
newWeights = originalWeights.mean(1, keepdim=True)
newLayer1 = nn.Conv2d(1, out_channels=firstLayer.out_channels, kernel_size=firstLayer.kernel_size, stride=firstLayer.stride,
                      padding = firstLayer.padding, bias = firstLayer.bias)
newLayer1.weight.data = newWeights
net.conv1 = newLayer1
net.fc = nn.Linear(net.fc.in_features, 1)
net = net.cuda()
#----------------------------------------------------------------------------------

checkPoint = torch.load(PATH)
net.load_state_dict(checkPoint['modelStateDict'])

#4) Define a loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
optimizer.load_state_dict(checkPoint['optimizerStateDict'])

# #5) Train the network
# Loop over data iterator, feed inputs to network, and optimize

for epoch in range(50): #1000 epochs
    running_loss = 0.0
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=10)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs.cuda())
        labels = labels.view(-1, 1).float()
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32).cuda())
        loss.backward() #back propogration??
        optimizer.step()
        
        running_loss+=loss.item()
        if i%100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
            
    #save model
    # filePath = os.path.join(directoryPath, f'resNet50DSAndNewAugments{epoch+1}.pth')
    # torch.save({'epoch': epoch, 'modelStateDict': net.state_dict(), 'optimizerStateDict': optimizer.state_dict()}, filePath)
    #run evaluation on each epoch
    eval(testloader, net)
    if epoch!=99:   
        # call the function in the dataset to add additonal samples, reset dataset, pool percentage of negative samples and posititive samples
        trainDataset.increaseSamples(increaseSamplesCount)
        increaseSamplesCount+=1
        #call the function in the dataset to decrease oversampling, reset dataset,
        # trainDataset.decreaseSamples(decreaseSamplesCount)
        # decreaseSamplesCount+=1
    
print('Finished Training')


