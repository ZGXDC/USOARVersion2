#Necessary Modules
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


#----------------------------------------------------------------------------------------------

#Class to make a dataset
class rsnaDataset(Dataset): #load in csv file and generate image paths
    #get whether or not image shows cancer /mnt/network/sgrieggs/rsna/train.csv
    #Creates custom dataset instance from loaded PNGS list
    def __init__(self, image_list, csv, transform=None, target_transform=None):#read csv file, save in variable
        #make into lists with image paths and ids
        #VARIABLES
        self.image_list = image_list
        self.transform = transform
        self.csvFile = csv
        self.target_transform = None
            
        #Reading CSV File and getting id list
        columns = ['patient_id', 'image_id', 'cancer']
        csvFile = pd.read_csv(self.csvFile, usecols=columns)
        self.patientIDColumn = csvFile.patient_id
        self.imageIDColumn = csvFile.image_id
        self.cancerColumn = csvFile.cancer
        #put into dictionary and map imageIDS and patientsIDS to cancer value?
        self.refDictionary = self.makeDictionary()
        #print(len(self.refDictionary))
        
    #Returns number of samples in particular dataset
    def __len__(self):
        return len(self.patientIDColumn)
    #Returns the sample at particular index
    def __getitem__(self, index): #return image and whether or not there is cancer
        # cancer=  noncancerous 1=True, 0=False #Get dataset loaded, same thing as example
        #take a path to an image, load it, store it in image variable, set cancer equal to 
        imageFile = str(self.patientIDColumn[index])+'_'+str(self.imageIDColumn[index])+'.png'
        for i in self.image_list:
            fileNameSubstring = i.filename.split('/')
            nameToSearch = fileNameSubstring[5]
            if nameToSearch == imageFile:
                image = i
        image = decode_image(image.filename)/255.0
        cancer = self.getCancerValue(imageFile)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            cancer = self.target_transform(cancer)
        return image, torch.tensor(cancer)
        #put in try/except clause
    #Makes a Dictionary of patientID_imageID keys to cancer values from csvFile
    def makeDictionary(self):
        idAndCancerDict = {}
        for i in range(len(self.patientIDColumn)):
            fileName = str(self.patientIDColumn[i])+'_'+str(self.imageIDColumn[i])+'.png'
            idAndCancerDict[fileName] = self.cancerColumn[i]
        return idAndCancerDict
    #Method to search reference ditionary for cancervalue according to patientID_imageID
    def getCancerValue(self, imageFile):
        #searches the dictionary
        cancerValue = self.refDictionary[imageFile]
        return cancerValue
    def getCancerLabels(self):
        return self.cancerColumn
    
#---------------------------------------------------------------------------------------------------------------

#Method to load the PNGS into the code from the directory
def loadPNG(dirPath):
    #1) Define path to PNG images
    path = dirPath

    #2) Use glob function to load all png from path into list of file names
    imageFiles = glob.glob(os.path.join(path, "**/*.png"), recursive = True)

    #3) Iterate through each file in imageFiles, read, apped to list
    imageObjects = []
    for file in imageFiles:
        try:
            img = Image.open(file)
            imageObjects.append(img)
        except Exception as e:
            print("ERROR", e)
   
    return imageObjects

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
            correct += (predicted == labels.cuda()).sum().item()
    print('Total = ', total, 'Correct = ', correct)
    print(f'Accuracy of the network on the test images: {100 * (correct // total)} %')

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


#MAIN ----------------------------------------------------------------------------------------
# 1) Load PNGS from directory into list of PIL Image Objects
trainImagesPNG = loadPNG('/home/zgxdc/USOAR/train_images')


#2) Load/Normalize training/test datasets
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.7),
    transforms.RandomRotation(20),
    transforms.Normalize((0.485), (0.229))
])

batch_size = 100

trainDataset = rsnaDataset(trainImagesPNG, 'train_split.csv', transform)

#Oversampling Attempt - read documentation of Weighted Random Sampler function - what percentage of images with cancer it's seeing
classCounts = Counter(trainDataset.getCancerLabels())
numSamples = len(trainDataset.getCancerLabels())
classWeights = {cls: numSamples / count for cls, count in classCounts.items()}
sampleWeights = [classWeights[label] for label in trainDataset.getCancerLabels()]
sampler = WeightedRandomSampler(weights = sampleWeights, num_samples = len(sampleWeights))

trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, sampler=sampler, num_workers=10)
#removed shuffle=True

testDataset = rsnaDataset(trainImagesPNG, 'val_split.csv', transform)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=10)

#show image to model and calculate loss
#3) Defining the Convolutional Neural Network 

PATH = "/home/zgxdc/rsnaPaths/modelEpoch18.pth"
directoryPath = '/home/zgxdc/rsnaPaths'

#NEW RESNET 50 MODEL
# net = models.resnet50(pretrained=True)

# #checkPoint = torch.load(PATH)
# #net.load_state_dict(checkPoint['modelStateDict'])

# #Modify Resnet50 for Grayscale images
# firstLayer = net.conv1
# originalWeights = firstLayer.weight.data
# newWeights = originalWeights.mean(1, keepdim=True)
# newLayer1 = nn.Conv2d(1, out_channels=firstLayer.out_channels, kernel_size=firstLayer.kernel_size, stride=firstLayer.stride,
#                       padding = firstLayer.padding, bias = firstLayer.bias)

# newLayer1.weight.data = newWeights
# net.conv1 = newLayer1
# net.fc = nn.Linear(net.fc.in_features, 1)
# net = net.cuda()

#4) Define a loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
#optimizer.load_state_dict(checkPoint['optimizerStateDict'])

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)

