#Necessary Modules
import glob
import os
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd #read csv in pandas
import re
from torchvision.io import decode_image
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
        return len(self.image_list)
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

#MAIN ----------------------------------------------------------------------------------------
# 1) Load PNGS from directory into list of PIL Image Objects
trainImagesPNG = loadPNG('/home/zgxdc/USOAR/train_images')


#2) Load/Normalize training/test datasets
transform = transforms.Compose([
    transforms.Normalize((0.5), (0.5))
])

trainDataset = rsnaDataset(trainImagesPNG, 'train_split.csv', transform)
print(trainDataset.__getitem__(1024))

testDataset = rsnaDataset(trainImagesPNG, 'val_split.csv', transform)

#show image to model and calculate loss