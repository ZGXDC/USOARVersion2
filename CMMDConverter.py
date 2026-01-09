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
import random
import pydicom

def dicomToPNG(imagePath, outputPath): #This will save the individual files into png in a specific directory
    
    dicomData = pydicom.dcmread(imagePath)
    pixelArray = dicomData.pixel_array
    
    #split imagePath name by '/', then splits the
    imagePathParts = imagePath.split('/')
    specificImage = imagePathParts[len(imagePathParts)-1].split('.')
     
    print(pixelArray)
    scaled_pixel_array = (np.maximum(pixelArray, 0) / pixelArray.max()) * 255.0
    scaled_pixel_array = np.uint8(scaled_pixel_array)
    
    image = Image.fromarray(scaled_pixel_array)
    outputPath += '/' + str(dicomData.PatientName) + '-'+str(specificImage[0])+'.png'

    print(outputPath)
    image.save(outputPath)

def directoryIteration(directoryPath):
#want to return an imagePath    
    imagePathList = []
    for subdir, dirs, files in os.walk(directoryPath):
        for f in files:
            name = os.path.join(subdir,f)
            imagePathList.append(name)
    return imagePathList

#gets list of image paths from specific directory
paths = directoryIteration('/home/zgxdc/USOAR/CMMD/CMMD')

#converts each path in list of image paths into a png and saves to desired directory
for p in paths:
    if(p=='/home/zgxdc/USOAR/CMMD/CMMD/LICENSE' or p=='/home/zgxdc/USOAR/CMMD/CMMD/.DS_Store'):
        continue
    if('.DS_Store' in p):
        continue
    else:
        dicomToPNG(p, '/home/zgxdc/USOAR/ConvertedCMMD')