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
     
    print(pixelArray)
    scaled_pixel_array = (np.maximum(pixelArray, 0) / pixelArray.max()) * 255.0
    scaled_pixel_array = np.uint8(scaled_pixel_array)
    
    image = Image.fromarray(scaled_pixel_array)
    outputPath += '/' + str(dicomData.PatientName) + '.png'
    image.save(outputPath)
    
#outputPath passed to upper function will need to be the folder 
def directoryIteration(rootDirectoryPath, x):
#need to iterate through the directory and subdirectories to get each individual file
    for dirPath, dirNames, filenames in os.walk(rootDirectoryPath):
        print(dirPath, ' Current Directory')
        print(dirNames, ' SubDirectories in Current Directory')
        print(filenames, ' Files names in current directory')


    
imagePath = '/home/zgxdc/USOAR/CMMD/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-1.dcm'
outputPath = '/home/zgxdc/USOAR/ConvertedCMMD'

#dicomToPNG(imagePath, outputPath)

directoryIteration('/home/zgxdc/USOAR/CMMD/CMMD', 0)

