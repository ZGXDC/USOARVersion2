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
import shutil

def copyFilesFromCSV(csv, destinationPath):
    columns = ['patient_id', 'image_id', 'cancer']
    csvFile = pd.read_csv(csv, usecols=columns)
    patientIDColumn = csvFile.patient_id
    imageIDColumn = csvFile.image_id
    
    #makeList of patientID_imageID pngs from csvFile
    imageNameList = []
    for i in range(len(patientIDColumn)):
        fileName = str(patientIDColumn[i])+'_'+str(imageIDColumn[i])+'.png'
        imageNameList.append(fileName)
    
    #create filePaths and copy from one directory to new directory
    for image in imageNameList:
        sourceFile = '/home/zgxdc/USOAR/train_images/'+str(image)
        shutil.copy(sourceFile, destinationPath)
        
        
copyFilesFromCSV('val_split.csv', '/home/zgxdc/USOAR/valSplitCsv')


    

