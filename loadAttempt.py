#Necessary Modules
import glob
import os
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# class CustomImageDataset(Dataset):
    

def loadPNG(dirPath):
    #1) Define path to PNG images
    path = dirPath

    #2) Use glob function to load all png from path into list of file names
    imageFiles = glob.glob(os.path.join(path, "**/*.png"), recursive = True)

    #3) Iterate through each file in imageFiles, read, apped to list
    images = []
    for file in imageFiles:
        try:
            img = Image.open(file)
            images.append(img)
        except Exception as e:
            print("ERROR", e)
   
    return images

#MAIN 
images = loadPNG('/home/zgxdc/USOAR/ImagesFromZipCode')
# #TEST OPENING
# plt.imshow(images[0],  cmap=plt.cm.gray)
# plt.savefig('workQuestion.png')
# plt.show()


    