import pydicom
import matplotlib.pyplot as plt
import cv2
import torch
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
from torchvision.transforms import v2

# dcmfilepath = '/home/zgxdc/USOAR/CMMD/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-2.dcm'
# ds = pydicom.dcmread(dcmfilepath)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(0, (0.03,0), scale=None, shear=None),
    transforms.RandomRotation(20),
    transforms.ElasticTransform(alpha=3.0, sigma=0.5),
    transforms.ToTensor(),
    transforms.v2.GaussianNoise(mean=0.0, sigma = 0.01),
    transforms.Resize((512, 512)),
    transforms.Normalize((0.1390), (0.2200))
])

imgPath = '/home/zgxdc/USOAR/ConvertedCMMD/D1-0001-1-1.png'
img = Image.open(imgPath)
newImg = transform(img)
print(newImg.shape)

img_np = newImg.squeeze(0).numpy()

plt.imshow(img_np, cmap="gray")
plt.axis("off")
plt.savefig('augmentDisplay1.png')



# pixelArray = newImg.pixel_array

# plt.imshow(pixelArray, cmap=plt.cm.gray)

# plt.savefig('augmentDisplay1.png')

#pixelArray = newImg.pixel_array

# test = torch.tensor(0).cuda()