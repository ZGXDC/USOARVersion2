import pydicom
import matplotlib.pyplot as plt
import cv2
import torch

dcmfilepath = '/home/zgxdc/USOAR/CMMD/CMMD/D1-0001/07-18-2010-NA-NA-79377/1.000000-NA-70244/1-2.dcm'
ds = pydicom.dcmread(dcmfilepath)

pixelArray = ds.pixel_array

plt.imshow(pixelArray, cmap=plt.cm.gray)

plt.savefig('cmmdTest2.png')

test = torch.tensor(0).cuda()