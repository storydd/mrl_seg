import scipy.io as sio
import skimage.io
import numpy as np
import matplotlib as plt
import tifffile as tiff
import gc
import cv2
label=np.load('../label/2000.npy')
RGB_image=skimage.io.imread('../data/prediction/20001017NNDiffuseCLIP1.tif')
print('label的形状',label.shape)
print('RGB_image的形状',RGB_image.shape)
for i in range(3383):
    for j in range(3740):
        if label[i,j]==0:
            RGB_image[i,j]=[0,0,0]
print('RGB_image的数据类型',RGB_image.dtype)
RGB_image=RGB_image.astype(np.uint8)
tiff.imwrite('../image/2000.tif', RGB_image)
