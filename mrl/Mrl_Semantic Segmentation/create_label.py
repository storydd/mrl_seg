import scipy.io as sio
import skimage.io
import numpy as np
import matplotlib as plt
import tifffile as tiff
import gc
import cv2
result_image=skimage.io.imread('result/2020.tif')
RGB_image=np.load('data/prediction/paded_image/2020.npy')
print('result_image的形状',result_image.shape)
print('RGB_image的形状',RGB_image.shape)
for i in range(6912):
    for j in range(7680):
        if np.all(result_image[i,j]==0):
            result_image[i,j]=RGB_image[i,j]
result_image=result_image.astype(np.uint8)
tiff.imwrite('result/biaoji/2020.tif', result_image)
