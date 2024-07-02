import numpy as np
from PIL import Image
import skimage.io
import tifffile as tiff
import cv2
# NIR_image21=skimage.io.imread('../data/all/21beach/image/21NIR.tif')
'data/image中图片的路径'
RGB_image=skimage.io.imread('../image/2000.tif')
'data/label中标签文件的地址路径'
label=np.load('../label/2000.npy')

print(RGB_image.shape)
print(label.shape)

RGB_image = cv2.copyMakeBorder(RGB_image, 101, 100, 50, 50, cv2.BORDER_CONSTANT, value=[0,0,0])
label = cv2.copyMakeBorder(label, 101, 100, 50, 50, cv2.BORDER_CONSTANT, value=0)
#
print(RGB_image.shape)
print(label.shape)
# #
# #
# '填充后标签保存路径'
np.save('../pad_data/2000label.npy',label)
# '填充后图片保存路径'
np.save('../pad_data/2000image.npy', RGB_image)