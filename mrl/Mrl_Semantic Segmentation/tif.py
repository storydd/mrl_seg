import scipy.io as sio
import skimage.io
import numpy as np
import matplotlib as plt
import tifffile as tiff
import gc
import cv2










import numpy as np
import matplotlib.pyplot as plt






result_image=skimage.io.imread('data/prediction/20001017NNDiffuseCLIP1.tif')
result_image1=skimage.io.imread('data/prediction/20131005NNDiffuseCLIP1.tif')
# label=np.load('label/2013.npy')
print('RGB_image的形状',result_image.shape)
print('label的形状',result_image1.shape)
# for i in range(6912):
#     for j in range(7680):
#         if label[i,j]!=0:
#             result_image[i,j]=label[i,j]
# RGB_image=RGB_image.astype(np.uint8)
# tiff.imwrite('image/2020.tif', RGB_image)
# 绘制图片


# tiff.imwrite('fin_result/2020(b).tif', result_image)
# cmap = plt.get_cmap('tab10')
# # plt.imshow(label, cmap=cmap, vmin=0, vmax=5)
# plt.imshow(result_image, cmap=cmap, vmin=0, vmax=5)
# plt.colorbar(ticks=np.arange(6))  # 添加颜色条，设置刻度为0到5
# plt.show()