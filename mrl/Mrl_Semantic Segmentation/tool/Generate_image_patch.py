from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
import skimage.io
import tifffile as tiff
from tensorflow.keras.utils import to_categorical
import gc
import cv2

scaler = MinMaxScaler()
patch_size = 256


path1='../New_data/data/卫星遥感RGB图像_paded.tif'
path2='../New_data/data/卫星遥感近红外图像_paded.tif'
RGB_tf = skimage.io.imread(path1)
Nir_tf = skimage.io.imread(path2)

print(RGB_tf.shape)
print(Nir_tf.shape)


def datapatches(data, shape: tuple, patch_size):
    dataset = []
    # 在给定的代码中，patchify 函数用于将输入的 data 数据切割成补丁(patch)。
    # data是要切割的数据，patch_shape是补丁的形状(patch_size×patch_size×num_of_channels)，step是补丁之间的步长（可选）
    patches_img = patchify(data, shape, step=patch_size)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            dataset.append(single_patch_img)
    return np.array(dataset)

def save_patch(dataset,path):
    for i in range(dataset.shape[0]):
        patch = Image.fromarray(dataset[i,0])
        patch.save(path+str(i+1)+'.tif', format='TIFF')


# 重塑训练集数据形状，生成patch并将其保存到文件中。
RGB_dataset = datapatches(RGB_tf, (patch_size,patch_size,3),patch_size)
print('RGB_dataset的形状：',RGB_dataset.shape)


Nir_dataset = datapatches(Nir_tf, (patch_size,patch_size,3),patch_size)
print('Nir_dataset的形状：',Nir_dataset.shape)

save_patch(RGB_dataset,'../New_data/image/RGB_patch/RGB_')
save_patch(Nir_dataset,'../New_data/image/Nir_patch/Nir_')

