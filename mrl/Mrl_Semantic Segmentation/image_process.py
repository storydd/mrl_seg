import scipy.io as sio
import skimage.io
import numpy as np
import matplotlib as plt
import tifffile as tf
import gc
import cv2

# train_image = skimage.io.imread("data/卫星遥感RGB图像.tif")
# val_image = skimage.io.imread("data/卫星遥感RGB图像.tif")
# train_label=np.load('data/train/SegmentationClass/train.npy')
val_label=np.load('data/val/SegmentationClass/val.npy')
water_image_path='data/water/image/16wei.tif'
# print(image.shape)
# print(train_label.shape)
# print(val_label.shape)

'''根据便签将标记好的数据从整幅图像中提取出来'''
def Extract_marker_images(image_path,label_path,save_path):
    image = skimage.io.imread(image_path)
    label=np.load(label_path)
    print('图像大小为:', image.shape)
    print('标签大小为:', label.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if label[i, j] == 0:
                image[i, j] = [0, 0, 0]
    image = image.astype(np.int8)
    print('提取后的图像大小为:', image.shape)
    tf.imwrite(save_path, image)

def Merge_Channels(rgbimage_path,nirimage_path,save_path):
    rgb_image = skimage.io.imread(rgbimage_path)
    nir_image = skimage.io.imread(nirimage_path)
    nir_image = nir_image[:,:,0].reshape([nir_image.shape[0],nir_image.shape[1],1])
    print('rgb图像大小:', rgb_image.shape)
    print('nir图像大小:', nir_image.shape)
    Merge_image = np.concatenate((rgb_image, nir_image), axis=2)
    print(Merge_image.shape)
    np.save(save_path, Merge_image)
    print('4通道数据存放在:'+save_path)
    print('合并后图片形状:',Merge_image.shape)

if __name__ == "__main__":
    path1 = 'data/water/image/16wei.tif'
    path2 = 'data/water/image/8weinir.tif'
    path3 = 'data/water/SegmentationClass/16wei.npy'
    path4 = 'data/water/data/Extract_rgb.tif'
    path5 = 'data/water/data/Extract_nir.tif'
    path6 = 'data/water/data/4td.npy'
    path7 = 'data/all/21_all/RGBimage.tif'
    path8 = 'data/all/21_all/NIRimage.tif'
    path9 = 'data/all/21_all/RGBN.npy'
    # Extract_marker_images(path1, path3, path4)
    # Extract_marker_images(path2, path3, path5)
    Merge_Channels(path7, path8, path9)






