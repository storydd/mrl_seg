from keras.utils import to_categorical
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
import cv2
import gc

scaler = MinMaxScaler()
patch_size = 256


def Datapatches(data, shape: tuple, patch_size, scaler=None):
    dataset = []
    # 在给定的代码中，patchify 函数用于将输入的 data 数据切割成补丁(patch)。
    # data是要切割的数据，patch_shape是补丁的形状(patch_size×patch_size×num_of_channels)，step是补丁之间的步长（可选）
    patches_img = patchify(data, shape, step=patch_size)
    '''补丁数组(patches_img)的形状将为 (num_patches_height, num_patches_width, num_patches_channel, patch_height, patch_width, num_channels)。
    num_patches_height 表示在垂直方向上的补丁数量。
    num_patches_width 表示在水平方向上的补丁数量。
    num_patches_channel 表示在通道方向上的补丁数量（在这种情况下，为1，因为每个补丁只有一个通道）。
    patch_height 表示每个补丁的高度。
    patch_width 表示每个补丁的宽度。
    num_channels 表示每个补丁的通道数。
    通过访问补丁数组的不同维度，可以获取每个补丁的数据。例如，patches[0, 0] 将获取第一个补丁的数据。'''
    print("patches_img的形状：",patches_img.shape)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i, j, :, :]

            if scaler:
                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds, for data only not for masks
                single_patch_img = (single_patch_img.astype('float32')) / 255.

            dataset.append(single_patch_img)
    return np.array(dataset)

def Save_npy(npdata, filename):
  np.save(f'{filename}',npdata)

def Pad(image_path,label_path,top, bottom, left, right):
    # 加载训练数据和标签
    data = np.load(image_path)
    label = np.load(label_path)

    # 填充图像和标签
    pading_data = cv2.copyMakeBorder(data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    pading_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    print('pading_data的形状：', pading_data.shape)
    print('pading_label的形状：', pading_label.shape)
    return pading_data,pading_label

'''    重塑训练集数据形状，生成patch并将其保存到文件中。'''
def Create_data_patch(pading_data,save_path):
    patched_dataset = Datapatches(pading_data, (patch_size,patch_size,3),patch_size,scaler)
    print('patched_dataset的形状：',patched_dataset.shape)
    np.save(save_path,patched_dataset)
    del pading_data,patched_dataset
    gc.collect()

def Create_label_patch(pading_label,save_path):
    # 从训练集标签生成 patches, 生成生成独热编码，（One-Hot Encoding）是一种将分类变量表示为二进制向量的方法,并保存到文件中.
    pading_labelset = Datapatches(pading_label, (patch_size, patch_size), patch_size)
    # to_categorical 是 Keras（一个深度学习库）中的一个函数，
    # 用于将整数标签转换为独热编码（one-hot encoding）的形式。它的目的是为了在神经网络训练中处理分类任务。
    pading_labels_cat = to_categorical(pading_labelset, num_classes=6)
    '''例如，假设有三个类别，则输出将如下所示：
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]]
    在这个例子中，原始的整数标签被转换为对应的独热编码。
    请注意，to_categorical 函数可以根据标签数组中的唯一值数量自动确定类别数量。它将整数标签转换为独热编码的形式，以便在分类任务中使用。'''
    print('pading_labels_cat的形状：', pading_labels_cat.shape)
    np.save(save_path,pading_labels_cat)
    del pading_labelset, pading_label, pading_labels_cat
    gc.collect()

if __name__ == "__main__":
    '填充后图片的路径'
    path1 = 'pad_data/2000image.npy'
    # '填充后标签的路径'
    path2 = 'pad_data/2000label.npy'
    data = np.load(path1)
    label = np.load(path2)
    print(data.shape)
    print(label.shape)
    '切片后后图片的保存路径'
    Create_data_patch(data,'patch_data/2000image.npy')
    # '切片后后标签的保存路径'
    Create_label_patch(label, 'patch_data/2000label.npy')
