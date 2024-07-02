import imgaug.augmenters as iaa
import numpy as np

# 创建数据增强器
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # 水平翻转
    iaa.Flipud(0.5),  # 垂直翻转
    iaa.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),  # 旋转和缩放
    iaa.GaussianBlur(sigma=(0.0, 1.0)),  # 高斯模糊
    iaa.AdditiveGaussianNoise(scale=(0, 0.05)),  # 加入高斯噪声
])

# 获取输入数据和标签
data = np.load('../data/all/21beach/21_data.npy')  # 输入数据
label = np.load('../data/all/21beach/21_label.npy')  # 标签数据

# input_data = data[0]
# label_data = label[0]
label = label.astype(np.int32)
print(data.shape)
print(label.shape)
# 执行数据增强
augmented_images, augmented_labels = augmenter(images=data, segmentation_maps=label)
print(len(augmented_images))
print(len(augmented_labels))
# 获取增强后的数据
augmented_input_data = augmented_images
augmented_label_data = augmented_labels
# 打印增强后的数据类型
print("增强后的输入数据形状:", type(augmented_input_data))
print("增强后的标签数据形状:", type(augmented_label_data))
# 打印增强后的数据形状
print("增强后的输入数据形状:", augmented_input_data.shape)
print("增强后的标签数据形状:", augmented_label_data.shape)
np.save('../data/all/21beach/augmented21_data.npy',augmented_input_data)
np.save('../data/all/21beach/augmented21_label.npy',augmented_label_data)