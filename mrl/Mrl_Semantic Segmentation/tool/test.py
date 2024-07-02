import numpy as np
import skimage

image_2000= skimage.io.imread("../data/prediction/20001017NNDiffuseCLIP1.tif")
image_2013= skimage.io.imread("../data/prediction/20131005NNDiffuseCLIP1.tif")
image_2020= skimage.io.imread("../data/prediction/20201024NNDiffuseCLIP1.tif")
print(image_2000.shape)
print(image_2013.shape)
print(image_2020.shape)
'''
(3383, 3740, 3)H:201,W:100
(6764, 7479, 3)H:148,W:201
(6764, 7479, 3)
'''