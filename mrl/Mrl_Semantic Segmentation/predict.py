import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import tifffile as tiff
import torch.nn as nn
import torchvision.models
import torchvision.utils
from PIL import Image
import matplotlib.pyplot as plt
# from tqdm import tqdm
import itertools
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from sklearn.model_selection import train_test_split
import gc
import multiprocessing
import scipy.io as sio
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet


dataset_path = 'data/all/23year/23data.npy'

classes = [
"_background_",
"Emergent plant",
"Vegetation",
"Pit and pond",
"Muddy beach"];


# 从numpy文件中加载数据
def load_npy(path):
    return np.load(path)

def load_data():
    dataset_rw = load_npy(dataset_path).astype('float32')
    print('卫星遥感数据形状：',dataset_rw.shape)
    dataset_rw = torch.from_numpy(dataset_rw)#torch.Size([399, 256, 256, 3])
    return dataset_rw

X=load_data()
print(X.shape)

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Encoder(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
    super(Encoder, self).__init__()
    self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride, padding, groups, bias)
    self.block2 = BasicBlock(out_channels, out_channels, kernel_size, 1, padding, groups, bias)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)

    return x

class Decoder(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
               bias=False):
    super(Decoder, self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1, 1, 0, bias=bias),
                               nn.BatchNorm2d(in_channels // 4),
                               nn.ReLU(inplace=True), )
    self.tp_conv = nn.Sequential(
      nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size, stride, padding, output_padding, bias=bias),
      nn.BatchNorm2d(in_channels // 4),
      nn.ReLU(inplace=True), )
    self.conv2 = nn.Sequential(nn.Conv2d(in_channels // 4, out_channels, 1, 1, 0, bias=bias),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(inplace=True), )
# 前向传播
  def forward(self, x):
    x = self.conv1(x)
    x = self.tp_conv(x)
    x = self.conv2(x)

    return x



class LinkResNet(nn.Module):
  def __init__(self, n_channels=4, n_classes=5):
    super(LinkResNet, self).__init__()

    base = resnet.resnet34(pretrained=True)

    self.in_block = nn.Sequential(
      nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
      base.bn1,
      base.relu,
      base.maxpool
    )

    self.encoder1 = base.layer1
    self.encoder2 = base.layer2
    self.encoder3 = base.layer3
    self.encoder4 = base.layer4

    self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
    self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
    self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
    self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

    # Classifier
    self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True), )
    self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(inplace=True), )
    self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

  def forward(self, x):
    # Initial block
    x = self.in_block(x)

    # Encoder blocks
    e1 = self.encoder1(x)
    e2 = self.encoder2(e1)
    e3 = self.encoder3(e2)
    e4 = self.encoder4(e3)

    # Decoder blocks
    d4 = e3 + self.decoder4(e4)
    d3 = e2 + self.decoder3(d4)
    d2 = e1 + self.decoder2(d3)
    d1 = x + self.decoder1(d2)

    # Classifier
    y = self.tp_conv1(d1)
    y = self.conv2(y)
    y = self.tp_conv2(y)

    return y

# 搭建模型
# 设置超参数
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32 # 64, 32, 16
num_workers = 0
h = 256
w = 256
num_channels = 4
num_classes = 5
pin_memory = True
load_model = False
def get_link_resnet_model(freeze=False):
  model = LinkResNet(num_channels,num_classes)
  model = model.to(device)
  if freeze:
    for l in [model.in_block,model.encoder1, model.encoder2, model.encoder3, model.encoder4]:
      for param in l.parameters():
        param.requires_grad = False
  return model


models = {
    'link_res':{
        'model': get_link_resnet_model(),
        'path':  'Models/mymodel.pt'
    }
}

model_name = 'link_res'


models[model_name]['model'].load_state_dict(torch.load('Model/model_checkpoint8.pth'))


models[model_name]['model'].eval()
# 节约性能
print(X.shape)
print(type(X))
transposed_X = X.permute(0, 3, 1, 2)
transposed_X = transposed_X.cuda()
with torch.no_grad():
    output = models[model_name]['model'](transposed_X)
print(output)


print(output.shape)

output = output.permute(0, 2, 3, 1)
print(output.shape)


output=torch.softmax(output,axis=3)
print(output.shape)

output=torch.argmax(output,axis=3)
print(output.shape)

class_result=output.reshape([21, 19, 256, 256])


# 获取原始图像的大小
original_height = 5376
original_width = 4864
# 创建一个空白图像作为还原的结果
class_image = np.zeros((original_height, original_width))
class_image = torch.from_numpy(class_image)
class_image = class_image.cuda()

# 将每个小块放回到对应位置
for row in range(class_result.shape[0]):
    for col in range(class_result.shape[1]):
        patch = class_result[row, col]  # 获取当前小块
        class_image[row * 256: (row + 1) * 256, col * 256: (col + 1) * 256] = patch
# 打印还原后的图像形状
print(class_image.shape)


result = torch.stack([class_image, class_image, class_image], dim=2)
print(result.shape)


green=[0,255,0]
lgreen=[0,100,0]
red=[255,0,0]
blue=[0,0,255]
green = torch.tensor(green)
lgreen = torch.tensor(lgreen)
red = torch.tensor(red)
blue = torch.tensor(blue)
cuda_green = green.to('cuda')
cuda_lgreen = lgreen.to('cuda')
cuda_red = red.to('cuda')
cuda_blue = blue.to('cuda')



for i in range(5376):
    for j in range(4864):
        if result[i,j,0]==1:
            result[i,j]=cuda_green
        if result[i,j,0]==2:
            result[i,j]=cuda_lgreen
        if result[i,j,0]==3:
            result[i,j]=cuda_blue
        if result[i,j,0]==4:
            result[i,j]=cuda_red

print(result.shape)
result = result.cpu()
result = result.numpy()


result=result.astype(np.uint8)
tiff.imwrite('data/all/23year/23pre.tif', result)






