import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import torch

import tifffile as tiff
import torch.nn as nn
import torchvision.models
import torchvision.utils
from sklearn.model_selection import train_test_split

dataset_path = '../data/all/21_all/21_alldata.npy'
labels_path = '../data/all/21_all/21_alllabel.npy'



classes = [
"_background_",
"Emergent plant",
"Vegetation",
"Pit and pond",
"Muddy beach"];

# 从numpy文件中加载数据
def load_npy(path):
    return np.load(path)

# 加载数据
def load_val_data():
  val_dataset_rw = load_npy(dataset_path).astype('float32')
  val_target_rw = load_npy(labels_path).astype('float32')
  val_target_rw = torch.from_numpy(val_target_rw)#torch.Size([399, 256, 256, 11])
  return val_dataset_rw,val_target_rw


# 设置超参数
lr = 1e-3
num_epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8 # 64, 32, 16
num_workers = 0
h = 256
w = 256
num_channels = 4
num_classes = 5
pin_memory = True
load_model = False
random_state=10


# 数据集和加载程序
X_val,y_val=load_val_data()
y_val = y_val.numpy()
print('验证集数据形状：',X_val.shape)
print('验证集标签形状：',y_val.shape)
X_val_index=[]
for j in range(399):
    if np.all(X_val[j] == 0):
        X_val_index.append(j)
print('验证集中空的patch的数量：',len(X_val_index))
X_val = np.delete(X_val, X_val_index,axis=0)
y_val = np.delete(y_val, X_val_index,axis=0)
print('去除全是背景后的验证集数据形状：',X_val.shape)
print('去除全是背景后的验证集标签形状：',y_val.shape)
# x_train,x_test,y_train,y_test= train_test_split(X_val,y_val,test_size=0.1,random_state = 1,shuffle = True)
x_test = X_val
y_test = y_val
print('测试集数据形状：',x_test.shape)
print('测试集数据形状：',y_test.shape)




import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet

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



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


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

        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)

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
        'path':  'Models/linknet_resnet_with_cel_check_points.pt'
    }
}
loss_values = {}
accuracy_values = {}

model_name = 'link_res'


# multiprocessing.freeze_support()
models[model_name]['model'].load_state_dict(torch.load('../all_model.pth'))
models[model_name]['model'].eval()
# 预测
x_test = torch.tensor(x_test)
transposed_X_test = x_test.permute(0, 3, 1, 2)
transposed_X_test = transposed_X_test.cuda()
with torch.no_grad():
    output = models[model_name]['model'](transposed_X_test)
print('模型输出形状：',output.shape)




output = output.permute(0, 2, 3, 1)
output=torch.softmax(output,axis=3)
# output = (output > 0.5).float()
print('模型输出softmax激活后的形状：',output.shape)
output = torch.argmax(output,axis=3)
output=output.to('cpu')
output = output.numpy()
print('预测标签的形状：',output.shape)
print('真实标签的形状：',y_test.shape)
y_test = np.argmax(y_test,axis=3)
print('真实标签的形状：',y_test.shape)

flatten_output=output.reshape([13631488])
flatten_y_test=y_test.reshape([13631488])





def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize=None)
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()
    # plt.show()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = int(float(format('%.2f' % cm[j, i]))/10)
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)


    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


draw_confusion_matrix(label_true=flatten_y_test,
                      label_pred=flatten_output,
                      label_name=["background","Emergent plant","Vegetation","Pit and pond","Muddy beach"],
                      title="Confusion Matrix",
                      pdf_save_path="../result/noA/Confusion_Matrix_S_num2.jpg",
                      dpi=300)