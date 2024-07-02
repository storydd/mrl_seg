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
from matplotlib import rcParams

#patch图片路径
train_dataset_path  = 'patch_data/2013image.npy'
#patch标签路径
train_labels_path = 'patch_data/2013label.npy'




# val_dataset_path = 'data/val/sval_data.npy'
# val_labels_path = 'data/val/sval_label.npy'

#自己的类别名
classes = [
'_background_',
'gengdi',
'tianranshidi',
'rengongshidi',
'jianzhuyongdi'
'zhibei'];

# 从numpy文件中加载数据
def load_npy(path):
    return np.load(path)

# 从numpy文件加载数据和标签
# 加载训练集
def load_train_data():
    train_dataset_rw = load_npy(train_dataset_path).astype('float32')
    print('训练集数据形状：',train_dataset_rw.shape)
    train_target_rw = load_npy(train_labels_path).astype('float32')
    print('训练集标签形状：',train_target_rw.shape)
    train_target_rw = np.rollaxis(train_target_rw,3,1)
    train_target_rw = torch.from_numpy(train_target_rw)#torch.Size([399, 256, 256, 11])
    return train_dataset_rw, train_target_rw

# def load_val_data():
#   val_dataset_rw = load_npy(val_dataset_path).astype('float32')
#   print('验证集数据形状：',val_dataset_rw.shape)
# #   val_dataset_rw = torch.from_numpy(val_dataset_rw)#torch.Size([399, 256, 256, 3])
#   val_target_rw = load_npy(val_labels_path).astype('float32')
#   print('验证集标签形状：',val_target_rw.shape)
#   val_target_rw = np.rollaxis(val_target_rw, 3, 1)
#   val_target_rw = torch.from_numpy(val_target_rw)#torch.Size([399, 256, 256, 11])
#   return val_dataset_rw, val_target_rw


# 设置超参数
lr = 1e-4
num_epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 # 64, 32, 16
num_workers = 0
h = 256
w = 256
num_channels = 3
num_classes = 6
pin_memory = True
load_model = False


# 数据集和加载程序
X_train,y_train = load_train_data()
# X_val,y_val=load_val_data()

y_train = y_train.numpy()
# y_val = y_val.numpy()


print(X_train.shape)
print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)


X_train_index=[]
X_val_index=[]
# X_test_index=[]
for i in range(810):
    if np.all(X_train[i] == 0):
        X_train_index.append(i)
print('训练集中空的patch的数量：',len(X_train_index))

# for j in range(399):
#     if np.all(X_val[j] == 0) or np.all(y_val[j, 3:,:] == 1):
#         X_val_index.append(j)
# print('验证集中空的patch的数量：',len(X_val_index))


X_train = np.delete(X_train, X_train_index,axis=0)
y_train = np.delete(y_train, X_train_index,axis=0)
# X_val = np.delete(X_val, X_val_index,axis=0)
# y_val = np.delete(y_val, X_val_index,axis=0)

X_train,X_val,y_train,y_val= train_test_split(X_train,y_train,test_size=0.2,random_state =20,shuffle = True)
print('训练集数据形状：',X_train.shape)
print('训练集标签形状：',y_train.shape)
print('测试集数据形状：',X_val.shape)
print('测试集数据形状：',y_val.shape)


dataset_len = {
    'train': len(X_train),
    'val': len(X_val)
}


class MyDataset(Dataset):
  def __init__(self, data, target, transform=None):
    self.data = data
    self.target = target
    self.transform = transform

  def __getitem__(self, index):
    x = self.data[index]
    y = self.target[index]

    if self.transform:
      x = self.transform(x)

    return x, y

  def __len__(self):
    return len(self.data)


def get_loaders(X_train, y_train, X_val, y_val):
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = MyDataset(X_train, y_train, trans)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
    )
    val_ds = MyDataset(X_val, y_val, trans)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
    )

    del X_train, y_train, X_val, y_val
    gc.collect()
    return train_loader, val_loader


train_loader,val_loader= get_loaders(X_train, y_train,X_val,y_val)

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
    def __init__(self, n_channels=3, n_classes=6):
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


# 计算模型在数据加载器上的准确率
import torchmetrics


def check_accuracy(loader, model, device="cuda", check_type='val'):
    model.eval()
    accuracy = {};
    acc_per_class = np.zeros(num_classes)
    for i in range(0, num_classes):
        accuracy[i] = torchmetrics.Accuracy(task='binary')
    # , ignore_index = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = nn.Softmax(dim=1)(model(x))
            preds = (preds > 0.5).float()
            preds = preds.cpu()
            for j in range(preds.shape[1]):
                for i in range(preds.shape[0]):
                    a1 = preds[i][j].flatten().to(torch.long)
                    a2 = y[i][j].flatten().to(torch.long)
                    accuracy[j](a1, a2)
    for i in range(1, num_classes):
        acc = accuracy[i].compute()
        acc_per_class[i] = acc

    model.train()
    return acc_per_class


# 训练一次模型
def train_model(loader, model, optimizer, loss_fn, scaler, phase='train'):
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.float().to(device=device)

        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast():
                predictions = model(data)
                # targets = targets.type(torch.int64)
                # print(targets.dtype)
                loss = loss_fn(predictions, targets)

            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        # statistics
        running_loss += loss.item() * data.size(0)

    return running_loss /dataset_len[phase]


# 迭代训练
def train(path2save, loader, model, optimizer, loss_fn, scaler):
    loss_values = {
        'train': [],
        'val': []
    }
    accuarcy_values = {
        'train': [],
        'val': []
    }
    # To save the best model
    # best_accuracy = 0

    # For Early Stopping
    # last_loss = None
    # patience = 2
    # triggertimes = 0
    best_loss=2
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_model(loader['train'], model, optimizer, loss_fn, scaler, phase='train')
        loss_values['train'].append(train_loss)
        model.eval()
        val_loss = train_model(loader['val'], model, optimizer, loss_fn, scaler, phase='val')
        loss_values['val'].append(val_loss)
        # if val_loss<best_loss:
        #     best_loss=val_loss
        #     torch.save(models[model_name]['model'].state_dict(), '2000(1).pth')
        print(f'Epoch {epoch}: Train Loss: {train_loss:.2f},Val loss: {val_loss:.2f}')

    return loss_values, accuarcy_values


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


multiprocessing.freeze_support()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, models[model_name]['model'].parameters()), lr=lr)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
loss_values[model_name], accuracy_values[model_name] = train(models[model_name]['path'], {'train':train_loader,'val':val_loader},
                                                             models[model_name]['model'], optimizer, loss_fn,
                                                             scaler)
rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.plot(loss_values[model_name]['train'], label="train loss")
plt.plot(loss_values[model_name]['val'], label = "val loss")
plt.xlabel('epoch', fontsize=18)
plt.ylabel('loss', fontsize=18)
plt.legend()
plt.show()
train_acc = check_accuracy(train_loader, models[model_name]['model'], device="cuda", check_type='train')
print("_______")
val_acc = check_accuracy(val_loader, models[model_name]['model'], device="cuda", check_type='val')
print('训练集准确率:',train_acc)
print('验证集准确率:',val_acc)

