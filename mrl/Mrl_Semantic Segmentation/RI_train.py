import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms, datasets, models
import tifffile as tiff
import torchvision.models
import torchvision.utils
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import gc
import torchmetrics
from Unet_Attention import *
from Unet import *
import torch.nn as nn



train_dataset_path  =  'data/UAV/train/2_train/ep_dataset.npy'
train_labels_path = 'data/UAV/train/2_train/ep_label.npy'
train_dataset_path2  = 'data/UAV/train/1_train/train_data.npy'
train_labels_path2 = 'data/UAV/train/1_train/train_label.npy'
train_dataset_path3  = 'data/UAV/AL/AL1/AL1_data.npy'
train_labels_path3 = 'data/UAV/AL/AL1/AL1_label.npy'
train_dataset_path4  = "data/water/patched_dataset.npy"
train_labels_path4 = 'data/water/pading_labels.npy'

# train_dataset_path2 = 'data/water/patched_dataset.npy'
# train_labels_path2 = 'data/water/pading_labels.npy'
val_dataset_path = 'data/UAV/val/val_data.npy'
val_labels_path = 'data/UAV/val/val_label.npy'

classes = [
"_background_",
"Emergent plant",
"Vegetation",
"Pit and pond",
"Muddy beach"];



# 从numpy文件中加载数据
def load_npy(path):
    return np.load(path)



# 从numpy文件加载数据和标签
# 加载训练集
def load_train_data():
    train_dataset_rw = load_npy(train_dataset_path).astype('float32')
    train_dataset_rw2 = load_npy(train_dataset_path2).astype('float32')
    train_dataset_rw3 = load_npy(train_dataset_path3).astype('float32')
    train_dataset_rw4 = load_npy(train_dataset_path4).astype('float32')
    print('训练集1数据形状：',train_dataset_rw.shape)
    print('训练集2数据形状：',train_dataset_rw2.shape)
    print('训练集3数据形状：', train_dataset_rw3.shape)
    print('训练集4数据形状：', train_dataset_rw4.shape)
    train_dataset_rw = np.concatenate((train_dataset_rw,train_dataset_rw2),axis=0)
    train_dataset_rw = np.concatenate((train_dataset_rw, train_dataset_rw3), axis=0)
    train_dataset_rw = np.concatenate((train_dataset_rw, train_dataset_rw4), axis=0)
    train_dataset_rw = np.rollaxis(train_dataset_rw, 3, 1).astype('float32')
    print('训练集数据形状：',train_dataset_rw.shape)
    train_target_rw = load_npy(train_labels_path).astype('float32')
    train_target_rw2 = load_npy(train_labels_path2).astype('float32')
    train_target_rw3 = load_npy(train_labels_path3).astype('float32')
    train_target_rw4 = load_npy(train_labels_path4).astype('float32')
    print('训练集1标签形状：',train_target_rw.shape)
    print('训练集2标签形状：',train_target_rw2.shape)
    print('训练集3标签形状：', train_target_rw3.shape)
    print('训练集4标签形状：', train_target_rw4.shape)
    train_target_rw = np.concatenate((train_target_rw,train_target_rw2),axis=0)
    train_target_rw = np.concatenate((train_target_rw, train_target_rw3), axis=0)
    train_target_rw = np.concatenate((train_target_rw, train_target_rw4), axis=0)
    train_target_rw = np.rollaxis(train_target_rw,3,1)
    train_target_rw = torch.from_numpy(train_target_rw)
    print('训练集标签形状：', train_target_rw.shape)
    return train_dataset_rw, train_target_rw
# 加载验证集
def load_val_data():
  val_dataset_rw = load_npy(val_dataset_path).astype('float32')
  val_dataset_rw = np.rollaxis(val_dataset_rw, 3, 1).astype('float32')
  print('验证集数据形状：',val_dataset_rw.shape)
#   val_dataset_rw = torch.from_numpy(val_dataset_rw)#torch.Size([399, 256, 256, 3])
  val_target_rw = load_npy(val_labels_path).astype('float32')
  print('验证集标签形状：',val_target_rw.shape)
  val_target_rw = np.rollaxis(val_target_rw, 3, 1)
  val_target_rw = torch.from_numpy(val_target_rw)#torch.Size([399, 256, 256, 11])
  return val_dataset_rw, val_target_rw


# 设置超参数
lr = 0.001
num_epochs = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
# batch_size = 8 # 64, 32, 16
num_workers = 0
h = 256
w = 256
num_channels = 4
num_classes = 5
pin_memory = True
load_model = False
random_state=10


# 数据集和加载程序
X_train,y_train = load_train_data()
X_val,y_val=load_val_data()



y_train = y_train.numpy()
y_val = y_val.numpy()

print('训练集数据形状：',X_train.shape)
print('训练集标签形状：',y_train.shape)
print('验证集数据形状：',X_val.shape)
print('训练集标签形状：',y_val.shape)



def del_black(train_num,val_num,X_train,y_train,X_val,y_val):
    X_train_index=[]
    X_val_index=[]
    # X_test_index=[]
    for i in range(train_num):
        if np.all(X_train[i] == 0):
            X_train_index.append(i)
    print('训练集中空的patch的数量：',len(X_train_index))

    for j in range(val_num):
        if np.all(X_val[j] == 0):
            X_val_index.append(j)
    print('验证集中空的patch的数量：',len(X_val_index))

    X_train = np.delete(X_train, X_train_index, axis=0)
    y_train = np.delete(y_train, X_train_index, axis=0)
    X_val = np.delete(X_val, X_val_index, axis=0)
    y_val = np.delete(y_val, X_val_index, axis=0)

    print('去除无用patch后的训练集数据形状：', X_train.shape)
    print('去除无用patch后的训练集标签形状：', y_train.shape)
    print('去除无用patch后的验证集数据形状：', X_val.shape)
    print('去除无用patch后的训练集标签形状：', y_val.shape)
    # X_test = np.delete(X_test, X_test_index,axis=0)
    return X_train,y_train,X_val,y_val

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
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    # ])


    train_ds = MyDataset(X_train, y_train)

    train_loader = DataLoader(
        train_ds,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
    )
    val_ds = MyDataset(X_val, y_val)
    sampler =  SequentialSampler(val_ds)
    val_loader = DataLoader(
        val_ds,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        shuffle=True,
        sampler=sampler
    )

    del X_train, y_train, X_val, y_val
    gc.collect()
    return train_loader, val_loader


X_train,y_train,X_val,y_val=del_black(2756,1073,X_train,y_train,X_val,y_val)
train_loader,val_loader= get_loaders(X_train, y_train,X_val,y_val)

# # 创建模型
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)
#         self.up1 = Up(1024, 512, bilinear)
#         self.up2 = Up(512, 256, bilinear)
#         self.up3 = Up(256, 128, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
class U_Net_v1(nn.Module):  # 添加了空间注意力和通道注意力
    def __init__(self, img_ch=4, output_ch=5):
        super(U_Net_v1, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)  # 64
        self.Conv2 = conv_block(ch_in=64, ch_out=128)  # 64 128
        self.Conv3 = conv_block(ch_in=128, ch_out=256)  # 128 256
        self.Conv4 = conv_block(ch_in=256, ch_out=512)  # 256 512
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)  # 512 1024

        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)  # 1024 512
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)  # 512 256
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)  # 256 128
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)  # 128 64
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)  # 64

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x1 = self.cbam1(x1) + x1

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.cbam2(x2) + x2

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.cbam3(x3) + x3

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.cbam4(x4) + x4

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



# 计算模型在数据加载器上的准确率
def check_accuracy(loader, model, device="cuda", check_type='val'):
    model.eval()
    accuracy = {};
    acc_per_class = np.zeros(num_classes)
    for i in range(0, num_classes):
        accuracy[i] = torchmetrics.Accuracy(task='binary', ignore_index=0)

    with torch.no_grad():
        model.eval()
        for x, y in loader:
            x = x.to(device)
            preds = nn.Softmax(dim=1)(model(x))
            print(preds[0, :, 3, 4])
            preds = (preds > 0.5).float()
            print(preds[0,:,3,4])
            # print()
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
    for data, targets in loader:
        data = data.to(device=device)
        targets = targets.float().to(device=device)
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(phase == 'train'):
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        # statistics
        running_loss += loss.item()

    return running_loss/dataset_len[phase]


# 迭代训练
def train(loader, model, optimizer, loss_fn, scaler):
    loss_values = {
        'train': [],
        'val': []
    }
    accuarcy_values = {
        'train': [],
        'val': []
    }

    for epoch in range(num_epochs):
        best_loss=10
        model.train()
        train_loss = train_model(loader['train'], model, optimizer, loss_fn, scaler, phase='train')
        loss_values['train'].append(train_loss)
        model.eval()
        val_loss = train_model(loader['val'], model, optimizer, loss_fn, scaler, phase='val')
        loss_values['val'].append(val_loss)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.2f},Val loss: {val_loss:.2f}')
        if val_loss<best_loss:
            torch.save(model.state_dict(),'Unet_params.pth')
            best_loss=val_loss
    return loss_values, accuarcy_values

# 搭建模型
def get_Unet_model():
    model = U_Net_v1(4, 5)
    model = model.to(device)
    return model

net = get_Unet_model()
loss_values = {}
accuracy_values = {}
print(net)




optimizer = optim.Adam(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
loss_values['Unet'], accuracy_values['Unet'] = train({'train':train_loader,'val':val_loader},net, optimizer, loss_fn,scaler)
plt.plot(loss_values['Unet']['train'], label="train loss")
plt.plot(loss_values['Unet']['val'], label = "val loss")
plt.legend()
plt.show()
plt.savefig('result/loss.png')
train_acc = check_accuracy(train_loader, net, device="cuda", check_type='train')
val_acc = check_accuracy(val_loader, net, device="cuda", check_type='val')
print('训练集准确率:',train_acc)
print('验证集准确率:',val_acc)