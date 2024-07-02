import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
train_loss=np.load('loss_data/无人机trainloss.npy')
val_loss=np.load('loss_data/无人机valloss.npy')
plt.figure(figsize=(8,6))
plt.xlabel("epoch", fontsize=14)
plt.ylabel("CrossEntropyLoss",fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.plot(train_loss, label="train loss")
plt.plot(val_loss, label = "val loss")
plt.legend(fontsize=14)
plt.show()
# np.savetxt("loss_data/无人机trainloss.txt",train_loss,fmt="%2f",delimiter=",")
# np.savetxt("loss_data/无人机valloss.txt",val_loss,fmt="%2f",delimiter=",")