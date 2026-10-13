# mrl_seg

遥感影像语义分割 —— 滨海湿地地物分类（卫星 RGB + 近红外，4 通道输入）。

代码位于 [`mrl/Mrl_Semantic Segmentation/`](mrl/Mrl_Semantic%20Segmentation/)。

## 类别

| 标签 | 类别 |
| --- | --- |
| 0 | `_background_` 背景 |
| 1 | Emergent plant 挺水植物 |
| 2 | Vegetation 植被 |
| 3 | Pit and pond 坑塘 |
| 4 | Muddy beach 泥滩 |

## 数据与预处理

- `image_process.py`：`Merge_Channels()` 把 RGB 与近红外影像合并成 4 通道 `npy`；
  `Extract_marker_images()` 按标签把已标注区域从整幅影像中裁出。
- `Generate_Patches.py`：用 `patchify` 把整幅影像切成 **256×256** patch；
  标签经 `to_categorical(num_classes=6)` 转 one-hot；提供 `cv2.copyMakeBorder` 填充。
- `tif.py` / `create_label.py`：tif 读写；把预测结果叠加回 RGB 底图，输出标注结果图。

## 模型与训练

- **LinkResNet**（`predict.py` / `predict_with_sta.py`）：ImageNet 预训练 ResNet34 编码器 +
  转置卷积解码器；推理时逐 patch 预测 → 拼回整幅（5376×4864）→ 按类别着色输出。
- `Resnet_train.py` / `RI_train.py`：训练脚本。
- `plot_test.py`：绘制 train / val 损失曲线。
- `tool/`：公共工具代码。

## 环境

```
torch
torchvision
tensorflow / keras
patchify
tifffile
opencv-python
scikit-image
scikit-learn
numpy
matplotlib
```

## 使用

脚本内均为相对路径，请在 `mrl/Mrl_Semantic Segmentation/` 目录下执行，例如：

```
python Generate_Patches.py   # 生成 patch 与 one-hot 标签
python Resnet_train.py       # 训练
python predict.py            # 推理并输出分类 tif
```

> `data/`、`Model/`、`loss_data/`、`result/` 等数据与权重目录未入库，
> 需按各脚本中的路径自行准备。

---

> 本 README 依据仓库现有代码整理，细节如与实现不符请以代码为准。
