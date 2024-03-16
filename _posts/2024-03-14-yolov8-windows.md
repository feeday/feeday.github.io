---
layout:     post
title:      Windows 安装 yolov8 
subtitle:   yolov8 intsall windows
date:       2024-03-14
author:     Puck
catalog: true
tags:
    - Python
---

## 安装软件
https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

- [x] Add Miniconda3 to my PATH environment variable

## 安装环境

```
conda create -n yolov8 python=3.8
conda activate yolov8
```

## 更换清华源

https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 安装软件

https://pytorch.org/get-started/previous-versions/

NVIDIA 控制面板 - 系统信息 - 组件
文件名 - NVCUDA64.DLL 
产品名称  -  NVIDIA CUDA 12.4.99 driver

### 查看显卡支持的版本
```
nvidia-smi
```

CUDA Version: 12.4

### 安装注意事项

```
Pytorch安装注意事项
16XX的显卡，安装cu102的版本，否则可能训练出现问题
30XX、40XX显卡，要安装cu111以上的版本，否则无法运行
```

### 用命令安装
CUDA 10.2

```
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
```

CUDA 11.6

```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
## 下载解压代码

https://github.com/ultralytics/ultralytics

```
(yolov8) F:\ultralytics-main>
pip install -e .
pip list
```

## 命令提示符运行代码

```
conda init cmd.exe
conda activate yolov8
```

## 测试代码

### 直接执行
```
(yolov8) F:\ultralytics-main>yolo predict model=yolov8n.pt source=ultralytics/assets/bus.jpg
```
### 脚本代码运行
```
from ultralytics import YOLO

yolo = YOLO("./yolov8n.pt",task="detect")

result = yolo(source="./ultralytics/assets/bus.jpg", save=True, conf=0.5)

# source="./ultralytics/assets/bus.jpg" 检测图片
# source="./demo1.mp4" 检测视频
# source="screen"  检测电脑桌面
# source=0  检测摄像头

# conf=0.5 值越小 检测框越多
# iou=0.7  值越小 检测框越少
```

### VSCODE 代码调试

```
pip install jupyterlab
```

```
from ultralytics import YOLO
yolo = YOLO("./yolov8n.pt",task="detect")
result = yolo(source="./ultralytics/assets/bus.jpg")
```

```
result[0]
```

```
# 检测结果可视化 Jupyter中使用，要可视化模型预测结果，一定要设置%matplotlib inline，否则无法使用
import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(result[0].plot()[:,:,::-1])
```

```
# boxes信息
result[0].boxes.xywh.cpu().numpy()
```


### 显卡测试运行结果

NVIDIA GeForce RTX 3060 Laptop GPU

```
image 1/1 F:\ultralytics-main\ultralytics\assets\bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 20.0ms
Speed: 2.3ms preprocess, 20.0ms inference, 7.0ms postprocess per image at shape (1, 3, 640, 480)
Results saved to runs\detect\predict2
💡 Learn more at https://docs.ultralytics.com/modes/predict
```

### 视频提取图片

```
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
# 打开视频
video = cv2.VideoCapture("./boy.mp4")
# 读取一帧
ret, frame =video.read()
plt.imshow(frame)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

video = cv2.VideoCapture("./demo.mp4")
num = 0  # 计数器
save_step = 30  # 间隔帧
while True:
    ret,frame = video.read()
    if not ret:
        break
    num += 1
    if num % save_step == 0:
        cv2.imwrite("./demo_images/"+ str(num)+".jpg",frame)
```

# 标注工具

```
pip install labelimg
labelimg
```

# 公开数据集

https://public.roboflow.com/object-detection

https://universe.roboflow.com/


# 数据集准备训练

```
# images:存放图片
#     train:训练集图片
#     val:验证集图片

# labels:存放标签
#     train:训练集标签文件，要与训练集图片名称--对应
#     val:验证集标签文件，要与验证集图片名称--对应

# F:\ultralytics-main\datasets\数据集文件名
# F:\ultralytics-main\ultralytics\cfg\datasets 官方描述文件

F:\ultralytics-main\datasets\coco.yaml

path: ../datasets/coco # dataset root dir
train: train2017.txt # train images (relative to 'path') 118287 images
val: val2017.txt # val images (relative to 'path') 5000 images
test: test-dev2017.txt # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
names:
  0: person
  1: bicycle
  2: car
```
### 开始训练

https://docs.ultralytics.com/zh/modes/train/#resuming-interrupted-trainings

```
yolo task=detect mode=train model=./yolov8n.pt data=yolo-coco.yaml epochs-30 workers=1 batch=16
```

## 视频教程

https://www.bilibili.com/video/BV13V4y1S7MK
