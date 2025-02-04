### 主要功能：

检测和删除重复图片：通过计算图片特征向量并使用欧氏距离判断图片是否重复。
记录重复图片对：重复图片的文件名会被写入到一个文本文件 duplicate_images.txt 中，方便查看哪些图片是重复的。

### 导入依赖：

os：用于文件操作，如获取当前目录和文件列表。
cv2：用于图像处理，读取图片并进行预处理（如调整大小和裁剪）。
numpy：用于处理图像特征向量和计算欧氏距离。
keras.applications.resnet50：加载预训练的ResNet50模型，用于提取图片的特征向量。

### 函数1

extract_image_features(image_path)：

读取图片并调整大小为 256x256 像素。
从中裁剪出 224x224 的中央区域。
扩展图像维度以适配 ResNet50 模型的输入格式。
对图像进行预处理，使其符合模型的输入要求。
通过 ResNet50 提取图片的特征向量，使用 avg 池化层来获取一个固定长度的特征向量。
归一化特征向量，使得特征向量的模长为 1。

### 函数2

delete_duplicate_images()：

获取当前工作目录下的所有文件。
对于每一个图片文件（.jpg 或 .png），提取其特征向量。
通过计算当前图片与已处理图片的特征向量之间的欧氏距离，判断是否为重复图片（距离小于 0.3）。
如果是重复图片，删除该图片，并记录当前图片与已存在重复图片的文件名对到 duplicate_pairs 列表。
将重复图片的文件名对（如 file1 与 file2 重复）写入 duplicate_images.txt 文件。

### 执行流程：

加载 ResNet50 模型（不包括顶层分类器，只使用特征提取部分）。
调用 delete_duplicate_images() 函数来执行重复图片检测和删除操作。

### 完整代码

```
import os
import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
 
# pip install opencv-python numpy keras tensorflow
 
def extract_image_features(image_path):
    image = cv2.imread(image_path)  # 读取图片
    image = cv2.resize(image, (256, 256))  # 缩放图片到统一尺寸
    image = image[16:240, 16:240]  # 裁剪中间区域(224x224)
    
    image = np.expand_dims(image, axis=0)  # 扩展维度以匹配模型输入要求
    image = preprocess_input(image)  # 预处理图片
    
    features = model.predict(image)  # 提取特征向量
    features /= np.linalg.norm(features)  # 归一化特征向量
    
    return features.flatten()  # 平铺特征向量
 
def delete_duplicate_images():
    current_dir = os.getcwd()  # 获取当前目录路径
    files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]  # 获取当前目录下的所有文件
 
    image_features = {}
    deleted_count = 0  # 记录删除的图片数量
    duplicate_pairs = []  # 用于保存重复图片的文件名对
 
    for file_name in files:
        if file_name.endswith(".jpg") or file_name.endswith(".png"):  # 筛选出图片文件
            file_path = os.path.join(current_dir, file_name)
            image_feature = extract_image_features(file_path)
 
            is_duplicate = False
            for existing_path, existing_feature in image_features.items():
                distance = np.linalg.norm(existing_feature - image_feature)  # 计算欧氏距离
                if distance < 0.3:  # 设定阈值来判断相似度，根据实际情况调整
                    is_duplicate = True
                    print(f"删除重复图片: {file_path}")
                    os.remove(file_path)
                    deleted_count += 1
                    # 记录重复的文件名对 (当前文件名和已有文件名)
                    duplicate_pairs.append((file_name, os.path.basename(existing_path)))
                    break
 
            if not is_duplicate:
                image_features[file_path] = image_feature
 
    # 将重复图片文件名对保存到txt文件
    if duplicate_pairs:
        with open("duplicate_images.txt", "w") as f:
            for file1, file2 in duplicate_pairs:
                f.write(f"{file1} 与 {file2} 重复\n")
    
    print("已删除 {} 张重复图片".format(deleted_count))
 
# 加载预训练的ResNet50模型
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
 
delete_duplicate_images()
```

