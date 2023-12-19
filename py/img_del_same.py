import os
import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from pathlib import Path  # 导入pathlib库
# pip install opencv-python numpy keras tensorflow

def extract_image_features(image_path):
    try:
        image = cv2.imread(image_path)  # 读取图片
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None

        image = cv2.resize(image, (256, 256))  # 缩放图片到统一尺寸
        image = image[16:240, 16:240]  # 裁剪中间区域(224x224)

        image = np.expand_dims(image, axis=0)  # 扩展维度以匹配模型输入要求
        image = preprocess_input(image)  # 预处理图片

        features = model.predict(image)  # 提取特征向量
        features /= np.linalg.norm(features)  # 归一化特征向量

        return features.flatten()  # 平铺特征向量
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None

def delete_duplicate_images():
    current_dir = Path(os.getcwd())  # 使用pathlib获取当前目录路径
    files = [f for f in current_dir.iterdir() if f.is_file()]  # 获取当前目录下的所有文件

    image_features = {}
    deleted_count = 0  # 记录删除的图片数量

    for file_path in files:
        if file_path.suffix in {".jpg", ".png"}:  # 筛选出图片文件
            try:
                image_feature = extract_image_features(str(file_path))  # 使用str()将Path对象转换为字符串
                if image_feature is not None:
                    is_duplicate = False
                    for existing_path, existing_feature in image_features.items():
                        distance = np.linalg.norm(existing_feature - image_feature)  # 计算欧氏距离
                        if distance < 0.3:  # 设定阈值来判断相似度，根据实际情况调整
                            is_duplicate = True
                            print(f"删除重复图片: {file_path}")
                            file_path.unlink()  # 使用unlink()删除文件
                            deleted_count += 1
                            break

                    if not is_duplicate:
                        image_features[file_path] = image_feature
            except Exception as e:
                print(f"处理图片时出错: {e}")

    print("已删除 {} 张重复图片".format(deleted_count))

# 加载预训练的ResNet50模型
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

delete_duplicate_images()
