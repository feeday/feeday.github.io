import cv2
import sys
import dlib
import numpy as np
import os
import shutil
from tqdm import tqdm

# 筛选移到复制人脸图片

# 设置源文件夹和目标文件夹路径
source_dir = r"H:\Dataset\Logs\del\1" 
dest_dir = r"H:\Dataset\Logs\del\face"

# 如果目标文件夹不存在，则创建
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 初始化 dlib 的面部检测器
detector = dlib.get_frontal_face_detector()

def face_detected(img):
    # 在图像中检测人脸
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    # 如果检测到人脸且至少有一个人脸的尺寸大于等于指定像素，则返回 True；否则返回 False
    for face in faces:
        if face.width() >= 50 and face.height() >= 50:
            return True
    return False

# 从源文件夹获取所有图片文件
image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 跟踪已复制的图片数量
copied_count = 0

# 使用 tqdm 创建进度条
with tqdm(total=len(image_files), desc="复制进度", unit="file") as pbar:
    # 遍历所有图片进行人脸检测
    for file in image_files:
        image_path = os.path.join(source_dir, file)
        # 使用 Python 的文件读取方法打开图像，然后用 cv2.imdecode 解码图像
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"读取图像时出错: {image_path}，错误: {e}")
            pbar.update(1)
            continue

        # 检查图像是否成功加载
        if img is None:
            #print(f"无法加载图像: {image_path}")
            pbar.update(1)
            continue

        # 检测到满足条件的人脸
        if face_detected(img):
            # 将检测到人脸的图片复制到目标文件夹
            dest_file = os.path.join(dest_dir, file)
            shutil.move(image_path, dest_file)  # 复制 shutil.copy
            copied_count += 1

        # 更新进度条的描述以包含复制的总数
        pbar.set_description(f"复制进度 ({copied_count} 张图片已复制)")
        # 更新进度条
        pbar.update(1)

print(f"人脸检测和复制完成，共复制了 {copied_count} 张图片。")
