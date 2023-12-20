from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image
import os

# pip install torch
# pip install transformers
# pip install Pillow

# 加载模型权重
processor = AutoImageProcessor.from_pretrained("model")
model = ResNetForImageClassification.from_pretrained("model")

# 指定图片目录
image_directory = "path/to/your/images"

# 遍历目录下的图片文件
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)

        # 对图片进行分类
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()

        # 获取分类标签
        label = model.config.id2label[predicted_label]

        # 构建新文件名，原文件名加分类标签
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{label}{ext}"
        new_path = os.path.join(image_directory, new_filename)

        # 移动或保存图片文件
        os.rename(image_path, new_path)