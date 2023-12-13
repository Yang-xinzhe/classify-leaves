import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid

data_frame = pd.read_csv('classify-leaves/train.csv')
label_mapping = {label: index for index, label in enumerate(data_frame['label'].unique())}

# Randomly sample 6 rows from the DataFrame
sampled_data = data_frame.sample(n=6, random_state=np.random.RandomState())

# Create a figure with 6 subplots in a 2x3 arrangement
plt.figure(figsize=(12, 8))

# Loop through the sampled data, load each image, and plot it
for index, (i, row) in enumerate(sampled_data.iterrows()):
    # Load the image
    img_path = os.path.join('classify-leaves', row['image'])
    image = Image.open(img_path)

    # Plot the image
    plt.subplot(2, 3, index + 1)
    plt.imshow(image)
    plt.title(f"Label: {row['label']}")
    plt.axis('off')

plt.tight_layout()
plt.show()

idx_to_label = {idx: label for label, idx in label_mapping.items()}
print(idx_to_label)

# 用于存储图像尺寸的列表
image_sizes = []

# 用于存储通道数量的列表
channel_counts = []

# 用于存储数据类型的集合
data_types = set()

# 指定图像文件夹路径
image_folder = 'classify-leaves/images'

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 遍历图像文件并分析它们
for image_file in image_files:
    # 构建图像文件的完整路径
    image_path = os.path.join(image_folder, image_file)

    # 打开图像
    image = Image.open(image_path)

    # 获取图像尺寸并添加到列表
    width, height = image.size
    image_sizes.append((height, width))

    # 获取通道数量（对于RGB图像，通常是3）并添加到列表
    channels = len(image.getbands())
    channel_counts.append(channels)

    # 获取图像的模式，这对应于数据类型
    data_type = image.mode
    data_types.add(data_type)

# 计算图像尺寸的统计信息
size_mean = np.mean(image_sizes, axis=0)
size_std = np.std(image_sizes, axis=0)
size_min = np.min(image_sizes, axis=0)
size_max = np.max(image_sizes, axis=0)

# 打印结果
print("图像尺寸统计信息:")
print(f"平均尺寸：{size_mean}")
print(f"尺寸标准差：{size_std}")
print(f"最小尺寸：{size_min}")
print(f"最大尺寸：{size_max}")

print("\n通道数量统计:")
channel_counts_unique = np.unique(channel_counts)
for channel_count in channel_counts_unique:
    count = channel_counts.count(channel_count)
    print(f"通道数 {channel_count} 的图像数量: {count}")

print("\n图像数据类型:")
for data_type in data_types:
    print(data_type)
