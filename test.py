import os
import pandas as pd
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import ResNet50_Weights

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('classify-leaves/train.csv')
label_mapping = {label: index for index, label in enumerate(train_df['label'].unique())}
idx_to_label = {idx: label for label, idx in label_mapping.items()}
print(idx_to_label)


class LeavesTestDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        """
        初始化函数
        Args:
        - data_frame (pandas.DataFrame): 包含图像文件名的DataFrame。
        - root_dir (string): 图像文件的根目录。
        - transform (callable, optional): 一个可选的转换函数，应用于图像。
        """
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        返回数据集中的样本数。
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。
        Args:
        - idx (int): 样本的索引。
        Returns:
        - image (PIL.Image): 转换后的图像。
        """
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)

        return image


# 加载模型
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)  # 或者你自己的模型
num_classes = len(idx_to_label)  # 根据你的类别数
print(num_classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model_best.pth'))
model.to(device)
model.eval()

# 准备测试数据
test_df = pd.read_csv('classify-leaves/test.csv')
test_transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = LeavesTestDataset(test_df, 'classify-leaves', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 进行预测
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# 转换数字标签为字符标签
predicted_labels = [idx_to_label[idx] for idx in predictions]

# 将预测结果添加到test.csv
test_df['label'] = predicted_labels
test_df.to_csv('test_with_predictions.csv', index=False)
