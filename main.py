import torch
import torch.nn as nn
from torch import utils
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
num_classes = 176
batch_size = 32
learning_rate = 0.001

writer = SummaryWriter('guns/leaves_classifier')

data_frame = pd.read_csv('classify-leaves/train.csv')

# test_data_frame = pd.read_csv('classify-leaves/sample_submission.csv')

train_data, val_data = train_test_split(data_frame, test_size=0.2)

label_mapping = {label: index for index, label in enumerate(data_frame['label'].unique())}


class LeavesDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # 确保图像是RGB格式
        label_name = self.data_frame.iloc[idx, 1]
        label = self.label_mapping[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = LeavesDataset(data_frame=data_frame, root_dir='classify-leaves', transform=transform)
val_dataset = LeavesDataset(data_frame=val_data, root_dir='classify-leaves', transform=transform)
# test_dataset = LeavesDataset(data_frame=test_data_frame, root_dir='classify-leaves', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# class Conv4F(nn.Module):
#     def __init__(self, num_classes):
#         super(Conv4F, self).__init__()
#         # 减少了卷积层的深度和神经元数量
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # 增加Dropout来减少过拟合
#         self.dropout = nn.Dropout(0.5)
#         # 调整全连接层
#         self.fc1 = nn.Linear(128 * 32 * 32, 512)  # 假设输入图像尺寸为256x256
#         self.fc2 = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#
#         x = x.view(-1, 128 * 32 * 32)  # 扁平化张量
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)  # 在全连接层后加入Dropout
#         x = self.fc2(x)
#         return x

# model = Conv4F(num_classes=176).to(device)
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # num_classes 是您的数据集的类别数量
model.fc = model.fc.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

best_val_acc = 0.0

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f'Epoch {epoch + 1}, Validation Accuracy: {val_acc}')

    # 记录验证准确率到TensorBoard
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # 保存模型如果验证准确率提高
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model_best.pth')
        print(f'Model improved and saved at epoch {epoch + 1} with Validation Accuracy: {val_acc}')

writer.close()
