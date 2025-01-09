import torch
import torchvision
from torchvision import models

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm
import wandb
import os
# Hyperparameters
num_epochs = 128
batch_size = 512

# Experiment settings
learning_rates = [0.01, 0.001, 0.0001] 
optimizers = ['Adam'] 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 数据增强和归一化的预处理步骤:
# 1. RandomResizedCrop(224): 随机裁剪图像并调整大小到224x224
# 2. RandomHorizontalFlip(): 随机水平翻转图像,增加数据多样性
# 3. ToTensor(): 将PIL图像转换为Tensor格式
# 4. Normalize(): 使用ImageNet数据集的均值和标准差进行归一化
#    - 均值: [0.485, 0.456, 0.406] 
#    - 标准差: [0.229, 0.224, 0.225]

# Load dataset
train_dataset = torchvision.datasets.ImageFolder(root='./dataset/data/mini-imagenet/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torchvision.datasets.ImageFolder(root='./dataset/data/mini-imagenet/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
save_path = "save_dir"
def evaluate_model(dataloader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Calculate loss if required
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

            # Calculate accuracy if required
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = total_loss / total_batches if total_batches > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    return loss, acc

for lr in learning_rates:
    for optimizer_name in optimizers:
        print(f"lr: {lr}, optimizer: {optimizer_name}")
        run = wandb.init(
            project="mini-imagenet",
            config={
                "learning_rate": lr,
                "optimizer": optimizer_name,
                "epochs": num_epochs,
                "batch_size": batch_size,
            },
        )
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
        
        model.to(device)

        # Set optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)

        best_acc = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                wandb.log({"running_loss": loss.item()})

            print(f'Epoch [{epoch+1}/{num_epochs}], Running Loss: {running_loss/len(train_loader):.4f}')

            loss_train, acc_train = evaluate_model(train_loader)
            loss_val, acc_val = evaluate_model(val_loader)

            wandb.log({
                "train_loss": loss_train,
                "train_acc": acc_train,
                "val_loss": loss_val,
                "val_acc": acc_val,
            })  

            if acc_val > best_acc:
                best_acc = acc_val
                torch.save(model.state_dict(), os.path.join(save_path, f"best_model_{optimizer_name}_{lr}.pth"))
            
        wandb.finish()

print('Training complete')