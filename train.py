import torch
import torchvision
from torchvision import models

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm

# Hyperparameters
num_epochs = 64
batch_size = 128

# Experiment settings
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1] 
optimizers = ['SGD'] 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_dataset = torchvision.datasets.ImageFolder(root='./dataset/data/mini-imagenet/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torchvision.datasets.ImageFolder(root='./dataset/data/mini-imagenet/val', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()

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

        # Load pre-trained ResNet model
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
        
        model.to(device)

        # Set optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)

        # Create a unique log file name
        log_filename = f'logs/bs{batch_size}_lr{lr}_opt{optimizer_name}.csv'
        
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

            print(f'Epoch [{epoch+1}/{num_epochs}], Running Loss: {running_loss/len(train_loader):.4f}')
    
            loss_train, acc_train = evaluate_model(train_loader)
            loss_val, acc_val = evaluate_model(val_loader)
            with open(log_filename, 'a') as f:
                f.write(f'{epoch+1},{loss_train:.4f},{acc_train:.4f},{loss_val:.4f},{acc_val:.4f}\n')

print('Training complete')