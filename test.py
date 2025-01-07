import argparse
import torch
import torchvision
# from torchvision import models
from models.resnet import resnet18_custom
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc

def extract_features_and_visualize(model, dataloader, device, method='pca'):
    model.eval()
    features = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            _, outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels_list.extend(labels.numpy())

    features = np.concatenate(features, axis=0)
    if method == 'pca': 
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
    elif method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features)
    else:
        raise ValueError(f"Invalid method: {method}")
    # 创建一个连续的颜色映射
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 10)))
    norm = BoundaryNorm(np.arange(11) - 0.5, cmap.N)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels_list, cmap=cmap, norm=norm, alpha=0.5)
    plt.colorbar(scatter, ticks=np.arange(10), label='Classes')  # 添加颜色条
    plt.title('2D PCA of Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # 保存图像到 visualize 文件夹
    plt.savefig(f'visualize/{method}_visualization.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on a test dataset.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the test dataset.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing.')
    parser.add_argument('--visualize', action='store_true', help='Visualize the features in 2D using PCA.')
    parser.add_argument('--method', type=str, default='pca', help='Method for feature visualization.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load test dataset
    test_dataset = torchvision.datasets.ImageFolder(root=args.data_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = resnet18_custom(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    # Extract features and visualize if requested
    if args.visualize:
        extract_features_and_visualize(model, test_loader, device, args.method)
    else:
        # Evaluate model
        accuracy = evaluate_model(model, test_loader, device)
        print(f'checkpoint: {args.ckpt} , dataset: {args.data_root} , Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()