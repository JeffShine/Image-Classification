import argparse
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc

def main():
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on a test dataset.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the test dataset.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing.')
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
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(test_dataset.classes))
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    # Evaluate model
    accuracy = evaluate_model(model, test_loader, device)
    print(f'checkpoint: {args.ckpt} , dataset: {args.data_root} , Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()