import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import random
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model.load_state_dict(torch.load('save_dir/best_model_Adam_0.001.pth'))
    target_layer = model.layer4[1].conv2

    grad_cam = GradCAM(model, target_layer)

    image_dir = 'dataset/data/mini-imagenet/val'
    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir()]
    # 随机选择一个子文件夹
    subfolder = random.choice(subfolders)
    # 获取子文件夹中的所有图片
    image_files = [f for f in os.listdir(subfolder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # 从选中的子文件夹中随机选择一张图片
    image_path = os.path.join(subfolder, random.choice(image_files))
    print(f"Selected image: {image_path}")
    input_image = preprocess_image(image_path)
    cam = grad_cam.generate_cam(input_image)

    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255

    cam_image = show_cam_on_image(img, cam)
    # 创建visualize文件夹(如果不存在)
    os.makedirs('visualize', exist_ok=True)
    
    # 保存GradCAM可视化结果到visualize文件夹
    cv2.imwrite(f'visualize/gradcam_{subfolder.split("/")[-1]}.jpg', cam_image)

if __name__ == '__main__':
    main()
