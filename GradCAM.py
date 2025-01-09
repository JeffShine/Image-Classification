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
        # 初始化GradCAM类，设置模型和目标层
        self.model = model
        self.target_layer = target_layer
        self.gradients = None  # 用于存储梯度
        self.activations = None  # 用于存储激活图
        self.hook_layers()  # 注册钩子函数

    def hook_layers(self):
        # 注册前向和后向钩子函数
        def forward_hook(module, input, output):
            # 前向传播时获取激活图
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            # 反向传播时获取梯度
            self.gradients = grad_out[0]

        # 在目标层注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        # 生成Grad-CAM热力图
        self.model.eval()  # 设置模型为评估模式
        output = self.model(input_image)  # 前向传播

        if target_class is None:
            # 如果未指定目标类别，选择预测概率最大的类别
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()  # 清除梯度
        target = output[0, target_class]  # 获取目标类别的输出
        target.backward()  # 反向传播以计算梯度

        # 获取梯度和激活图
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # 计算每个通道的权重
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        # 计算加权和
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 应用ReLU函数
        cam = np.maximum(cam, 0)
        # 调整热力图大小到输入图像大小
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        # 归一化热力图
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(image_path):
    # 预处理输入图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    image = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB
    image = transform(image).unsqueeze(0)  # 应用变换并增加批次维度
    return image

def show_cam_on_image(img, mask):
    # 将热力图叠加到原始图像上
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # 生成热力图
    heatmap = np.float32(heatmap) / 255  # 归一化热力图
    cam = heatmap + np.float32(img)  # 叠加热力图和原始图像
    cam = cam / np.max(cam)  # 归一化结果
    return np.uint8(255 * cam)  # 转换为8位图像

def main():
    # 主函数
    model = models.resnet18(weights=None)  # 加载ResNet18模型
    num_ftrs = model.fc.in_features  # 获取全连接层输入特征数
    model.fc = nn.Linear(num_ftrs, 100)  # 修改全连接层以适应100个类别
    model.load_state_dict(torch.load('save_dir/best_model_Adam_0.001.pth'))  # 加载预训练模型权重
    target_layer = model.layer4[1].conv2  # 选择目标层

    grad_cam = GradCAM(model, target_layer)  # 创建GradCAM对象

    image_dir = 'dataset/data/mini-imagenet/val'  # 图像目录
    subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir()]  # 获取所有子文件夹
    subfolder = random.choice(subfolders)  # 随机选择一个子文件夹
    image_files = [f for f in os.listdir(subfolder) if f.endswith(('.jpg', '.jpeg', '.png'))]  # 获取子文件夹中的所有图片
    image_path = os.path.join(subfolder, random.choice(image_files))  # 随机选择一张图片
    print(f"Selected image: {image_path}")
    input_image = preprocess_image(image_path)  # 预处理图像
    cam = grad_cam.generate_cam(input_image)  # 生成Grad-CAM热力图

    img = cv2.imread(image_path, 1)  # 读取图像
    img = cv2.resize(img, (224, 224))  # 调整图像大小
    img = np.float32(img) / 255  # 归一化图像

    cam_image = show_cam_on_image(img, cam)  # 叠加热力图
    os.makedirs('visualize', exist_ok=True)  # 创建visualize文件夹(如果不存在)
    
    # 保存GradCAM可视化结果到visualize文件夹
    cv2.imwrite(f'visualize/gradcam_{subfolder.split("/")[-1]}.jpg', cam_image)

if __name__ == '__main__':
    main()  # 执行主函数
