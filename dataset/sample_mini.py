import os
import random
import shutil
from pathlib import Path
import numpy as np

def sample_mini_imagenet(
    source_train_path: str,
    source_val_path: str,
    target_path: str,
    n_classes: int = 10,
    n_samples: int = 50
):
    """
    从 Mini-ImageNet 数据集中随机采样类别和图像
    
    参数:
    source_train_path: 训练集源路径
    source_val_path: 验证集源路径
    target_path: 采样后数据存放的目标路径
    n_classes: 要采样的类别数量
    n_samples: 每个类别要采样的图像数量
    """
    target_path = Path(target_path)
    
    # 获取所有训练集类别
    train_classes = [d for d in os.listdir(source_train_path) if os.path.isdir(os.path.join(source_train_path, d))]
    # 获取所有验证集类别
    val_classes = [d for d in os.listdir(source_val_path) if os.path.isdir(os.path.join(source_val_path, d))]
    
    # 随机选择训练集类别
    selected_train_classes = random.sample(train_classes, n_classes)
    # 从验证集中选择不同于训练集的类别
    remaining_val_classes = list(set(val_classes) - set(selected_train_classes))
    selected_val_out_classes = random.sample(remaining_val_classes, n_classes)
    
    # 采样训练集
    _sample_subset(source_train_path, target_path / 'train-10-50', selected_train_classes, n_samples)
    
    # 采样验证集（同类别）
    _sample_subset(source_val_path, target_path / 'val-10-50-in', selected_train_classes, n_samples)
    
    # 采样验证集（不同类别）
    _sample_subset(source_val_path, target_path / 'val-10-50-out', selected_val_out_classes, n_samples)
    
    print(f"\n完成!")
    print(f"训练集类别: {', '.join(selected_train_classes)}")
    print(f"验证集(out)类别: {', '.join(selected_val_out_classes)}")

def _sample_subset(source_path: str, target_path: str, selected_classes: list, n_samples: int):
    """辅助函数：采样并复制图像"""
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    
    for class_name in selected_classes:
        class_path = target_path / class_name
        class_path.mkdir(exist_ok=True)
        
        source_class_path = Path(source_path) / class_name
        images = list(source_class_path.glob('*.jpg'))
        
        # 如果可用图像数量少于请求的样本数，使用所有可用图像
        actual_n_samples = min(len(images), n_samples)
        selected_images = random.sample(images, actual_n_samples)
        
        for img_path in selected_images:
            shutil.copy2(img_path, class_path / img_path.name)
        
        print(f"已处理类别 {class_name}: {actual_n_samples} 张图像 (请求: {n_samples})")

# 使用示例
if __name__ == "__main__":
    source_train_path = "./dataset/data/mini-imagenet/train"
    source_val_path = "./dataset/data/mini-imagenet/val"
    target_path = "./dataset/data/mini-imagenet/sampled"
    
    sample_mini_imagenet(
        source_train_path=source_train_path,
        source_val_path=source_val_path,
        target_path=target_path,
        n_classes=10,
        n_samples=50
    )