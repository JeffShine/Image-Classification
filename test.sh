#!/bin/bash

# 获取参数
CHECKPOINT_PATH=$1
DATA_ROOT=$2

# 运行 Python 测试脚本
# # mini-imagenet-whole
# python test.py --ckpt "save_dir/best_model_SGD_0.01.pth" --data_root "dataset/data/mini-imagenet/val"
# python test.py --ckpt "save_dir/best_model_SGD_0.001.pth" --data_root "dataset/data/mini-imagenet/val"
# python test.py --ckpt "save_dir/best_model_SGD_0.0001.pth" --data_root "dataset/data/mini-imagenet/val"
python test.py --ckpt "save_dir/best_model_Adam_0.01.pth" --data_root "dataset/data/mini-imagenet/val"
python test.py --ckpt "save_dir/best_model_Adam_0.001.pth" --data_root "dataset/data/mini-imagenet/val"
python test.py --ckpt "save_dir/best_model_Adam_0.0001.pth" --data_root "dataset/data/mini-imagenet/val"