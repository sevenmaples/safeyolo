#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_torch.py
# @Time      :2025/6/9 14:07
# @Author    :雨霓同学
# @Function  :
import torch

print(f"pytorch版本: {torch.__version__}")
print(f"cuda版本: {torch.version.cuda}")
print(f"cudnn版本: {torch.backends.cudnn.version()}")
print(f"显卡是否可用: {torch.cuda.is_available()}")
print(f"可用的显卡数量: {torch.cuda.device_count()}")
print(f"可用的显卡信息: {torch.cuda.get_device_name(0)}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"当前显卡的索引: {torch.cuda.device(0)}")
print(f"当前显卡的显存使用情况: {torch.cuda.memory_allocated(0) / 1024 ** 3} GB")
print(f"当前显卡的cuda算力: {torch.cuda.get_device_capability(0)}")
print(f"当前显卡的显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")


if __name__ == "__main__":
    run_code = 0
