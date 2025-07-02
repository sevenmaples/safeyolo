import os
import random
from pathlib import Path
import shutil
import yaml

# 配置
IMG_DIR = Path('raw/images')
LABEL_DIR = Path('raw/yolo_staged_labels')
OUT_ROOT = Path('dataset')
SPLITS = ['train', 'val', 'test']
SPLIT_RATIO = [0.8, 0.1, 0.1]  # 8:1:1

random.seed(42)

# 1. 收集所有图片文件
img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
all_imgs = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in img_exts]

# 2. 打乱并划分
random.shuffle(all_imgs)
n = len(all_imgs)
train_n = int(n * SPLIT_RATIO[0])
val_n = int(n * SPLIT_RATIO[1])
train_imgs = all_imgs[:train_n]
val_imgs = all_imgs[train_n:train_n+val_n]
test_imgs = all_imgs[train_n+val_n:]
split_map = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

# 3. 创建目标目录
for split in SPLITS:
    (OUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

# 4. 复制图片和标签
for split, imgs in split_map.items():
    for img_path in imgs:
        # 复制图片
        tgt_img = OUT_ROOT / 'images' / split / img_path.name
        shutil.copy2(img_path, tgt_img)
        # 复制标签
        label_name = img_path.with_suffix('.txt').name
        label_path = LABEL_DIR / label_name
        tgt_label = OUT_ROOT / 'labels' / split / label_name
        if label_path.exists():
            shutil.copy2(label_path, tgt_label)
        else:
            print(f'警告: 找不到标签 {label_path}，跳过')

# 5. 生成data.yaml
names = ['head', 'ordinary_clothes', 'person', 'reflective_vest', 'safety_helmet']  # 如需自动获取可改
nc = len(names)
data_yaml = {
    'train': str((OUT_ROOT / 'images' / 'train').resolve()),
    'val': str((OUT_ROOT / 'images' / 'val').resolve()),
    'test': str((OUT_ROOT / 'images' / 'test').resolve()),
    'nc': nc,
    'names': names
}
with open(OUT_ROOT / 'data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print('数据集整理完成！') 