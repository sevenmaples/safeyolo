import os
import random
from pathlib import Path
import shutil
import yaml

def organize_dataset(
    img_dir=Path('raw/images'),
    label_dir=Path('raw/yolo_staged_labels'),
    out_root=Path('dataset'),
    split_ratio=[0.8, 0.1, 0.1],
    names=None,
    seed=42
):
    """
    整理图片和标签，按比例划分 train/val/test，并生成 data.yaml。
    返回 data.yaml 路径。
    """
    if names is None:
        names = ['head', 'ordinary_clothes', 'person', 'reflective_vest', 'safety_helmet']
    splits = ['train', 'val', 'test']
    random.seed(seed)
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    all_imgs = [p for p in Path(img_dir).iterdir() if p.suffix.lower() in img_exts]
    random.shuffle(all_imgs)
    n = len(all_imgs)
    train_n = int(n * split_ratio[0])
    val_n = int(n * split_ratio[1])
    train_imgs = all_imgs[:train_n]
    val_imgs = all_imgs[train_n:train_n+val_n]
    test_imgs = all_imgs[train_n+val_n:]
    split_map = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split in splits:
        (Path(out_root) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(out_root) / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for split, imgs in split_map.items():
        for img_path in imgs:
            tgt_img = Path(out_root) / 'images' / split / img_path.name
            shutil.copy2(img_path, tgt_img)
            label_name = img_path.with_suffix('.txt').name
            label_path = Path(label_dir) / label_name
            tgt_label = Path(out_root) / 'labels' / split / label_name
            if label_path.exists():
                shutil.copy2(label_path, tgt_label)
            else:
                print(f'警告: 找不到标签 {label_path}，跳过')
    nc = len(names)
    data_yaml = {
        'train': str((Path(out_root) / 'images' / 'train').resolve()),
        'val': str((Path(out_root) / 'images' / 'val').resolve()),
        'test': str((Path(out_root) / 'images' / 'test').resolve()),
        'nc': nc,
        'names': names
    }
    data_yaml_path = Path(out_root) / 'data.yaml'
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    print('数据集整理完成！')
    return data_yaml_path 