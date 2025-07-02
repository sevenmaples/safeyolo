from pathlib import Path
from typing import Tuple, List, Dict
import logging
import yaml
import random

# =====================
# YOLO 数据集验证逻辑模块
# =====================

def verify_dataset_config(
    yaml_path: Path,
    current_logger: logging.Logger,
    mode: str = "FULL",
    task_type: str = "detection"
) -> Tuple[bool, List[Dict]]:
    """
    验证 data.yaml 及数据集结构、标签内容、类别ID、坐标范围等。
    :param yaml_path: data.yaml 文件路径
    :param current_logger: 日志实例
    :param mode: 验证模式 FULL/SAMPLE
    :param task_type: detection/segmentation
    :return: (是否通过, 不合法样本列表)
    """
    current_logger.info("[VALIDATE] 开始验证数据集配置...")
    invalid_list = []
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as e:
        current_logger.error(f"[VALIDATE] 读取data.yaml失败: {e}")
        return False, [{"error_message": f"data.yaml读取失败: {e}"}]

    # 检查nc和names一致性
    nc = data_cfg.get('nc')
    names = data_cfg.get('names')
    if not isinstance(names, list) or nc != len(names):
        current_logger.error(f"[VALIDATE] nc与names数量不一致: nc={nc}, names={names}")
        invalid_list.append({"error_message": "nc与names数量不一致"})
        return False, invalid_list

    splits = ['train', 'val', 'test']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for split in splits:
        img_dir = Path(data_cfg.get(split, ''))
        if not img_dir.exists() or not any(img_dir.glob('*')):
            current_logger.error(f"[VALIDATE] {split}目录不存在或为空: {img_dir}")
            invalid_list.append({"error_message": f"{split}目录不存在或为空: {img_dir}"})
            continue
        img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
        if not img_files:
            current_logger.warning(f"[VALIDATE] {split}目录下无图片: {img_dir}")
            continue
        # SAMPLE模式下只抽查部分图片
        if mode.upper() == 'SAMPLE' and len(img_files) > 20:
            img_files = random.sample(img_files, 20)
        for img_path in img_files:
            # 修正：查找标签于labels/{split}/xxx.txt
            label_dir = img_dir.parent.parent / 'labels' / split
            label_path = label_dir / img_path.name.replace(img_path.suffix, '.txt')
            if not label_path.exists():
                current_logger.error(f"[VALIDATE] 缺少标签文件: {label_path}")
                invalid_list.append({
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "error_message": "缺少标签文件"
                })
                continue
            # 检查标签内容
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                current_logger.error(f"[VALIDATE] 标签文件读取失败: {label_path}, {e}")
                invalid_list.append({
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "error_message": f"标签文件读取失败: {e}"
                })
                continue
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 检查格式
                if task_type == 'detection':
                    if len(parts) != 5:
                        current_logger.error(f"[VALIDATE] 检测任务标签格式错误: {label_path}, 行{idx+1}")
                        invalid_list.append({
                            "image_path": str(img_path),
                            "label_path": str(label_path),
                            "error_message": f"检测任务标签格式错误: 行{idx+1}"
                        })
                        continue
                elif task_type == 'segmentation':
                    if len(parts) < 7 or (len(parts)-1)%2 != 0:
                        current_logger.error(f"[VALIDATE] 分割任务标签格式错误: {label_path}, 行{idx+1}")
                        invalid_list.append({
                            "image_path": str(img_path),
                            "label_path": str(label_path),
                            "error_message": f"分割任务标签格式错误: 行{idx+1}"
                        })
                        continue
                # 检查数值
                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except Exception as e:
                    current_logger.error(f"[VALIDATE] 标签内容非数字: {label_path}, 行{idx+1}")
                    invalid_list.append({
                        "image_path": str(img_path),
                        "label_path": str(label_path),
                        "error_message": f"标签内容非数字: 行{idx+1}"
                    })
                    continue
                # 检查类别ID
                if not (0 <= class_id < nc):
                    current_logger.error(f"[VALIDATE] 类别ID越界: {label_path}, 行{idx+1}, id={class_id}")
                    invalid_list.append({
                        "image_path": str(img_path),
                        "label_path": str(label_path),
                        "error_message": f"类别ID越界: 行{idx+1}, id={class_id}"
                    })
                # 检查坐标范围
                for v in coords:
                    if not (0.0 <= v <= 1.0):
                        current_logger.error(f"[VALIDATE] 坐标超出范围: {label_path}, 行{idx+1}, v={v}")
                        invalid_list.append({
                            "image_path": str(img_path),
                            "label_path": str(label_path),
                            "error_message": f"坐标超出范围: 行{idx+1}, v={v}"
                        })
            # 检查是否有多余的标签文件
        # 检查标签文件是否有对应图片
        label_files = list(img_dir.glob('*.txt'))
        for label_path in label_files:
            img_path = label_path.with_suffix('.jpg')
            if not img_path.exists():
                img_path = label_path.with_suffix('.png')
            if not img_path.exists():
                img_path = label_path.with_suffix('.jpeg')
            if not img_path.exists():
                img_path = label_path.with_suffix('.bmp')
            if not img_path.exists():
                img_path = label_path.with_suffix('.tif')
            if not img_path.exists():
                img_path = label_path.with_suffix('.tiff')
            if not img_path.exists():
                current_logger.error(f"[VALIDATE] 标签文件无对应图片: {label_path}")
                invalid_list.append({
                    "image_path": None,
                    "label_path": str(label_path),
                    "error_message": "标签文件无对应图片"
                })
    passed = len(invalid_list) == 0
    if passed:
        current_logger.info("[VALIDATE] 数据集基础验证通过！")
    else:
        current_logger.error(f"[VALIDATE] 数据集基础验证未通过，共发现{len(invalid_list)}个问题样本。")
    return passed, invalid_list

def verify_split_uniqueness(
    yaml_path: Path,
    current_logger: logging.Logger
) -> bool:
    """
    检查train/val/test三者图片文件名无交集
    :param yaml_path: data.yaml 文件路径
    :param current_logger: 日志实例
    :return: True为无重复，False为有重复
    """
    current_logger.info("[VALIDATE] 检查数据集分割唯一性...")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except Exception as e:
        current_logger.error(f"[VALIDATE] 读取data.yaml失败: {e}")
        return False
    splits = ['train', 'val', 'test']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    split_files = {}
    for split in splits:
        img_dir = Path(data_cfg.get(split, ''))
        if not img_dir.exists():
            continue
        img_files = [p.name for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
        split_files[split] = set(img_files)
    overlap = False
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            if split_files.get(s1) and split_files.get(s2):
                common = split_files[s1] & split_files[s2]
                if common:
                    current_logger.error(f"[VALIDATE] {s1}和{s2}集存在重复图片: {common}")
                    overlap = True
    if not overlap:
        current_logger.info("[VALIDATE] 数据集分割唯一性验证通过！")
    else:
        current_logger.error("[VALIDATE] 数据集分割唯一性验证未通过！")
    return not overlap

def delete_invalid_files(
    invalid_data_list: list,
    current_logger: logging.Logger
):
    """
    删除不合法的图片和标签文件
    :param invalid_data_list: verify_dataset_config返回的不合法样本列表
    :param current_logger: 日志实例
    """
    current_logger.warning("[VALIDATE] 开始删除不合法的图片和标签文件...")
    for item in invalid_data_list:
        img_path = item.get('image_path')
        label_path = item.get('label_path')
        for p in [img_path, label_path]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                    current_logger.warning(f"[VALIDATE] 已删除: {p}")
                except Exception as e:
                    current_logger.error(f"[VALIDATE] 删除失败: {p}, 错误: {e}") 