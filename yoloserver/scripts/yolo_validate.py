import argparse
from pathlib import Path
import sys
import os
import random
import shutil
import yaml
import logging
from datetime import datetime
import colorlog
import math
from collections import defaultdict
import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = kwargs.get('current_logger', None)
        if logger is None and len(args) > 0 and hasattr(args[0], 'info'):
            logger = args[0]
        if logger:
            logger.info(f"[PERF] {func.__name__} 总耗时: {elapsed:.2f} 秒")
        else:
            print(f"[PERF] {func.__name__} 总耗时: {elapsed:.2f} 秒")
        return result
    return wrapper

# =====================
# 日志初始化
# =====================
def setup_logging(base_path: Path,
                  log_type: str = "general",
                  model_name: str = None,
                  log_level: int = logging.INFO,
                  temp_log: bool = False,
                  logger_name: str = "YOLO DEFAULT",
                  encoding: str = "utf-8"):
    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "temp" if temp_log else log_type.replace(" ", "_")
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "_"))
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename
    main_logger = logging.getLogger(logger_name)
    main_logger.setLevel(log_level)
    main_logger.propagate = False
    if main_logger.handlers:
        for h in main_logger.handlers:
            main_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s"))
    main_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s : %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))
    main_logger.addHandler(console_handler)
    main_logger.info("日志记录器开始初始化".center(50, '='))
    main_logger.info(f"日志记录器已初始化，日志文件保存在: {log_file}")
    main_logger.info(f"日志记录器初始化时间：{datetime.now()}")
    main_logger.info(f"日志记录器名称: {logger_name}")
    main_logger.info(f"日志记录器最低记录级别: {logging.getLevelName(log_level)}")
    main_logger.info("日志记录器初始化完成".center(50, '='))
    return main_logger

def _calculate_std_dev(data_list):
    if len(data_list) < 2:
        return 0.0
    mean = sum(data_list) / len(data_list)
    variance = sum([(x - mean) ** 2 for x in data_list]) / (len(data_list) - 1)
    return math.sqrt(variance)

# =====================
# 数据集整理
# =====================
def organize_dataset(
    img_dir=Path('raw/images'),
    label_dir=Path('raw/yolo_staged_labels'),
    out_root=Path('dataset'),
    split_ratio=None,
    names=None,
    seed=42
):
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
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

# =====================
# 数据集验证
# =====================
@time_it
def verify_dataset_config(
    yaml_path: Path,
    current_logger: logging.Logger,
    mode: str = "FULL",
    task_type: str = "detection",
    sample_ratio: float = 0.1,
    min_samples: int = 10
):
    current_logger.info("[VALIDATE] 开始验证数据集配置...")
    invalid_list = []
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        current_logger.error(f"[VALIDATE] data.yaml 文件未找到: {e}")
        return False, [{"error_message": f"data.yaml 文件未找到: {e}"}]
    except yaml.YAMLError as e:
        current_logger.error(f"[VALIDATE] data.yaml 解析错误: {e}")
        return False, [{"error_message": f"data.yaml 解析错误: {e}"}]
    except Exception as e:
        current_logger.error(f"[VALIDATE] 读取data.yaml时发生未知错误: {e}")
        return False, [{"error_message": f"data.yaml读取失败: {e}"}]
    nc = data_cfg.get('nc')
    names = data_cfg.get('names')
    if not isinstance(names, list) or nc != len(names):
        current_logger.error(f"[VALIDATE] nc与names数量不一致: nc={nc}, names={names}")
        invalid_list.append({"error_message": "nc与names数量不一致"})
        return False, invalid_list
    splits = ['train', 'val', 'test']
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    split_analysis_data = {}
    for split in splits:
        img_dir = Path(data_cfg.get(split, ''))
        label_dir = img_dir.parent.parent / 'labels' / split
        if not img_dir.exists() or not any(img_dir.glob('*')):
            current_logger.error(f"[VALIDATE] {split}目录不存在或为空: {img_dir}")
            invalid_list.append({"error_message": f"{split}目录不存在或为空: {img_dir}"})
            continue
        img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in img_exts]
        if not img_files:
            current_logger.warning(f"[VALIDATE] {split}目录下无图片: {img_dir}")
            continue
        # SAMPLE模式下采样
        if mode.upper() == 'SAMPLE' and len(img_files) > min_samples:
            sample_size_actual = max(min_samples, int(len(img_files) * sample_ratio))
            img_files = random.sample(img_files, min(sample_size_actual, len(img_files)))
            current_logger.info(f"{split} 划分验证模式为 SAMPLE，随机抽样 {len(img_files)} 张图像进行详细验证和分析。")
        else:
            current_logger.info(f"{split} 划分验证模式为 FULL，正在验证和分析所有 {len(img_files)} 张图像。")
        # 统计分析数据结构
        split_analysis_data[split] = {
            "total_images_analyzed": len(img_files),
            "total_instances": 0,
            "class_counts": defaultdict(int),
            "images_per_class": defaultdict(int),
            "bbox_areas": defaultdict(list),
            "bbox_aspect_ratios": defaultdict(list),
        }
        for img_path in img_files:
            label_path = label_dir / img_path.with_suffix('.txt').name
            if not label_path.exists():
                current_logger.error(f"[VALIDATE] 缺少标签文件: {label_path}")
                invalid_list.append({
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "error_message": "缺少标签文件"
                })
                continue
            try:
                with open(label_path, 'r', encoding='utf-8') as lf:
                    lines = lf.readlines()
            except Exception:
                current_logger.error(f"[VALIDATE] 标签文件读取失败: {label_path}")
                invalid_list.append({
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "error_message": f"标签文件读取失败"
                })
                continue
            classes_in_this_image = set()
            for line in lines:
                parts = line.strip().split()
                if not parts or not parts[0].isdigit():
                    continue
                class_id = int(parts[0])
                if 0 <= class_id < nc:
                    split_analysis_data[split]["class_counts"][class_id] += 1
                    split_analysis_data[split]["total_instances"] += 1
                    classes_in_this_image.add(class_id)
                    # detection/segmentation统计
                    if task_type == 'detection' and len(parts) == 5:
                        bbox_w_norm = float(parts[3])
                        bbox_h_norm = float(parts[4])
                        if bbox_w_norm > 0 and bbox_h_norm > 0:
                            area_norm = bbox_w_norm * bbox_h_norm
                            split_analysis_data[split]["bbox_areas"][class_id].append(area_norm)
                            split_analysis_data[split]["bbox_aspect_ratios"][class_id].append(bbox_w_norm / bbox_h_norm)
                    elif task_type == 'segmentation' and len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        points = [float(p) for p in parts[1:]]
                        xs = [points[i] for i in range(0, len(points), 2)]
                        ys = [points[i] for i in range(1, len(points), 2)]
                        if xs and ys:
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            bbox_w_norm = max_x - min_x
                            bbox_h_norm = max_y - min_y
                            if bbox_w_norm > 0 and bbox_h_norm > 0:
                                area_norm = bbox_w_norm * bbox_h_norm
                                split_analysis_data[split]["bbox_areas"][class_id].append(area_norm)
                                split_analysis_data[split]["bbox_aspect_ratios"][class_id].append(bbox_w_norm / bbox_h_norm)
            for class_id in classes_in_this_image:
                split_analysis_data[split]["images_per_class"][class_id] += 1
    passed = len(invalid_list) == 0
    if passed:
        current_logger.info("[VALIDATE] 数据集基础验证通过！")
    else:
        current_logger.error(f"[VALIDATE] 数据集基础验证未通过，共发现{len(invalid_list)}个问题样本。")
    return passed, invalid_list, split_analysis_data

def verify_split_uniqueness(
    yaml_path: Path,
    current_logger: logging.Logger
) -> bool:
    current_logger.info("[VALIDATE] 检查数据集分割唯一性...")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        current_logger.error(f"[VALIDATE] data.yaml 文件未找到: {e}")
        return False
    except yaml.YAMLError as e:
        current_logger.error(f"[VALIDATE] data.yaml 解析错误: {e}")
        return False
    except Exception as e:
        current_logger.error(f"[VALIDATE] 读取data.yaml时发生未知错误: {e}")
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
    current_logger.warning("[VALIDATE] 开始删除不合法的图片和标签文件...")
    deleted_image_count = 0
    deleted_label_count = 0
    for item in invalid_data_list:
        img_path = item.get('image_path')
        label_path = item.get('label_path')
        for p in [img_path, label_path]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                    if p == img_path:
                        deleted_image_count += 1
                    else:
                        deleted_label_count += 1
                    current_logger.warning(f"[VALIDATE] 已删除: {p}")
                except Exception as e:
                    current_logger.error(f"[VALIDATE] 删除失败: {p}, 错误: {e}")
    current_logger.info(f"SUMMARY: 删除操作完成。共删除了 {deleted_image_count} 个图像文件和 {deleted_label_count} 个标签文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO数据集验证工具", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml', type=str, default=None, help='data.yaml文件路径')
    parser.add_argument('--mode', type=str, default='FULL', choices=['FULL', 'SAMPLE'], help='验证模式：FULL(全量)/SAMPLE(抽样)')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'segmentation'], help='任务类型')
    parser.add_argument('--delete-invalid', action='store_true', help='是否自动删除不合法文件')
    parser.add_argument('--prepare', action='store_true', help='是否先整理原始数据集并生成 data.yaml')
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='SAMPLE模式下采样比例')
    parser.add_argument('--min-samples', type=int, default=10, help='SAMPLE模式下每split最小采样数')
    args = parser.parse_args()

    # 日志初始化（控制台彩色+文件）
    logger = setup_logging(base_path=Path('yoloserver/logs'), log_type='yolo_validate', model_name=None, temp_log=False)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel('INFO')

    # 新增：如指定 --prepare，先整理数据集
    if args.prepare:
        logger.info("[TOP] 开始整理原始数据集...")
        data_yaml_path = organize_dataset(
            img_dir=Path('raw/images'),
            label_dir=Path('raw/yolo_staged_labels'),
            out_root=Path('dataset'),
            split_ratio=None,
            names=['head', 'ordinary_clothes', 'person', 'reflective_vest', 'safety_helmet']
        )
        args.yaml = str(data_yaml_path)
        logger.info(f"[TOP] 数据集整理完成，生成配置文件: {args.yaml}")

    # 自动查找 data.yaml（升级：查找 safeyolo 根目录下的 dataset/data.yaml）
    if args.yaml is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [root_dir / 'dataset/data.yaml', root_dir / 'yoloserver/configs/data.yaml']
        found = False
        for cand in candidates:
            if cand.exists():
                args.yaml = str(cand)
                logger.info(f"[TOP] 未指定 --yaml，自动使用: {args.yaml}")
                found = True
                break
        if not found:
            logger.error("[TOP] 未找到 data.yaml，请用 --yaml 指定配置文件路径，或先运行数据集整理脚本！")
            sys.exit(1)

    yaml_path = Path(args.yaml)
    logger.info(f"[TOP] 开始数据集验证，配置文件: {yaml_path}")

    # 新增：打印 data.yaml 配置内容、类别名称和类别数量
    with open(yaml_path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    logger.info(f"配置内容: {configs}")
    logger.info(f"类别名称: {configs.get('names')}")
    logger.info(f"类别数量: {configs.get('nc')}")

    # 1. 基础验证
    passed, invalid_list, split_analysis_data = verify_dataset_config(
        yaml_path, logger, args.mode, args.task, args.sample_ratio, args.min_samples)
    if not passed:
        logger.error(f"[TOP] 数据集基础验证未通过，共发现{len(invalid_list)}个问题样本。")
        if args.delete_invalid and invalid_list:
            logger.warning("[TOP] 启动自动删除不合法文件...")
            delete_invalid_files(invalid_list, logger)
            logger.warning("[TOP] 不合法文件已删除，请重新检查数据集！")
        else:
            logger.warning("[TOP] 检测到不合法文件，未自动删除。可加 --delete-invalid 参数自动删除。")
    else:
        logger.info("[TOP] 数据集基础验证通过！")

    # 2. 分割唯一性验证
    unique = verify_split_uniqueness(yaml_path, logger)
    if not unique:
        logger.error("[TOP] 数据集分割唯一性验证未通过，请检查train/val/test是否有重复图片！")
    else:
        logger.info("[TOP] 数据集分割唯一性验证通过！")

    # 3. 总结
    if passed and unique:
        logger.info("[TOP] 数据集验证全部通过，可以安全开始训练！")
        logger.info("=" * 100)
        logger.info("数据集类别分布分析结果".center(100))
        logger.info("=" * 100)
        names = configs.get('names', [])
        nc = configs.get('nc', 0)
        for split_name in ['train', 'val', 'test']:
            data = split_analysis_data.get(split_name, None)
            if data is None:
                continue
            logger.info(f"\n--- {split_name.upper()} 划分分析 ---")
            if data["total_images_analyzed"] == 0:
                logger.info(f"  此 {split_name.upper()} 划分中没有可分析的图像文件。")
                continue
            logger.info(f"  分析图像总数: {data['total_images_analyzed']}")
            logger.info(f"  总实例数: {data['total_instances']}")
            if data["total_instances"] == 0:
                logger.info(f"  此 {split_name.upper()} 划分中所有分析的图像均无标注实例。")
                logger.info("-" * 130)
                logger.info(f"{'Class ID':<8} {'Class Name':<20} {'Total Instances':>15} {'Instance %':>12} {'Image Count':>12} {'Image %':>12} {'Avg Inst/Img':>12} {'Avg Bbox Area':>15} {'StdDev Area':>12} {'Avg Aspect Ratio':>15} {'StdDev Aspect':>12}")
                logger.info("-" * 130)
                continue
            max_class_name_len = max(len(str(name)) for name in names) if names else len('Class Name')
            class_name_col_width = max(max_class_name_len, len('Class Name'))
            header = (
                f"{'Class ID':<8} "
                f"{'Class Name':<{class_name_col_width}} "
                f"{'Total Instances':>15} "
                f"{'Instance %':>12} "
                f"{'Image Count':>12} "
                f"{'Image %':>12} "
                f"{'Avg Inst/Img':>12} "
                f"{'Avg Bbox Area':>15} "
                f"{'StdDev Area':>12} "
                f"{'Avg Aspect Ratio':>15} "
                f"{'StdDev Aspect':>12} "
            )
            separator = "-" * len(header)
            logger.info(separator)
            logger.info(header)
            logger.info(separator)
            for class_id in range(nc):
                class_name = names[class_id] if 0 <= class_id < len(names) else '未知类别'
                instance_count = data["class_counts"][class_id]
                instance_percentage = (instance_count / data["total_instances"]) * 100 if data["total_instances"] > 0 else 0.0
                image_coverage_count = data["images_per_class"][class_id]
                image_coverage_percentage = (image_coverage_count / data["total_images_analyzed"]) * 100 if data["total_images_analyzed"] > 0 else 0.0
                avg_instances_per_image = instance_count / image_coverage_count if image_coverage_count > 0 else 0.0
                bbox_areas_list = data["bbox_areas"][class_id]
                avg_bbox_area = sum(bbox_areas_list) / len(bbox_areas_list) if bbox_areas_list else 0.0
                std_dev_area = _calculate_std_dev(bbox_areas_list)
                bbox_aspect_ratios_list = data["bbox_aspect_ratios"][class_id]
                avg_bbox_aspect_ratio = sum(bbox_aspect_ratios_list) / len(bbox_aspect_ratios_list) if bbox_aspect_ratios_list else 0.0
                std_dev_aspect = _calculate_std_dev(bbox_aspect_ratios_list)
                row = (
                    f"{str(class_id):<8} "
                    f"{class_name:<{class_name_col_width}} "
                    f"{instance_count:>15} "
                    f"{instance_percentage:>12.2f} "
                    f"{image_coverage_count:>12} "
                    f"{image_coverage_percentage:>12.2f} "
                    f"{avg_instances_per_image:>12.2f} "
                    f"{avg_bbox_area:>15.4f} "
                    f"{std_dev_area:>12.4f} "
                    f"{avg_bbox_aspect_ratio:>15.2f} "
                    f"{std_dev_aspect:>12.2f} "
                )
                logger.info(row)
            logger.info(separator)
        logger.info("=" * 100)
        logger.info("数据集分析完成！".center(100))
        logger.info("=" * 100)
        sys.exit(0)
    else:
        logger.error("[TOP] 数据集验证未通过，请根据日志修复问题后重试！")
        sys.exit(1)

print("当前工作目录:", os.getcwd()) 