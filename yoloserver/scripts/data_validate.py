import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import argparse
import yaml
from pathlib import Path
import logging
import random
from typing import Dict, List, Tuple, Union
import colorlog

# 定义常量
YOLO_SPLITS = ["train", "val", "test"]
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"]
DATA_YAML_ERROR_PREFIX = "FATAL ERROR: data.yaml"

# 彩色日志初始化
logger = logging.getLogger("data_validate")
logger.setLevel(logging.INFO)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
logger.handlers = []
logger.addHandler(handler)

# 从 performance_utils 导入（若实际有该模块需确保可导入）
# from performance_utils import time_it   # 若有此装饰器可取消注释使用

# ------------------------------ 1. 辅助函数：加载 YAML 文件 ------------------------------
def _load_yaml_file(yaml_path: Path) -> Dict:
    """
    加载并解析 yaml 文件
    :param yaml_path: yaml 文件的路径
    :return: yaml 的配置内容
    """
    if not yaml_path.exists():
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 文件不存在: {yaml_path}, 请检查配置文件路径是否正确")
        raise FileNotFoundError(f"data.yaml 文件不存在: {yaml_path}")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise yaml.YAMLError(f"data.yaml 文件内容不是一个有效的字典结构")
        return config
    except yaml.YAMLError as e:
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 文件解析错误: {yaml_path}, 请检查配置文件格式是否正确")
        raise yaml.YAMLError(f"data.yaml 文件解析错误: {yaml_path}, 请检查配置文件格式是否正确")
    except Exception as e:
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 读取时发生未知错误: {e}")
        raise e

# ------------------------------ 2. 验证 data.yaml 内容（类别名称和数量） ------------------------------
def _validate_yaml_config_content(config: Dict) -> Tuple[List[str], int]:
    """
    确保 names 长度和 nc 数量一致
    :param config: data.yaml 的内容
    :return: 类别名称列表、类别数量
    """
    classes_names = config.get("names", [])
    nc = config.get("nc", 0)

    # 校验 names 字段
    if not classes_names or not isinstance(classes_names, list) or not all(isinstance(name, str) for name in classes_names):
        error_msg = f"配置文件中的缺少 names 字段或者 names 字段必须是一个列表，且列表中的元素必须是字符串"
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} {error_msg}")
        raise ValueError(f"{DATA_YAML_ERROR_PREFIX} {error_msg}")

    # 校验 nc 字段
    if not isinstance(nc, int) or nc <= 0:
        error_msg = f"配置文件中, 缺少 nc 字段, 或者 nc 字段必须是一个正整数"
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} {error_msg}")
        raise ValueError(f"{DATA_YAML_ERROR_PREFIX} {error_msg}")

    # 校验 nc 与 names 长度是否一致
    if nc != len(classes_names):
        error_msg = f"配置文件中, nc 字段的值与 names 字段的长度不一致"
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} {error_msg}")
        raise ValueError(f"{DATA_YAML_ERROR_PREFIX} {error_msg}")

    logger.info(f"数据集类别数量与配置文件一致，类别数量为: {nc} 类别名称为: {classes_names}")
    return classes_names, nc

# ------------------------------ 3. 获取指定图像目录下所有图像的文件路径 ------------------------------
def _get_image_paths_in_directory(directory_path: Path) -> Union[List[Path], None]:
    """
    获取指定目录下的所有图像文件路径
    :param directory_path: 图像目录
    :return: 所有图像文件路径列表，不存在或无图像时返回 None
    """
    if not directory_path.exists():
        logger.critical(f"图像目录: {directory_path} 不存在")
        return None
    all_imgs = []
    for ext in IMG_EXTENSIONS:
        all_imgs.extend(list(directory_path.glob(ext)))
    if not all_imgs:
        logger.critical(f"图像目录: {directory_path} 中没有找到任何图像文件")
        return None
    else:
        logger.info(f"图像目录: {directory_path} 中找到 {len(all_imgs)} 个图像文件")
        return all_imgs

# ------------------------------ 4. 安全读取所有 YOLO 标注文件的所有行 ------------------------------
def _read_label_file_lines(label_path: Path) -> Tuple[List[str], str]:
    """
    读取 yolo txt 所有的文本行
    :param label_path: 标签文件的路径
    :return: 文本行列表、错误信息（无错误时为空字符串）
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return lines, ""
    except Exception as e:
        error_msg = f"读取标签文件失败: {label_path}, 错误信息为: {e}"
        logger.critical(error_msg)
        return [], error_msg

# ------------------------------ 5. 验证单个标签文件内容格式和合法性 ------------------------------
def _validate_single_label_content(
    lines: List[str],
    label_path: Path,
    nc: int,
    task_type: str = "detection"
) -> Tuple[bool, str]:
    """
    验证单个标签文件的内容格式和合法性
    :param lines: 标签文件的所有行
    :param label_path: 标签文件的路径
    :param nc: 类别数量
    :param task_type: 任务类型，可选值有: detection, segmentation
    :return: 验证结果（True 表示通过，False 表示失败）、错误详情
    """
    if not lines:
        return True, ""  # 允许标签文件为空，表示图像没有标注

    for line_idx, line in enumerate(lines):
        parts = line.split(" ")
        is_format_correct = True
        error_detail = ""

        # 1. 检查 YOLO 标记文件的格式字段数量
        if task_type == "detection":
            if len(parts) != 5:
                error_detail = f"不符合 YOLO 目标检测格式 (期望是5个字段: 类别ID, 归一化的坐标)"
                is_format_correct = False
        elif task_type == "segmentation":
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                error_detail = f"不符合 YOLO 目标分割格式 (期望是7个字段: 类别ID, 归一化的坐标...)"
                is_format_correct = False
        else:
            error_detail = f"不支持的任务类型 {task_type} "
            is_format_correct = False

        if not is_format_correct:
            return (
                False,
                f"标签文件 {label_path} 第 {line_idx + 1} 行: '{line}' - {error_detail}"
            )

        # 2. 检查数值类型和范围
        try:
            class_id = int(parts[0])
            if not (0 <= class_id < nc):
                return (
                    False,
                    f"标签文件 {label_path} 第 {line_idx + 1} 行: '{line}' - 类别ID {class_id} 不在有效范围"
                )

            coords = [float(x) for x in parts[1:]]
            if not all(0 <= x <= 1 for x in coords):
                return (
                    False,
                    f"标签文件 {label_path} 第 {line_idx + 1} 行: '{line}' - 坐标值 {coords} 不在有效范围"
                )
        except ValueError:
            return (
                False,
                f"标签文件 {label_path} 第 {line_idx + 1} 行: '{line}' - 数值类型错误"
            )

    return True, ""  # 所有的标签文件都验证通过

# ------------------------------ 主程序入口（可根据需要扩展） ------------------------------
if __name__ == "__main__":
    # argparse 支持 data.yaml 路径和图片目录路径
    parser = argparse.ArgumentParser(description="YOLO数据集标签与配置检测工具")
    parser.add_argument('--yaml', type=str, default='dataset/data.yaml', help='data.yaml 配置文件路径')
    parser.add_argument('--image_dir', type=str, default='dataset/images/train', help='图片目录路径')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'segmentation'], help='任务类型')
    args = parser.parse_args()

    yaml_file_path = Path(args.yaml)
    image_dir = Path(args.image_dir)
    task_type = args.task

    # 1. 加载 data.yaml 文件
    configs = _load_yaml_file(yaml_file_path)

    # 2. 验证 data.yaml 内容
    class_names, class_nc = _validate_yaml_config_content(configs)

    # 3. 获取图像路径
    image_paths = _get_image_paths_in_directory(image_dir)

    # 4. 读取并验证标签文件
    if image_paths:
        for img_path in image_paths:
            label_path = img_path.with_suffix('.txt')
            label_lines, err = _read_label_file_lines(label_path)
            if err:
                logger.error(err)
                continue
            valid, detail = _validate_single_label_content(label_lines, label_path, class_nc, task_type=task_type)
            if not valid:
                logger.error(detail)

    # 打印调试信息
    print("配置内容:", configs)
    print("类别名称:", class_names)
    print("类别数量:", class_nc)