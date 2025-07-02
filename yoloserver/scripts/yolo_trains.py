#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_trains.py
# @Function  :端到端自动化数据准备脚本

import sys
import argparse
import shutil
import logging
import yaml
import time
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import colorlog

# ==== 路径设置（根据实际项目结构调整） ====
sys.path.append(str(Path(__file__).resolve().parent.parent))  # yoloserver/
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))

from yoloserver.utils.data_converters_utils import (
    convert_data_to_yolo, YOLO_STAGED_LABELS_DIR, RAW_DATA_DIR, ORIGINAL_ANNOTATIONS_DIR
)

# ========== 日志与性能 ==========
def setup_color_logger():
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
    logger = logging.getLogger("YOLO_DataConversion")
    logger.handlers = []  # 清空旧handler
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger("YOLO_DataConversion").info(f"总耗时: {elapsed:.2f} 秒")
        return result
    return wrapper

# ========== 数据集处理核心类 ==========
class YOLODatasetProcessor:
    def __init__(self, 
                 annotation_format: str,
                 train_rate: float,
                 valid_rate: float,
                 classes: List[str] = None,
                 coco_task: str = "detection",
                 coco_cls91to80: bool = False,
                 logger=None):
        self.annotation_format = annotation_format
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.test_rate = 1.0 - train_rate - valid_rate
        if not (0 < train_rate < 1 and 0 <= valid_rate < 1 and 0 <= self.test_rate < 1 and abs(train_rate + valid_rate + self.test_rate - 1.0) < 1e-3):
            raise ValueError("训练、验证、测试集比例之和必须为1且各自为有效比例")
        self.logger = logger or logging.getLogger("YOLO_DataConversion")
        # 路径
        self.raw_images_path = RAW_DATA_DIR / "images"
        self.yolo_staged_labels_path = YOLO_STAGED_LABELS_DIR
        self.output_data_path = Path(__file__).resolve().parent.parent / "data"
        self.config_path = Path(__file__).resolve().parent.parent / "configs" / "data.yaml"
        self.output_dirs = {
            "train/images": self.output_data_path / "train/images",
            "train/labels": self.output_data_path / "train/labels",
            "val/images": self.output_data_path / "val/images",
            "val/labels": self.output_data_path / "val/labels",
            "test/images": self.output_data_path / "test/images",
            "test/labels": self.output_data_path / "test/labels",
        }
        self.classes = classes or []
        self.coco_task = coco_task
        self.coco_cls91to80 = coco_cls91to80

    def clean_and_initialize_dirs(self):
        # 清理输出数据集目录和 configs/data.yaml
        for out_dir in self.output_dirs.values():
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            self.config_path.unlink()
        # 清理暂存区
        if self.yolo_staged_labels_path.exists():
            shutil.rmtree(self.yolo_staged_labels_path)
        self.yolo_staged_labels_path.mkdir(parents=True, exist_ok=True)
        self.logger.info("已清理并初始化输出目录和配置文件。")

    def check_staged_data_existence(self):
        txts = list(self.yolo_staged_labels_path.glob("*.txt"))
        imgs = list(self.raw_images_path.glob("*"))
        if not txts:
            self.logger.critical(f"YOLO标签暂存区无txt文件: {self.yolo_staged_labels_path}")
            raise RuntimeError("YOLO标签暂存区无txt文件")
        if not imgs:
            self.logger.critical(f"原始图片目录无图片: {self.raw_images_path}")
            raise RuntimeError("原始图片目录无图片")
        self.logger.info(f"暂存区标签文件数: {len(txts)}, 原始图片数: {len(imgs)}")

    def find_matching_files(self) -> List[Tuple[Path, Path]]:
        # 匹配标签和图片
        image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        pairs = []
        for label_path in self.yolo_staged_labels_path.glob("*.txt"):
            stem = label_path.stem
            for ext in image_exts:
                img_path = self.raw_images_path / f"{stem}{ext}"
                if img_path.exists():
                    pairs.append((label_path, img_path))
                    break
        self.logger.info(f"成功配对 {len(pairs)} 对标签和图片")
        return pairs

    def process_single_split(self, pairs: List[Tuple[Path, Path]], split: str):
        img_dir = self.output_dirs[f"{split}/images"]
        label_dir = self.output_dirs[f"{split}/labels"]
        for label_path, img_path in pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, label_dir / label_path.name)
        self.logger.info(f"{split}集: 复制图片{len(pairs)}张，标签{len(pairs)}个")

    def split_and_process_data(self, pairs: List[Tuple[Path, Path]]):
        if not pairs:
            self.logger.warning("没有可用的图片-标签对，跳过数据集划分。")
            return
        # 数据集划分
        train_pairs, temp_pairs = train_test_split(pairs, test_size=(1-self.train_rate), random_state=42)
        if self.valid_rate > 0 and temp_pairs:
            val_size = self.valid_rate / (self.valid_rate + self.test_rate)
            val_pairs, test_pairs = train_test_split(temp_pairs, test_size=(1-val_size), random_state=42)
        else:
            val_pairs, test_pairs = [], temp_pairs
        # 小数据集特殊处理
        if len(pairs) < 3:
            train_pairs, val_pairs, test_pairs = pairs, [], []
        self.process_single_split(train_pairs, "train")
        self.process_single_split(val_pairs, "val")
        self.process_single_split(test_pairs, "test")

    def generate_data_yaml(self):
        data_yaml = {
            "path": "dataset",  # 固定为相对路径 dataset
            "train": str(self.output_dirs["train/images"].resolve()),
            "val": str(self.output_dirs["val/images"].resolve()),
            "test": str(self.output_dirs["test/images"].resolve()),
            "nc": len(self.classes),
            "names": self.classes
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, allow_unicode=True)
        self.logger.info(f"已生成 data.yaml: {self.config_path}")

    @time_it
    def process_dataset(self):
        # 步骤1：处理原始标注
        if self.annotation_format == "yolo":
            if not self.classes:
                self.logger.critical("原生YOLO格式必须指定类别列表 --classes")
                raise ValueError("YOLO格式必须指定类别列表")
            self.yolo_staged_labels_path = ORIGINAL_ANNOTATIONS_DIR
            if not list(self.yolo_staged_labels_path.glob("*.txt")):
                self.logger.critical("原生YOLO目录下无txt标签文件")
                raise RuntimeError("原生YOLO目录下无txt标签文件")
        elif self.annotation_format in ["coco", "pascal_voc"]:
            if self.yolo_staged_labels_path.exists():
                shutil.rmtree(self.yolo_staged_labels_path)
            self.yolo_staged_labels_path.mkdir(parents=True, exist_ok=True)
            if not self.raw_images_path.exists() or not ORIGINAL_ANNOTATIONS_DIR.exists():
                self.logger.critical("原始图片或标注目录不存在")
                raise RuntimeError("原始图片或标注目录不存在")
            # 中文注释路径logger美化输出
            self.logger.info(f"[TOP] 原标签路径: {ORIGINAL_ANNOTATIONS_DIR}")
            self.logger.info(f"[TOP] yolo格式标签路径: {YOLO_STAGED_LABELS_DIR}")
            # 修复：只有self.classes为非空列表时才传递，否则传None让底层自动扫描类别
            final_classes_order = self.classes if self.classes else None
            self.classes = convert_data_to_yolo(
                input_dir=ORIGINAL_ANNOTATIONS_DIR,
                annotation_format=self.annotation_format,
                final_classes_order=final_classes_order,
                coco_task=self.coco_task,
                coco_cls91to80=self.coco_cls91to80,
                yolo_output_dir=self.yolo_staged_labels_path
            )
            if not self.classes:
                self.logger.critical("转换失败，未获取到类别信息")
                raise RuntimeError("转换失败，未获取到类别信息")
        else:
            self.logger.critical(f"不支持的标注格式: {self.annotation_format}")
            raise ValueError(f"不支持的标注格式: {self.annotation_format}")
        # 步骤2：检查暂存区
        self.check_staged_data_existence()
        # 步骤3：配对与划分
        pairs = self.find_matching_files()
        self.split_and_process_data(pairs)
        # 步骤4：生成data.yaml
        self.generate_data_yaml()
        self.logger.info("数据集处理流程全部完成！")
        self.logger.info(f" 原标签路径: {ORIGINAL_ANNOTATIONS_DIR}")
        self.logger.info(f" yolo格式标签路径: {YOLO_STAGED_LABELS_DIR}")
        self.logger.info(f" yolo暂存区路径: {self.yolo_staged_labels_path}")
        self.logger.info(f" 原始图片路径: {self.raw_images_path}")

# ========== 命令行接口 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO数据集自动化准备脚本")
    parser.add_argument("--format", type=str, required=True, choices=["yolo", "coco", "pascal_voc"], help="原始标注格式")
    parser.add_argument("--train_rate", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--valid_rate", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--classes", type=str, nargs="*", help="类别名称列表（YOLO格式必须指定）")
    parser.add_argument("--coco_task", type=str, default="detection", help="COCO任务类型")
    parser.add_argument("--coco_cls91to80", action="store_true", help="COCO是否91类转80类")
    return parser.parse_args()

def test_pascal_voc_process():
    """
    测试 pascal_voc 格式的数据集处理流程。
    """
    test_log_path = Path(__file__).resolve().parent / "test_pascal_voc.log"
    test_logger = setup_color_logger()
    try:
        test_processor = YOLODatasetProcessor(
            annotation_format="pascal_voc",
            train_rate=0.8,
            valid_rate=0.1,
            classes=None,  # 自动类别
            logger=test_logger
        )
        test_processor.clean_and_initialize_dirs()
        test_processor.process_dataset()
        print("Pascal VOC 测试完成，请检查 data/ 目录和 configs/data.yaml。")
    except Exception as test_e:
        print(f"Pascal VOC 测试失败: {test_e}")

if __name__ == "__main__":
    main_args = parse_args()
    # 使用 colorlog 美化日志
    main_logger = setup_color_logger()
    main_logger.info("[TOP] 日志美化已启用 (colorlog)")
    main_log_path = Path(__file__).resolve().parent / "yolo_trains.log"
    try:
        main_processor = YOLODatasetProcessor(
            annotation_format=main_args.format,
            train_rate=main_args.train_rate,
            valid_rate=main_args.valid_rate,
            classes=main_args.classes,
            coco_task=main_args.coco_task,
            coco_cls91to80=main_args.coco_cls91to80,
            logger=main_logger
        )
        main_logger.info("[TOP] 开始清理和初始化数据集目录")
        main_processor.clean_and_initialize_dirs()
        main_logger.info("[TOP] 目录清理完成，准备开始数据处理流程")
        main_logger.info("[TOP] 调用 main_processor.process_dataset()，将进入中间层")
        main_logger.info("[MID] 开始执行 convert_data_to_yolo ")
        main_processor.process_dataset()
        main_logger.info("[MID] convert_data_to_yolo 执行完毕")
        main_logger.info("[TOP] main_processor.process_dataset() 执行完毕")
        main_logger.info("[TOP] 全部流程结束。请检查 data/ 目录和 configs/data.yaml。")
    except Exception as main_e:
        main_logger.critical(f"脚本执行失败: {main_e}", exc_info=True)
        sys.exit(1)

    # 测试函数调用（如需测试，取消注释）
    # test_pascal_voc_process()
