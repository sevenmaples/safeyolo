# @Function    :统一数据转换器,根据原始标注,根据原始标注格式,决定使用那个数据转换器
import logging
from pathlib import Path
from typing import List, Union
import os

# 路径管理
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # safeyolo
RAW_DATA_DIR = PROJECT_ROOT / "raw"
YOLO_STAGED_LABELS_DIR = RAW_DATA_DIR / "yolo_staged_labels"
ORIGINAL_ANNOTATIONS_DIR = RAW_DATA_DIR / "original_annotations"

from yoloserver.utils.data_converters.coco import coco2yolo_convert
from yoloserver.utils.data_converters.pascal_voc import voc2yolo_convert

logger = logging.getLogger(__name__)


def convert_data_to_yolo(
    input_dir: Union[str, Path],
    annotation_format: str = "pascal_voc",
    final_classes_order: Union[List[str], None] = None,
    coco_task: str = "detection",
    coco_cls91to80: bool = False,
    yolo_output_dir: Union[str, Path] = None,
) -> List[str]:
    """
    统一的标注转换函数，根据原始标注格式，自动调用对应的底层转换器。
    :param input_dir: 原始标注文件夹路径（str 或 Path）
    :param annotation_format: 标注格式（'coco' 或 'pascal_voc'）
    :param final_classes_order: 指定类别顺序（可选）
    :param coco_task: COCO专用参数，任务类型
    :param coco_cls91to80: COCO专用参数，是否91转80类
    :param yolo_output_dir: YOLO标签输出目录（可选，默认自动管理）
    :return: 返回转换得到的类别列表
    """
    input_dir = Path(input_dir)
    if yolo_output_dir is None:
        yolo_output_dir = YOLO_STAGED_LABELS_DIR
    else:
        yolo_output_dir = Path(yolo_output_dir)

    logger.info(f"开始标注转换: 格式={annotation_format}, 输入={input_dir}")
    if not input_dir.exists():
        logger.error(f"原始标注数据文件夹不存在: {input_dir}")
        raise FileNotFoundError(f"原始标注数据文件夹不存在: {input_dir}")
    classes: List[str] = []
    try:
        if annotation_format == "coco":
            # 参数协调：COCO专用参数
            classes = coco2yolo_convert(
                str(input_dir),
                str(yolo_output_dir),
                final_classes_order,  # 只要底层支持就传递
                logger
            )
        elif annotation_format == "pascal_voc":
            # 参数协调：Pascal VOC专用参数
            classes = voc2yolo_convert(
                str(input_dir),
                str(yolo_output_dir),
                final_classes_order,
                logger
            )
            if not classes:
                logger.error("Pascal Voc转换失败, 未提取到任何有效类别标签")
                return []
            logger.info(f"Pascal Voc转换成功, 类别: {classes}")
        else:
            logger.error(f"不支持的标注格式: {annotation_format}")
            raise ValueError(f"不支持的标注格式: {annotation_format}")
    except Exception as e:
        logger.critical(f"转换致命错误: {annotation_format}, 错误: {e}", exc_info=True)
        return []
    if not classes:
        logger.error("转换未提取到任何类别, 请检查输入数据和配置")
    else:
        logger.info(f"转换完成, 格式: {annotation_format}, 类别: {classes}")
    return classes
