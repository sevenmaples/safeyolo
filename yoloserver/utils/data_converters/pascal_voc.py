import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Set, Any

# 常量
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
XML_DIR = PROJECT_ROOT / "raw" / "original_annotations"
YOLO_DIR = PROJECT_ROOT / "raw" / "yolo_staged_labels"
YOLO_EXT = ".txt"
LOGGER_NAME = "voc2yolo"


def parse_voc_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """
    解析单个 Pascal VOC XML 文件，返回图片尺寸和所有目标信息。
    参数：
        xml_path (Path): XML 文件路径。
    返回：
        Optional[Dict[str, Any]]: 包含宽、高和目标列表的字典，若无效则为 None。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            logging.warning(f"XML 缺少 <size> 标签: {xml_path}")
            return None
        width = int(size.findtext("width", default="0"))
        height = int(size.findtext("height", default="0"))
        if width <= 0 or height <= 0:
            logging.warning(f"XML 图片尺寸无效: {xml_path}")
            return None
        objects = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bndbox = obj.find("bndbox")
            if name is None or bndbox is None:
                continue
            try:
                xmin = float(bndbox.findtext("xmin", default="0"))
                ymin = float(bndbox.findtext("ymin", default="0"))
                xmax = float(bndbox.findtext("xmax", default="0"))
                ymax = float(bndbox.findtext("ymax", default="0"))
            except (TypeError, ValueError):
                logging.warning(f"边界框坐标解析失败: {xml_path}")
                continue
            if xmin >= xmax or ymin >= ymax:
                logging.warning(f"无效边界框: {xml_path} [{xmin}, {ymin}, {xmax}, {ymax}]")
                continue
            objects.append({
                "name": name,
                "bbox": [xmin, ymin, xmax, ymax]
            })
        return {"width": width, "height": height, "objects": objects}
    except (ET.ParseError, FileNotFoundError) as e:
        logging.error(f"解析 XML 失败: {xml_path} | 错误: {e}")
        return None


def voc_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """
    VOC 边界框 [xmin, ymin, xmax, ymax] 转为 YOLO 格式 [x_center, y_center, w, h]（归一化）。
    参数：
        bbox (List[float]): [xmin, ymin, xmax, ymax]
        img_w (int): 图片宽度
        img_h (int): 图片高度
    返回：
        List[float]: [x_center, y_center, w, h]（归一化）
    """
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return [x_center, y_center, w, h]


def convert_voc_dir_to_yolo(
    xml_dir: Path,
    yolo_dir: Path,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    将 xml_dir 下所有 Pascal VOC XML 文件转为 YOLO txt 格式，写入 yolo_dir，并返回类别名称列表。
    参数：
        xml_dir (Path): Pascal VOC XML 文件目录。
        yolo_dir (Path): YOLO 标签输出目录。
        class_names (Optional[List[str]]): 指定类别顺序（可选），如不指定则自动收集。
        logger (Optional[logging.Logger]): 日志记录器（可选）。
    返回：
        List[str]: 最终类别名称列表。
    """
    logger = logger or logging.getLogger(LOGGER_NAME)
    if not xml_dir.exists():
        logger.error(f"输入目录不存在: {xml_dir}")
        return []
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # 第一次遍历收集所有类别
    all_classes: Set[str] = set(class_names) if class_names else set()
    xml_files = list(xml_dir.glob("*.xml"))
    for xml_path in xml_files:
        info = parse_voc_xml(xml_path)
        if info is None:
            continue
        for obj in info["objects"]:
            all_classes.add(obj["name"])
    # 确定类别顺序
    if class_names is None:
        class_names = sorted(all_classes)
    class2id = {name: idx for idx, name in enumerate(class_names)}

    # 第二次遍历，转换并写入 YOLO 标签
    for xml_path in xml_files:
        info = parse_voc_xml(xml_path)
        if info is None:
            continue
        yolo_lines = []
        for obj in info["objects"]:
            if obj["name"] not in class2id:
                logger.warning(f"类别未在类别列表中: {obj['name']} | 文件: {xml_path}")
                continue
            class_id = class2id[obj["name"]]
            yolo_box = voc_bbox_to_yolo(obj["bbox"], info["width"], info["height"])
            yolo_line = f"{class_id} {' '.join(f'{x:.6f}' for x in yolo_box)}"
            yolo_lines.append(yolo_line)
        # 写入 txt
        txt_name = xml_path.stem + YOLO_EXT
        txt_path = yolo_dir / txt_name
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            logger.info(f"已生成: {txt_path}（{len(yolo_lines)} 个目标）")
        except OSError as e:
            logger.error(f"写入 YOLO 标签失败: {txt_path} | 错误: {e}")

    logger.info(f"VOC->YOLO 转换完成，类别列表: {class_names}")
    return class_names


def voc2yolo_convert(
    xml_dir: str,
    yolo_dir: str,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    高层封装接口，支持字符串路径和类别，自动转换为 Path 并调用核心转换函数。
    参数：
        xml_dir (str): Pascal VOC XML 文件目录（字符串路径）
        yolo_dir (str): YOLO 标签输出目录（字符串路径）
        class_names (Optional[List[str]]): 指定类别顺序（可选）
        logger (Optional[logging.Logger]): 日志记录器（可选）
    返回：
        List[str]: 最终类别名称列表
    """
    return convert_voc_dir_to_yolo(Path(xml_dir), Path(yolo_dir), class_names, logger)



