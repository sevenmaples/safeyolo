import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Set, Any, Union

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


def _parse_xml_annotation(xml_path: Path, classes: List[str]) -> List[str]:
    """
    核心功能: 解析pascal_voc 为 Yolo 格式，支持自动模式和手动模式
    :param xml_path: xml的地址
    :param classes: 自定义的列表
    :return: 解析后的列表
    """
    yolo_labels = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_elem = root.find("size")

        if size_elem is None:
            logging.error(f"XML文件 '{xml_path.name}'格式错误: 缺少size元素,无法提取图片尺寸信息，跳过")
            return []

        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)

        if width <= 0 or height <= 0:
            logging.error(f"XML文件 '{xml_path.name}'格式错误: 图片尺寸信息错误 (W：{width}, H: {height})，跳过")
            return []

        for obj in root.iter("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                logging.warning(f"XML文件 '{xml_path.name}'格式错误: 缺少name元素,跳过")
                continue
            name = name_elem.text.strip().lower()  # 优化：strip+lower

            # 优化：比对时也用strip+lower
            if name not in [c.strip().lower() for c in classes]:
                continue  # 直接跳过当前对象，不进行后续处理
            class_id = [c.strip().lower() for c in classes].index(name)
            xml_box = obj.find("bndbox")
            if xml_box is None:
                logging.error(f"XML文件 '{xml_path.name}'格式错误:对象 '{name}' 缺少bndbox元素,跳过")
                continue
            try:
                xmin = int(xml_box.find("xmin").text)
                ymin = int(xml_box.find("ymin").text)
                xmax = int(xml_box.find("xmax").text)
                ymax = int(xml_box.find("ymax").text)
            except (AttributeError, ValueError) as e:
                logging.error(f"XML文件 '{xml_path.name}'格式错误:对象 '{name}'的边界框解析失败,跳过: {e}")
                continue

            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                logging.warning(f"XML文件 '{xml_path.name}'格式错误:对象 '{name}'的边界框超出图片尺寸范围，跳过")
                continue

            center_x = (xmin + xmax) / 2 / width
            center_y = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            center_x  = max(0.0, min(1.0, center_x))
            center_y  = max(0.0, min(1.0, center_y))
            box_width  = max(0.0, min(1.0, box_width))
            box_height  = max(0.0, min(1.0, box_height))

            yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        return yolo_labels
    except FileNotFoundError:
        logging.error(f"XML文件 '{xml_path.name}'不存在，跳过")
    except ET.ParseError as e:
        logging.error(f"XML文件 '{xml_path.name}'解析失败: {e}，跳过")
    except Exception as e:
        logging.error(f"XML文件 '{xml_path.name}'处理失败: {e}，跳过")
    return []


def convert_pascal_voc_to_yolo(
        xml_input_dir: Path,
        output_yolo_txt_dir: Path,
        target_classes_for_yolo: Union[List[str], None] = None) -> List[str]:
    """
    核心转换函数
    :param xml_input_dir: xml的输入地址
    :param output_yolo_txt_dir: 输出地址
    :param target_classes_for_yolo: 目标标签列表
    :return: 目标标签列表
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"开始将Pascal VOC XML文件从 '{xml_input_dir}' "
                f"转换为Yolo格式文件 '{output_yolo_txt_dir}'".center(50, "="))

    if not xml_input_dir.exists():
        logger.error(f"输入目录 '{xml_input_dir}' 不存在")
        raise FileNotFoundError(f"输入目录 '{xml_input_dir}' 不存在")

    xml_files_found = list(xml_input_dir.glob("*.xml"))
    if not xml_files_found:
        logger.error(f"输入目录 '{xml_input_dir}' 中不存在XML文件")
        raise []

    if target_classes_for_yolo is not None:
        # 优化：手动模式也统一strip+lower
        classes = [c.strip().lower() for c in target_classes_for_yolo]
        logger.info(f"Pascal VOC转换模式为：手动模式，已指定目标类别为: {classes}".center(50, "="))
    else:
        unique_classes: Set[str] = set()
        logger.info(f"Pascal VOC转换模式为：自动模式，"
                    f"开始扫描XML文件以获取所有类别信息".center(50, "="))
        for xml_file in xml_files_found:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.iter("object"):
                    name_elem = obj.find("name")
                    if name_elem is not None and name_elem.text:
                        unique_classes.add(name_elem.text.strip().lower())  # 优化：strip+lower
            except ET.ParseError as e:
                logger.warning(f"扫描XML文件 '{xml_file.name}'解析失败: {e}")
            except Exception as e:
                logger.warning(f"扫描XML文件 '{xml_file.name}'处理失败: {e}")
        classes = sorted(list(unique_classes))
        if not classes:
            logger.error(f"从XML文件：{xml_input_dir}中未找到任何类别信息，请检查XML文件")
            raise []
        logger.info(f"Pascal Voc模式转换，自动模式，已获取所有类别信息: {classes}".center(50, "="))

    output_yolo_txt_dir.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    for xml_file in xml_files_found:
        yolo_labels = _parse_xml_annotation(xml_file, classes)
        if yolo_labels:
            txt_file_path = output_yolo_txt_dir / (xml_file.stem + ".txt")
            try:
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    for label in yolo_labels:
                        f.write(label + "\n")
                converted_count += 1
            except Exception as e:
                logger.error(f"写入Yolo格式文件 '{txt_file_path.name}' 失败: {e}")
                continue
        else:
            logger.warning(f"XML文件 '{xml_file.name}' 未生成有效的Yolo标签，可能为无类别目标或解析失败")
    logger.info(f"从'{xml_input_dir}'转换完成，共转换 {converted_count} "
                f"个XML文件为Yolo格式文件，保存在 '{output_yolo_txt_dir}'")
    return classes


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
    return convert_pascal_voc_to_yolo(Path(xml_dir), Path(yolo_dir), class_names)



