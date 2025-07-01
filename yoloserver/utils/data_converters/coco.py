import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

YOLO_EXT = ".txt"
LOGGER_NAME = "coco2yolo"


def parse_coco_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    解析 COCO JSON 文件，返回 images、annotations、categories 列表。
    参数：
        json_path (Path): COCO JSON 文件路径
    返回：
        Optional[Dict[str, Any]]: 解析后的字典，若失败则为 None
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"解析 COCO JSON 失败: {json_path} | 错误: {e}")
        return None


def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """
    COCO bbox [x_min, y_min, width, height] 转为 YOLO 格式 [x_center, y_center, w, h]（归一化）。
    参数：
        bbox (List[float]): [x_min, y_min, width, height]
        img_w (int): 图片宽度
        img_h (int): 图片高度
    返回：
        List[float]: [x_center, y_center, w, h]（归一化）
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [x_center, y_center, w_norm, h_norm]


def convert_coco_json_to_yolo(
    json_path: Path,
    yolo_dir: Path,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    将 COCO JSON 标注文件转为 YOLO txt 格式，写入 yolo_dir，并返回类别名称列表。
    参数：
        json_path (Path): COCO JSON 文件路径
        yolo_dir (Path): YOLO 标签输出目录
        class_names (Optional[List[str]]): 指定类别顺序（可选），如不指定则自动收集
        logger (Optional[logging.Logger]): 日志记录器（可选）
    返回：
        List[str]: 最终类别名称列表
    """
    logger = logger or logging.getLogger(LOGGER_NAME)
    if not json_path.exists():
        logger.error(f"输入文件不存在: {json_path}")
        return []
    yolo_dir.mkdir(parents=True, exist_ok=True)

    data = parse_coco_json(json_path)
    if data is None:
        return []

    # 收集类别
    coco_categories = data.get("categories", [])
    id2name = {cat["id"]: cat["name"] for cat in coco_categories}
    all_classes = set(id2name.values())
    if class_names is None:
        class_names = sorted(all_classes)
    class2id = {name: idx for idx, name in enumerate(class_names)}

    # 构建图片信息映射
    images = {img["id"]: img for img in data.get("images", [])}
    # 按图片分组标注
    imgid2annos = {}
    for anno in data.get("annotations", []):
        imgid2annos.setdefault(anno["image_id"], []).append(anno)

    for img_id, img_info in images.items():
        img_w = img_info.get("width")
        img_h = img_info.get("height")
        file_name = img_info.get("file_name")
        if img_w is None or img_h is None or file_name is None:
            logger.warning(f"图片信息缺失: {img_info}")
            continue
        yolo_lines = []
        for anno in imgid2annos.get(img_id, []):
            cat_id = anno.get("category_id")
            cat_name = id2name.get(cat_id)
            if cat_name not in class2id:
                logger.warning(f"类别未在类别列表中: {cat_name} | 文件: {file_name}")
                continue
            bbox = anno.get("bbox")
            if not bbox or len(bbox) != 4:
                logger.warning(f"无效 bbox: {anno}")
                continue
            class_id = class2id[cat_name]
            yolo_box = coco_bbox_to_yolo(bbox, img_w, img_h)
            yolo_line = f"{class_id} {' '.join(f'{x:.6f}' for x in yolo_box)}"
            yolo_lines.append(yolo_line)
        # 写入 txt
        txt_name = Path(file_name).stem + YOLO_EXT
        txt_path = yolo_dir / txt_name
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            logger.info(f"已生成: {txt_path}（{len(yolo_lines)} 个目标）")
        except OSError as e:
            logger.error(f"写入 YOLO 标签失败: {txt_path} | 错误: {e}")

    logger.info(f"COCO->YOLO 转换完成，类别列表: {class_names}")
    return class_names


def coco2yolo_convert(
    json_path: str,
    yolo_dir: str,
    class_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    高层封装接口，支持字符串路径和类别，自动转换为 Path 并调用核心转换函数。
    参数：
        json_path (str): COCO JSON 文件路径（字符串路径）
        yolo_dir (str): YOLO 标签输出目录（字符串路径）
        class_names (Optional[List[str]]): 指定类别顺序（可选）
        logger (Optional[logging.Logger]): 日志记录器（可选）
    返回：
        List[str]: 最终类别名称列表
    """
    return convert_coco_json_to_yolo(Path(json_path), Path(yolo_dir), class_names, logger)
