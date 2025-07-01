# @Function  :用于初始化项目结构，包括检查并创建核心结构

import sys
import logging
from pathlib import Path

# 自动将项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from yoloserver.utils.logging_utils import setup_logging #不暴露外部接口
from yoloserver.utils import setup_logging#暴露外部接口后可以这样调用
from yoloserver.utils import LOGS_DIR
from yoloserver.utils import (
    YOLOSERVER_ROOT, 
    CONFIGS_DIR, DATA_DIR, 
    RUNS_DIR, MODELS_DIR, 
    PRETRAINED_MODELS_DIR,
    CHECKPOINTS_DIR, 
    SCRIPTS_DIR, 
    RAW_DATA_DIR, 
    RAW_IMAGES_DIR,
    ORIGINAL_ANNOTATIONS_DIR, 
    YOLO_STAGED_LABELS_DIR
)

# =========================
# 1. 配置日志记录器
# =========================
logger = setup_logging(
    base_path=LOGS_DIR,
    log_type="init_project",
    log_level=logging.INFO,
    logger_name="YOLO Init Project"
)

def initialize_project(logger_instance:logging.Logger=logger):
    """
    检查并创建项目所需文件夹的结构，并检查原始数据集目录
    :param logger_instance: 日志记录器实例
    """
    logger_instance.info("开始检查并创建项目核心目录".center(50,"="))

    # =========================
    # 2. 定义需要创建的标准目录
    # =========================
    standard_data_to_create = [
        CONFIGS_DIR, 
        DATA_DIR, 
        RUNS_DIR, 
        MODELS_DIR, 
        PRETRAINED_MODELS_DIR,
        CHECKPOINTS_DIR, 
        SCRIPTS_DIR, 
        LOGS_DIR, 
        RAW_DATA_DIR, 
        RAW_IMAGES_DIR,
        ORIGINAL_ANNOTATIONS_DIR, 
        YOLO_STAGED_LABELS_DIR,
        DATA_DIR / "train" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "val" / "images",
        DATA_DIR / "val" / "labels",
        DATA_DIR / "test" / "images",
        DATA_DIR / "test" / "labels",
    ]

    created_dirs = []   # 新创建的目录
    existed_dirs = []   # 已存在的目录

    # =========================
    # 3. 检查并创建目录
    # =========================
    for d in standard_data_to_create:
        if not d.exists():
            try:
                d.mkdir(parents=True,exist_ok=True)
                logger_instance.info(f"已创建目录: {d.relative_to(YOLOSERVER_ROOT)}")
                created_dirs.append(str(d.relative_to(YOLOSERVER_ROOT)))
            except Exception as e:
                logger_instance.error(f"创建目录失败: {d.relative_to(YOLOSERVER_ROOT)} | 错误: {e}")
        else:
            logger_instance.info(f"目录已存在: {d.relative_to(YOLOSERVER_ROOT)}")
            existed_dirs.append(str(d.relative_to(YOLOSERVER_ROOT)))

    logger_instance.info("目录检查与创建完毕".center(50,"="))

    # =========================
    # 4. 检查原始数据集目录状态
    # =========================
    logger_instance.info("检查原始数据集目录".center(50,"="))
    raw_dirs_to_check = [
        ("原始图片目录", RAW_IMAGES_DIR),
        ("原始标注目录", ORIGINAL_ANNOTATIONS_DIR)
    ]
    for desc, raw_dir in raw_dirs_to_check:
        if not raw_dir.exists():
            # 目录不存在
            logger_instance.warning(f"{desc} 不存在: {raw_dir.relative_to(YOLOSERVER_ROOT)}")
        elif not any(raw_dir.iterdir()):
            # 目录存在但为空
            logger_instance.warning(f"{desc} 为空: {raw_dir.relative_to(YOLOSERVER_ROOT)}，请放入相应数据文件。")
        else:
            # 目录存在且有内容
            logger_instance.info(f"{desc} 已就绪: {raw_dir.relative_to(YOLOSERVER_ROOT)}，文件数: {len(list(raw_dir.iterdir()))}")

    logger_instance.info("项目初始化流程结束".center(50,"="))

if __name__ == "__main__":
    initialize_project(logger_instance=logger)

