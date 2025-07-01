import sys
from pathlib import Path

# --- 动态添加项目根目录到 sys.path ---
# 这解决了在终端直接运行脚本时, 'yoloserver' 模块找不到的问题。
# __file__ -> .../safeyolo/yoloserver/scripts/initialize_project.py
# .parent -> .../yoloserver/scripts
# .parent.parent -> .../yoloserver
# .parent.parent.parent -> .../safeyolo (项目根目录)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import os
from yoloserver.utils import path
from yoloserver.utils.logging_utils import setup_logging

def create_directories(logger):
    """
    根据 path.py 中的定义创建所有标准目录。
    
    Args:
        logger: 配置好的日志记录器实例。
    """
    logger.info("--- 1. 开始创建标准目录结构 ---")
    
    # 从 path.py 动态收集所有以 _DIR 结尾的路径变量
    dirs_to_create = [
        v for k, v in path.__dict__.items() if k.endswith("_DIR") and isinstance(v, path.Path)
    ]
    
    # 为了让日志输出的父子目录关系更清晰，先对路径进行排序
    for directory in sorted(dirs_to_create):
        # 检查目录是否存在
        if directory.exists():
            logger.info(f"目录已存在，跳过: {directory.relative_to(path.YOLOSERVER_ROOT)}")
        else:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"成功创建目录: {directory.relative_to(path.YOLOSERVER_ROOT)}")
            except OSError as e:
                logger.error(f"创建目录失败: {directory.relative_to(path.YOLOSERVER_ROOT)} | 错误: {e}")
    logger.info("--- 目录结构检查/创建完成 ---\n")

def check_raw_data_directories(logger):
    """
    检查原始数据目录的状态并提供用户指导。

    Args:
        logger: 配置好的日志记录器实例。
    """
    logger.info("--- 2. 开始检查原始数据目录 ---")
    
    # 检查原始图片目录
    try:
        num_images = len(os.listdir(path.RAW_IMAGES_DIR))
        status = "为空" if num_images == 0 else f"包含 {num_images} 个文件"
        logger.info(f"检查 '{path.RAW_IMAGES_DIR.relative_to(path.YOLOSERVER_ROOT)}' ... 状态: {status}")
    except FileNotFoundError:
        logger.warning(f"目录 '{path.RAW_IMAGES_DIR.relative_to(path.YOLOSERVER_ROOT)}' 未找到，请先确保目录已创建。")


    # 检查原始标注目录
    try:
        num_annotations = len(os.listdir(path.ORIGINAL_ANNOTATIONS_DIR))
        status = "为空" if num_annotations == 0 else f"包含 {num_annotations} 个文件"
        logger.info(f"检查 '{path.ORIGINAL_ANNOTATIONS_DIR.relative_to(path.YOLOSERVER_ROOT)}' ... 状态: {status}")
    except FileNotFoundError:
        logger.warning(f"目录 '{path.ORIGINAL_ANNOTATIONS_DIR.relative_to(path.YOLOSERVER_ROOT)}' 未找到，请先确保目录已创建。")

    logger.info("--- 原始数据目录检查完成 ---\n")
    
    # 打印用户操作指南
    print("="*80)
    print("【下一步操作指南】")
    print("项目初始化已完成。请将您的原始数据（图片和标注文件）放入以下目录：")
    print(f"  -> 原始图片 (如 .jpg, .png): \n     {path.RAW_IMAGES_DIR}")
    print(f"  -> 原始标注 (如 .xml): \n     {path.ORIGINAL_ANNOTATIONS_DIR}")
    print("="*80)

def main():
    """
    项目初始化主函数
    """
    # 1. 初始化日志记录器
    # 使用 __name__ 作为 logger_name 是一个好习惯
    logger = setup_logging(logger_name=__name__, log_type='project_init')
    logger.info("==========================================")
    logger.info("          启动 SafeYolo 项目初始化脚本          ")
    logger.info("==========================================\n")
    
    # 2. 创建目录结构
    create_directories(logger)
    
    # 3. 检查原始数据目录并提供指导
    check_raw_data_directories(logger)
    
    logger.info("\n项目初始化流程顺利结束。")

if __name__ == '__main__':
    main() 