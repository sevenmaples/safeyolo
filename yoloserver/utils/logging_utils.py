import sys
from pathlib import Path

# 自动将项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime
from yoloserver.utils import path
import colorlog


def setup_logging(base_path: Path,
                  log_type: str = "general",
                  model_name: str = None,
                  log_level: int = logging.INFO,
                  temp_log: bool = False,
                  logger_name: str = "YOLO DEFAULT",
                  encoding: str = "utf-8"):
    """
    配置日志记录器，确保日志存储到指定路径的子目录当中，并同时输出到控制台，
    日志文件名称同时包含类型和时间戳

    :param base_path: 日志文件的根路径
    :param log_type: 日志的类别
    :param model_name: 可选模型名称，用于日志文件名
    :param log_level: 日志记录器最低记录级别
    :param temp_log: 是否启用临时命名
    :param logger_name: 日志记录器logger实例名称
    :param encoding: 日志文件的编码格式
    :return: logger 配置好的日志记录器
    """
    # 1. 构建日志文件存放的完整路径
    log_dir = base_path / log_type
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成带有时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 根据temp_log参数决定是否启用临时文件名
    prefix = "temp" if temp_log else log_type.replace(" ", "_")

    # 构建日志文件名，前缀_时间戳_模型名称.log
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "_"))

    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename

    # 3. 获取或创建指定的名称的日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    # 阻止日志时间传播到父级logger
    logger.propagate = False

    # 4. 避免重复添加日志处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # 5. 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s"))
    logger.addHandler(file_handler)

    # 6. 创建控制台处理器
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
    logger.addHandler(console_handler)

    # 7. 输出一些初始化信息
    logger.info("日志记录器开始初始化".center(50, '='))
    logger.info(f"日志记录器已初始化，日志文件保存在: {log_file}")
    logger.info(f"日志记录器初始化时间：{datetime.now()}")
    logger.info(f"日志记录器名称: {logger_name}")
    logger.info(f"日志记录器最低记录级别: {logging.getLevelName(log_level)}")
    logger.info("日志记录器初始化完成".center(50, '='))

    return logger

# --- 用于测试 ---
if __name__ == '__main__':
    print("--- 场景1: 测试带时间戳的永久日志 ---")
    init_logger = setup_logging(
        base_path=path.LOGS_DIR,
        logger_name='project_initializer', 
        log_type='project_init'
    )
    init_logger.info("这是一条项目初始化的 INFO 日志。")
    init_logger.warning("这是一条项目初始化的 WARNING 日志。")

    print("--- 场景2: 测试可覆盖的临时日志 (DEBUG 级别) ---")
    conv_logger = setup_logging(
        base_path=path.LOGS_DIR,
        logger_name='data_converter', 
        log_type='data_conversion', 
        log_level=logging.DEBUG, 
        temp_log=True
    )
    conv_logger.debug("这是一条数据转换的 DEBUG 日志。")
    conv_logger.error("这是一条数据转换的 ERROR 日志。")