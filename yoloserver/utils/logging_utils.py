import logging
import sys
from datetime import datetime
from yoloserver.utils import path

def setup_logging(logger_name: str, 
                  log_type: str, 
                  level: int = logging.INFO, 
                  temp_log: bool = False) -> logging.Logger:
    """
    配置并返回一个功能强大的日志记录器 (Logger)。

    该函数实现了以下功能:
    1.  集中化配置: 项目中任何需要日志的地方都可以调用此函数获取 logger。
    2.  参数化支持: 可自定义 logger 名称、日志类型、日志级别。
    3.  避免重复: 在配置前会清空已有的 handlers，防止日志重复输出。
    4.  结构化管理: 日志会根据 log_type 存放在不同的子目录中。
    5.  动态命名: 日志文件名可带时间戳，或使用固定的临时文件名。

    Args:
        logger_name (str): 日志记录器的唯一名称，建议使用模块名 (e.g., __name__)。
        log_type (str): 日志的类型，决定了日志文件的存放目录 (e.g., 'project_init', 'data_conversion')。
        level (int, optional): 日志记录的级别。默认为 logging.INFO。
        temp_log (bool, optional): 是否使用临时日志文件。
                                   如果为 True，日志文件名为 {log_type}_temp.log，每次运行会覆盖。
                                   如果为 False，日志文件名将带有时间戳。
                                   默认为 False。

    Returns:
        logging.Logger: 配置好的日志记录器实例。
    """
    # 1. 获取 logger 实例
    logger = logging.getLogger(logger_name)

    # 2. 避免重复添加处理器 (Requirement 3)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. 设置日志级别
    logger.setLevel(level)

    # 4. 结构化和动态日志文件管理 (Requirement 4)
    # 根据 log_type 创建日志子目录
    log_directory = path.LOGS_DIR / log_type
    log_directory.mkdir(parents=True, exist_ok=True)

    # 根据 temp_log 参数决定日志文件名
    if temp_log:
        log_file_path = log_directory / f"{log_type}_temp.log"
        file_mode = 'w'  # 覆盖模式，适合临时日志
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_directory / f"{log_type}_{timestamp}.log"
        file_mode = 'a'  # 追加模式

    # 5. 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 6. 创建并添加处理器 (Console 和 File)
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, mode=file_mode, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 7. 禁止日志向上传播
    logger.propagate = False

    return logger

# --- 用于测试 ---
if __name__ == '__main__':
    # 场景1: 测试带时间戳的永久日志
    print("--- 场景1: 测试带时间戳的永久日志 ---")
    init_logger = setup_logging(
        logger_name='project_initializer', 
        log_type='project_init'
    )
    init_logger.info("这是一条项目初始化的 INFO 日志。")
    init_logger.warning("这是一条项目初始化的 WARNING 日志。")
    print(f"日志文件已保存至: {init_logger.handlers[1].baseFilename}\n")

    # 场景2: 测试可覆盖的临时日志，并设置级别为 DEBUG
    print("--- 场景2: 测试可覆盖的临时日志 (DEBUG 级别) ---")
    conv_logger = setup_logging(
        logger_name='data_converter', 
        log_type='data_conversion', 
        level=logging.DEBUG, 
        temp_log=True
    )
    conv_logger.debug("这是一条数据转换的 DEBUG 日志。")
    conv_logger.error("这是一条数据转换的 ERROR 日志。")
    print(f"临时日志文件已保存至: {conv_logger.handlers[1].baseFilename}") 