import argparse
from pathlib import Path
import sys
from yoloserver.utils.dataset_validation import verify_dataset_config, verify_split_uniqueness, delete_invalid_files
from yoloserver.utils.logging_utils import setup_logging
import colorlog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO数据集验证工具", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--yaml', type=str, default='yoloserver/configs/data.yaml', help='data.yaml文件路径')
    parser.add_argument('--mode', type=str, default='FULL', choices=['FULL', 'SAMPLE'], help='验证模式：FULL(全量)/SAMPLE(抽样)')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'segmentation'], help='任务类型')
    parser.add_argument('--delete-invalid', action='store_true', help='是否自动删除不合法文件')
    args = parser.parse_args()

    # 日志初始化（控制台彩色+文件）
    logger = setup_logging(base_path=Path('yoloserver/logs'), log_type='yolo_validate', model_name=None, temp_log=False)
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
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel('INFO')

    yaml_path = Path(args.yaml)
    logger.info(f"[TOP] 开始数据集验证，配置文件: {yaml_path}")

    # 1. 基础验证
    passed, invalid_list = verify_dataset_config(yaml_path, logger, args.mode, args.task)
    if not passed:
        logger.error(f"[TOP] 数据集基础验证未通过，共发现{len(invalid_list)}个问题样本。")
        if args.delete_invalid and invalid_list:
            logger.warning("[TOP] 启动自动删除不合法文件...")
            delete_invalid_files(invalid_list, logger)
            logger.warning("[TOP] 不合法文件已删除，请重新检查数据集！")
        else:
            logger.warning("[TOP] 检测到不合法文件，未自动删除。可加 --delete-invalid 参数自动删除。")
    else:
        logger.info("[TOP] 数据集基础验证通过！")

    # 2. 分割唯一性验证
    unique = verify_split_uniqueness(yaml_path, logger)
    if not unique:
        logger.error("[TOP] 数据集分割唯一性验证未通过，请检查train/val/test是否有重复图片！")
    else:
        logger.info("[TOP] 数据集分割唯一性验证通过！")

    # 3. 总结
    if passed and unique:
        logger.info("[TOP] 数据集验证全部通过，可以安全开始训练！")
        sys.exit(0)
    else:
        logger.error("[TOP] 数据集验证未通过，请根据日志修复问题后重试！")
        sys.exit(1) 