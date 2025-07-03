import yaml
from pathlib import Path
import logging
import argparse

def load_yaml_config(yaml_path: str) -> dict:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        logging.info(f'加载配置文件: {yaml_path}')
        return config
    except FileNotFoundError:
        logging.error(f'配置文件不存在: {yaml_path}')
        raise
    except yaml.YAMLError as e:
        logging.error(f'YAML 解析错误: {e}')
        raise

def generate_default_yaml(yaml_path: str) -> dict:
    default_config = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': 0,
        'names': [],
        'epochs': 5,
        'batch': 16
    }
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(default_config, f)
    logging.info(f'生成默认配置文件: {yaml_path}')
    return default_config

def merge_configs() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=str, default='configs/data.yaml')
    args, _ = parser.parse_known_args()
    config = generate_default_yaml(args.data) if not Path(args.data).exists() else load_yaml_config(args.data)
    final_config = {'epochs': 5, 'batch': 16}  # 默认值
    final_config.update(config)  # YAML 覆盖默认值
    final_config.update({k: v for k, v in vars(args).items() if v is not None})  # CLI 覆盖
    for k, v in final_config.items():
        source = 'CLI' if k in vars(args) and vars(args)[k] is not None else 'YAML' if k in config else '默认值'
        logging.info(f'- {k}: {v} (来源: {source})')
    return final_config 