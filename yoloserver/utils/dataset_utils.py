from pathlib import Path
import yaml
import logging
from utils.config_utils import load_yaml_config

def get_dataset_info(yaml_path: str) -> dict:
    config = load_yaml_config(yaml_path)
    train_path = Path(config.get('train', 'images/train'))
    val_path = Path(config.get('val', 'images/val'))
    info = {
        'Config': yaml_path,
        'Classes': config.get('nc', 0),
        'Names': config.get('names', []),
        'Train_Samples': len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png'))),
        'Val_Samples': len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
    }
    logging.info(f'数据集信息: {info}')
    return info 