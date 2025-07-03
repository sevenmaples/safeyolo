import psutil
import platform
import json
try:
    import torch
except ImportError:
    torch = None
import logging

def get_device_info() -> dict:
    try:
        info = {
            'OS': {'Type': platform.system(), 'Version': platform.version()},
            'CPU': {'Cores': psutil.cpu_count(), 'Usage': psutil.cpu_percent()},
            'Memory': {'Total': f'{psutil.virtual_memory().total / 1e9:.2f} GB'},
            'GPU': {'Available': torch.cuda.is_available() if torch else False, 'Count': torch.cuda.device_count() if torch else 0}
        }
        if info['GPU']['Available']:
            info['GPU']['Model'] = torch.cuda.get_device_name(0)
        logging.info(f'设备信息: {json.dumps(info, indent=2, ensure_ascii=False)}')
        return info
    except Exception as e:
        logging.warning(f'设备信息获取失败: {e}')
        return {'Errors': [str(e)]} 