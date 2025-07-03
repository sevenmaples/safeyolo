import argparse
from pathlib import Path
import sys
import os
import shutil
import logging
import json
import yaml
from datetime import datetime
from yoloserver.utils.logging_utils import setup_logging
from yoloserver.utils.config_utils import load_yaml_config, generate_default_yaml, merge_configs
from yoloserver.utils.dataset_utils import get_dataset_info
from yoloserver.scripts.yolo_trains import YOLODatasetProcessor
import platform
import getpass
import socket
import psutil
import torch
import importlib
import subprocess
import time

def print_usage():
    print("""
用法说明：
  -m, --mode      运行模式，可选 train/val/test，默认 train
  -p, --prepare   是否准备数据集（加此参数则准备数据）
  --model         模型类型，如 yolov8, yolov11, custom，默认 yolov8
  --weights       预训练权重路径，默认 pretrained_models/yolov8n.pt

其它参数请在 configs/data.yaml 中配置，或在代码中自动合并。
示例：
  python yolo_train.py -m train -p --model yolov8 --weights pretrained_models/yolov8n.pt
  python yolo_train.py --mode val
""")

def safe_rename_log_file(logger, temp_log_path: Path, model_name: str) -> Path:
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    log_dir = temp_log_path.parent
    existing_logs = [f.name for f in log_dir.glob('train*.log')]
    n = max([int(f.split('-')[0].replace('train', '')) for f in existing_logs if f.startswith('train')] or [0]) + 1
    new_name = log_dir / f'train{n}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-{model_name}.log'
    temp_log_path.rename(new_name)
    return new_name

def log_block(logger, title):
    logger.info("="*40)
    logger.info(title)
    logger.info("="*40)

def log_kv(logger, k, v):
    logger.info(f"{k:<20}: {v}")

def log_param_with_source(logger, k, v, source):
    logger.info(f"{k:<20}: {v}  来源: [{source}]")

def log_device_info(logger):
    log_block(logger, '设备信息概览')
    logger.info('基本设备信息:')
    logger.info(f'    操作系统            : {platform.system()} {platform.version()}')
    logger.info(f'    Python版本          : {platform.python_version()}')
    logger.info(f'    Python解释器路径    : {sys.executable}')
    logger.info(f'    Python虚拟环境      : {os.environ.get("CONDA_DEFAULT_ENV", "无")}')
    logger.info(f'    当前检测时间        : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'    主机名              : {socket.gethostname()}')
    logger.info(f'    当前用户            : {getpass.getuser()}')
    logger.info('CPU信息:')
    logger.info(f'    CPU型号             : {platform.processor()}')
    logger.info(f'    CPU物理核心数       : {psutil.cpu_count(logical=False)}')
    logger.info(f'    CPU逻辑核心数       : {psutil.cpu_count(logical=True)}')
    logger.info(f'    CPU使用率           : {psutil.cpu_percent()}%')
    # GPU信息
    logger.info('GPU信息:')
    try:
        cuda_available = torch.cuda.is_available()
        logger.info(f'    CUDA是否可用        : {cuda_available}')
        if cuda_available:
            logger.info(f'    CUDA版本            : {torch.version.cuda}')
            # 获取NVIDIA驱动版本
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True)
                driver_version = result.stdout.strip() if result.returncode == 0 else '未知'
            except Exception:
                driver_version = '未知'
            logger.info(f'    NVIDIA驱动程序版本  : {driver_version}')
            logger.info(f'    可用的GPU数量       : {torch.cuda.device_count()}')
        else:
            logger.info(f'    NVIDIA驱动程序版本  : 无')
            logger.info(f'    可用的GPU数量       : 0')
    except Exception as e:
        logger.info(f'    CUDA是否可用        : 检测失败({e})')
        logger.info(f'    NVIDIA驱动程序版本  : 检测失败')
        logger.info(f'    可用的GPU数量       : 检测失败')
    # 内存信息
    logger.info('内存信息:')
    mem = psutil.virtual_memory()
    logger.info(f'    总内存              : {mem.total / 1e9:.2f} GB')
    logger.info(f'    已使用内存          : {mem.used / 1e9:.2f} GB')
    logger.info(f'    剩余内存            : {mem.available / 1e9:.2f} GB')
    logger.info(f'    内存使用率          : {mem.percent}%')
    # 环境信息
    logger.info('环境信息:')
    def get_ver(pkg, attr='__version__'):
        try:
            m = importlib.import_module(pkg)
            return getattr(m, attr, '未知')
        except Exception:
            return '未安装'
    logger.info(f'    PyTorch版本         : {get_ver("torch")}')
    logger.info(f'    cuDNN版本           : {getattr(torch.backends.cudnn, "version", lambda: "未知")()}')
    logger.info(f'    Ultralytics_Version : {get_ver("ultralytics")}')
    logger.info(f'    ONNX版本            : {get_ver("onnx")}')
    logger.info(f'    Numpy版本           : {get_ver("numpy")}')
    logger.info(f'    OpenCV版本          : {get_ver("cv2")}')
    logger.info(f'    Pillow版本          : {get_ver("PIL", "PILLOW_VERSION") or get_ver("PIL", "__version__")}')
    logger.info(f'    Torchvision版本     : {get_ver("torchvision")}')
    # 磁盘信息
    logger.info('磁盘信息:')
    disk = psutil.disk_usage('/')
    logger.info(f'    总空间              : {disk.total / 1e9:.2f} GB')
    logger.info(f'    已用空间            : {disk.used / 1e9:.2f} GB')
    logger.info(f'    剩余空间            : {disk.free / 1e9:.2f} GB')
    logger.info(f'    使用率              : {disk.percent}%')
    # GPU详细列表
    logger.info('GPU详细列表:')
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f'  --- GPU {i} 详情 ---')
                logger.info(f'    GPU_{i}_型号               : {props.name}')
                logger.info(f'    GPU_{i}_总显存             : {props.total_memory / 1e9:.2f} GB')
                logger.info(f'    GPU_{i}_算力               : {getattr(props, "major", "?")}.{getattr(props, "minor", "?")}')
                logger.info(f'    GPU_{i}_多处理器数量       : {getattr(props, "multi_processor_count", "?")}')
                # PyTorch显存分配
                allocated = torch.cuda.memory_allocated(i) / 1e6
                cached = torch.cuda.memory_reserved(i) / 1e6
                logger.info(f'    GPU_{i}_PyTorch_已分配显存 : {allocated:.2f} MB')
                logger.info(f'    GPU_{i}_PyTorch_已缓存显存 : {cached:.2f} MB')
                # 利用率（需nvidia-smi）
                try:
                    result = subprocess.run([
                        'nvidia-smi',
                        f'--query-gpu=utilization.gpu,utilization.memory,memory.used',
                        f'--format=csv,noheader,nounits',
                        f'--id={i}'
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        gpu_util, mem_util, mem_used = result.stdout.strip().split(',')
                        logger.info(f'    GPU_{i}_利用率             : GPU:{gpu_util.strip()}% / Mem:{mem_util.strip()}%')
                        logger.info(f'    GPU_{i}_实时使用显存       : {float(mem_used)/1024:.2f} GB' if float(mem_used)>1024 else f'{mem_used} MB')
                    else:
                        logger.info(f'    GPU_{i}_利用率             : 未知')
                        logger.info(f'    GPU_{i}_实时使用显存       : 未知')
                except Exception:
                    logger.info(f'    GPU_{i}_利用率             : 未知')
                    logger.info(f'    GPU_{i}_实时使用显存       : 未知')
        else:
            logger.info('    无可用GPU')
    except Exception as e:
        logger.info(f'    检测GPU详细信息失败: {e}')
    logger.info('='*40)

def log_training_results(logger, results):
    log_block(logger, '训练结果  # 训练性能与指标分块')
    logger.info(f"训练耗时: {getattr(results, 'elapsed', '未知')} 秒  # 总训练耗时")
    logger.info(f"mAP50: {getattr(results.box, 'map50', 0):.4f}  # mAP@0.5 精度")
    logger.info(f"mAP50-95: {getattr(results.box, 'map', 0):.4f}  # mAP@0.5:0.95 精度")
    logger.info(f"Precision: {getattr(results.box, 'mp', 0):.4f}  # 精确率")
    logger.info(f"Recall: {getattr(results.box, 'mr', 0):.4f}  # 召回率")
    logger.info(f"Fitness Score: {getattr(results, 'fitness', 0):.4f}  # 综合得分")
    logger.info(f"保存目录: {getattr(results, 'save_dir', '未知')}  # 训练结果保存路径")
    if hasattr(results.box, 'maps') and hasattr(results, 'names'):
        logger.info("Class-wise mAP@0.5:0.95 (Box Metrics)  # 各类别mAP")
        logger.info("----------------------------------------")
        for i, m in enumerate(results.box.maps):
            logger.info(f"{results.names[i]:<20}: {m:.4f}  # {results.names[i]} 的mAP")
    logger.info("="*60)

def train_model(model_type, weights_path, config, logger):
    if model_type == 'yolov8':
        from ultralytics import YOLO
        logger.info("初始化模型,加载模型: %s", weights_path)
        model = YOLO(weights_path)
        # 训练开始UI
        logger.info("="*60)
        logger.info("【模型训练开始】  # 开始执行模型训练流程")
        logger.info("="*60)
        print("="*60)
        print("【模型训练开始】  # 开始执行模型训练流程")
        print("="*60)
        start = datetime.now()
        # 直接调用 model.train，不再屏蔽输出
        results = model.train(
            data=config['data'],
            epochs=config.get('epochs', 5),
            batch=config.get('batch', 16),
            imgsz=config.get('imgsz', 640),
            device=config.get('device', 0),
            workers=config.get('workers', 8),
            project=config.get('project', 'runs/detect'),
            name=config.get('name', 'train'),
            exist_ok=config.get('exist_ok', False),
        )
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"新能测试：'模型训练' 执行耗时: {elapsed//60:.0f} 分 {elapsed%60:.3f} 秒")
        logger.info("="*60)
        logger.info("【模型训练结束】  # 模型训练流程已完成")
        logger.info("="*60)
        print("="*60)
        print("【模型训练结束】  # 模型训练流程已完成")
        print("="*60)
        logger.info("YOLO Results Summary (Detect Task)")
        logger.info("="*60)
        logger.info(f"Task                : detect")
        logger.info(f"Save Directory      : {results.save_dir}")
        logger.info(f"Timestamp           : {datetime.now().isoformat()}")
        logger.info(f"----------------------------------------")
        logger.info(f"Processing Speed (ms/image)")
        logger.info(f"----------------------------------------")
        logger.info(f"Preprocess          : {getattr(results, 'speed', {}).get('preprocess', 0):.3f} ms")
        logger.info(f"Inference           : {getattr(results, 'speed', {}).get('inference', 0):.3f} ms")
        logger.info(f"Loss Calc           : {getattr(results, 'speed', {}).get('loss', 0):.3f} ms")
        logger.info(f"Postprocess         : {getattr(results, 'speed', {}).get('postprocess', 0):.3f} ms")
        logger.info(f"Total Per Image     : {getattr(results, 'speed', {}).get('total', 0):.3f} ms")
        logger.info(f"----------------------------------------")
        logger.info(f"Overall Evaluation Metrics")
        logger.info(f"----------------------------------------")
        logger.info(f"Fitness Score       : {getattr(results, 'fitness', 0):.4f}")
        logger.info(f"Precision(B)        : {getattr(results.box, 'mp', 0):.4f}")
        logger.info(f"Recall(B)           : {getattr(results.box, 'mr', 0):.4f}")
        logger.info(f"mAP50(B)            : {getattr(results.box, 'map50', 0):.4f}")
        logger.info(f"mAP50-95(B)         : {getattr(results.box, 'map', 0):.4f}")
        logger.info(f"----------------------------------------")
        logger.info(f"Class-wise mAP@0.5:0.95 (Box Metrics)")
        logger.info(f"----------------------------------------")
        if hasattr(results.box, 'maps') and hasattr(results, 'names'):
            for i, m in enumerate(results.box.maps):
                logger.info(f"{results.names[i]:<20}: {m:.4f}")
        logger.info("="*60)
        return results
    elif model_type == 'yolov11':
        logger.info('YOLOv11训练接口待实现（请集成YOLOv11官方API）')
        # 未来集成YOLOv11训练代码
    else:
        logger.error(f'不支持的模型类型: {model_type}，仅支持 yolov8/yolov11')
        raise ValueError(f'不支持的模型类型: {model_type}，仅支持 yolov8/yolov11')

def copy_checkpoint_models(results, model_name, logger):
    log_block(logger, '模型拷贝')
    save_dir = Path(getattr(results, 'save_dir', 'runs/detect/train'))
    weights_dir = save_dir / 'weights'
    checkpoints_dir = Path('models/checkpoints')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    n = save_dir.parent.name.replace('train', '') or '1'
    for pt in ['best.pt', 'last.pt']:
        src = weights_dir / pt
        if src.exists():
            dst = checkpoints_dir / f"train{n}-{time_str}-{model_name}-{pt}"
            shutil.copy(src, dst)
            logger.info(f"{pt} 已拷贝到 {dst}")
        else:
            logger.warning(f"未找到 {src}")

def log_dataset_info(logger, dataset_info, mode, config_file, class_names, sample_count, data_source):
    logger.info('='*40)
    logger.info(f'【数据集信息分块】({mode.capitalize()} 模式)  # 数据集相关统计信息')
    logger.info('-'*40)
    logger.info(f'Config File         : {config_file}    # 数据集配置文件路径')
    logger.info(f'Class Count         : {len(class_names)}    # 类别数量')
    logger.info(f'Class Names         : {", ".join(class_names)}    # 类别名称')
    logger.info(f'Sample Count        : {sample_count}    # 样本总数')
    logger.info(f'Data Source         : {data_source}    # 数据来源路径')
    logger.info('-'*40)

def log_params_info(logger, config, param_sources):
    logger.info('='*40)
    logger.info('===============开始模型参数信息================  # 训练超参数分块')
    logger.info('Parameters  # 训练参数列表')
    logger.info('-'*40)
    for k, v in config.items():
        source = param_sources.get(k, 'YAML')
        logger.info(f'{k:<20}: {v}  来源: [{source}]  # 参数来源')
    logger.info('-'*40)

def log_training_start(logger):
    logger.info('='*40)
    logger.info('【训练流程分块】  # 训练主流程开始')
    logger.info('开始训练...  # 启动模型训练')

def log_model_summary(logger, model):
    logger.info('='*40)
    logger.info('【模型结构摘要】  # 模型结构和参数信息')
    try:
        summary_str = str(model)
        for line in summary_str.splitlines():
            logger.info(line + '  # 模型结构')
    except Exception as e:
        logger.info(f'模型结构摘要获取失败: {e}')
    logger.info('-'*40)

def log_train_param_summary(logger, config, class_names):
    logger.info('='*45)
    logger.info('【训练参数摘要】')
    keys = [
        'batch', 'epochs', 'imgsz', 'device', 'optimizer', 'weights', 'data', 'project', 'name', 'save_dir'
    ]
    for k in keys:
        v = config.get(k, None)
        if v is not None:
            logger.info(f'{k:<12}: {v}')
    logger.info(f'classes     : {", ".join(class_names)}')
    logger.info('='*45)

if __name__ == '__main__':
    print_usage()
    parser = argparse.ArgumentParser(description='YOLO极简训练主脚本')
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'val', 'test'], help='运行模式')
    parser.add_argument('-p', '--prepare', action='store_true', help='准备数据集')
    parser.add_argument('--model', type=str, default='yolov8', choices=['yolov8', 'yolov11'], help='模型类型，仅支持 yolov8 或 yolov11')
    parser.add_argument('--weights', type=str, default='pretrained_models/yolov8n.pt', help='预训练权重路径')
    parser.add_argument('--classes', type=str, nargs='*', help='类别名称列表，如 fire smoke')
    args = parser.parse_args()

    # 日志初始化
    logs_dir = Path('logs')
    logger = setup_logging(
        base_path=logs_dir,
        log_type='train',
        model_name=args.model,
        temp_log=True,
        logger_name='YOLO_TRAIN',
        encoding='utf-8-sig'
    )
    logger.info('========================日志记录器初始化开始=========================')
    logger.info(f'当前日志记录器的根目录: {logs_dir.resolve()}')
    logger.info(f'当前日志记录器的名称: YOLO_TRAIN')
    logger.info(f'当前日志记录器的类型: train')
    logger.info(f'单前日志记录器的级别: INFO')
    logger.info('========================日志记录器初始化成功=========================')
    logger.info('===============================YOLO 训练脚本启动=================================')

    # 设备信息
    log_device_info(logger)

    # 配置合并
    config = merge_configs()
    log_block(logger, '参数信息')
    for k, v in config.items():
        source = '命令行' if k in vars(args) and getattr(args, k, None) is not None else 'YAML'
        log_param_with_source(logger, k, v, source)

    # 合并类别来源，优先级：命令行 > yaml > 自动推断
    class_names = []
    class_source = ''
    if args.classes and len(args.classes) > 0:
        class_names = args.classes
        class_source = '命令行 --classes'
    elif 'names' in config and config['names']:
        class_names = config['names']
        class_source = 'YAML configs/data.yaml'
    else:
        staged_labels_path = config.get('yolo_staged_labels_path', 'raw/yolo_staged_labels')
        class_ids = set()
        import re
        from pathlib import Path
        for txt in Path(staged_labels_path).glob('*.txt'):
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    m = re.match(r'^(\d+)', line.strip())
                    if m:
                        class_ids.add(int(m.group(1)))
        if class_ids:
            class_names = [f'class_{i}' for i in sorted(class_ids)]
        else:
            class_names = []
        class_source = '自动推断标签文件夹'
    logger.info(f'类别来源: {class_source}, 类别: {class_names}')

    # 前置检查：如果未能确定类别，则终止并提供清晰的错误信息
    if not class_names:
        error_msg = ("错误：未能确定类别名称，训练无法继续。\n"
                     "请通过以下方式之一指定类别：\n"
                     "  1. 在命令行中使用 --classes <name1> <name2> ...\n"
                     "  2. 在 configs/data.yaml 文件中定义 'names' 列表。\n"
                     "  3. 确保标签文件(.txt)存在于 raw/yolo_staged_labels/ 目录中且包含有效的类别ID。")
        logger.error(error_msg)
        sys.exit(1)

    # 数据准备
    if args.prepare:
        log_block(logger, '数据准备')
        logger.info('开始数据准备...')
        processor = YOLODatasetProcessor(
            annotation_format=config.get('format', 'yolo'),
            train_rate=config.get('train_rate', 0.8),
            valid_rate=config.get('valid_rate', 0.1),
            classes=class_names,
            coco_task=config.get('coco_task', 'detection'),
            coco_cls91to80=config.get('coco_cls91to80', False),
            logger=logger
        )
        processor.clean_and_initialize_dirs()
        processor.process_dataset()
        logger.info('数据集准备完成。')

    # 数据集信息
    mode_str = args.mode if hasattr(args, 'mode') else 'train'
    dataset_info = get_dataset_info(config.get('data', 'configs/data.yaml'))
    config_file = config.get('data', 'configs/data.yaml')
    class_names_str = class_names if class_names else dataset_info.get('Names', [])
    sample_count = dataset_info.get('Train_Samples', 0)
    data_source = dataset_info.get('Train', '')
    log_dataset_info(logger, dataset_info, mode_str, config_file, class_names_str, sample_count, data_source)

    # 参数信息
    param_sources = {}
    for k in config:
        if k in vars(args) and getattr(args, k, None) is not None:
            param_sources[k] = '命令行'
        else:
            param_sources[k] = 'YAML'
    log_params_info(logger, config, param_sources)
    # 训练参数摘要
    log_train_param_summary(logger, config, class_names_str)

    # 日志重命名（安全）
    log_file = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            log_file = Path(h.baseFilename)
            break
    if log_file:
        new_log = safe_rename_log_file(logger, log_file, args.model)
        logger.info(f'日志文件已重命名为: {new_log}')

    # 为保证训练时使用的data.yaml包含正确的类别信息，动态生成一个唯一的train.yaml配置文件
    data_yaml_path = Path(config.get('data', 'configs/data.yaml'))
    train_yaml_path = None
    try:
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
        else:
            data_config = {
                'train': config.get('train', ''),
                'val': config.get('val', ''),
            }
        data_config['names'] = class_names
        data_config['nc'] = len(class_names)
        # 生成唯一的 train.yaml 路径
        time_str = time.strftime('%Y%m%d_%H%M%S')
        train_yaml_path = data_yaml_path.parent / f"train_{time_str}.yaml"
        with open(train_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"本次训练专用配置文件已生成: {train_yaml_path}")
        training_config = config.copy()
        training_config['data'] = str(train_yaml_path)
        # 主流程分支
        if args.mode == 'train':
            log_training_start(logger)
            results = train_model(args.model, args.weights, training_config, logger)
            from ultralytics import YOLO
            model = YOLO(args.weights)
            log_model_summary(logger, model)
            log_training_results(logger, results)
            copy_checkpoint_models(results, args.model, logger)
            logger.info('训练流程已完成。')
        elif args.mode == 'val':
            log_block(logger, '验证流程')
            logger.info('开始验证...')
            logger.info('验证流程（伪代码）已执行。')
        elif args.mode == 'test':
            log_block(logger, '测试流程')
            logger.info('开始测试...')
            logger.info('测试流程（伪代码）已执行。')
        else:
            logger.warning('未知模式')
    finally:
        # 不自动删除 train.yaml，便于溯源
        pass
    logger.info(f'本次训练所用配置文件: {train_yaml_path}')
    logger.info('主流程执行完毕，详细信息请查看 logs/train 目录下日志文件。')
    print('主流程执行完毕，详细信息请查看 logs/train 目录下日志文件。') 