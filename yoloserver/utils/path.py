from pathlib import Path

# YOLO服务的项目根目录，无论项目从哪个目录运行，该目录都是项目根目录
YOLOSERVER_ROOT = Path(__file__).resolve().parent.parent

# 核心目录定义
# 配置文件目录
CONFIGS_DIR = YOLOSERVER_ROOT / "configs"

# 数据存放目录
DATA_DIR = YOLOSERVER_ROOT / "data"

# 结果存放目录
RUNS_DIR = YOLOSERVER_ROOT / "runs"

# 模型存放目录
MODELS_DIR = YOLOSERVER_ROOT / "models"

# 预训练模型存放目录
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"

# 训练好的模型存放目录
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# 顶层脚本存放目录
SCRIPTS_DIR = YOLOSERVER_ROOT / "scripts"

# 日志文件存放目录
LOGS_DIR = YOLOSERVER_ROOT / "logs"

# 具体数据存放路径
RAW_DATA_DIR = DATA_DIR / "raw"

# 原始图像存放路径
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
# 原始非YOLO标注标签存放路径
ORIGINAL_ANNOTATIONS_DIR = RAW_DATA_DIR / "original_annotations"

# YOLO txt 数据暂存目录
YOLO_STAGED_LABELS_DIR = RAW_DATA_DIR / "yolo_staged_labels"


if __name__ == "__main__":
    for _path in [
        YOLOSERVER_ROOT,
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
    ]:
        _path.mkdir(parents=True, exist_ok=True)
