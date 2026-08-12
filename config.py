# config.py - ViT 项目全局配置
"""所有可调参数集中管理，修改此处即可影响整个项目。"""

from pathlib import Path

# ---- 模型 ----
MODEL_NAME = "vit_base_patch16_224"          # timm 模型名
PRETRAINED = True                             # 是否加载预训练权重
NUM_CLASSES = 1000                            # 输出类别数（ImageNet-1K 默认 1000；21K 时改为 21841）
IMAGE_SIZE = 224                              # 模型输入尺寸

# ---- 设备 ----
DEVICE = "cuda"                               # "cpu" 或 "cuda"

# ---- 推理 ----
DEFAULT_TOP_K = 5                             # 默认返回 Top-K 结果

# ---- 训练 ----
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_WORKERS = 0                       # Windows 下多进程 DataLoader 容易出问题
USE_AMP = True                                # 自动混合精度（fp16），GPU 训练时推荐开启
AMP_DTYPE = "float16"                         # 混合精度类型：float16 或 bfloat16

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent
LABELS_DIR = BASE_DIR / "labels"
LABEL_FILE = LABELS_DIR / "imagenet_label.txt"      # ImageNet-1K 类别名
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BILINGUAL_LABEL_FILE = LABELS_DIR / "imagenet_label_bilingual.txt"   # 中英双语标签
