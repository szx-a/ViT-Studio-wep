# utils.py - 工具函数：标签加载、图像读取、结果格式化
"""辅助函数，供 model.py / inference.py / train.py 共用。"""

import json
import sys
from pathlib import Path
from PIL import Image

import config


# ---------------------------------------------------------------------------
# 标签
# ---------------------------------------------------------------------------
_labels_cache: dict[int, str] | None = None


def load_labels() -> dict[int, str]:
    """加载标签文件，返回 {class_id: class_name} 映射。

    支持两种格式：
      1) 每行 "class_id<TAB>class_name"
      2) JSON {"0": "name0", "1": "name1", ...}
    """
    global _labels_cache
    if _labels_cache is not None:
        return _labels_cache

    label_file = config.LABEL_FILE
    if not label_file.exists():
        print(f"[警告] 标签文件不存在: {label_file}，使用数字类别名。")
        _labels_cache = {i: f"class_{i}" for i in range(config.NUM_CLASSES)}
        return _labels_cache

    text = label_file.read_text(encoding="utf-8").strip()

    # 尝试 JSON 格式
    if text.startswith("{"):
        raw = json.loads(text)
        _labels_cache = {int(k): v for k, v in raw.items()}
    else:
        # TAB / 空格分隔
        _labels_cache = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)          # 按任意空白拆分，最多两部分
            if len(parts) == 2:
                try:
                    _labels_cache[int(parts[0])] = parts[1]
                except ValueError:
                    pass

    return _labels_cache


# ---------------------------------------------------------------------------
# 图像加载（含容错）
# ---------------------------------------------------------------------------
def load_image(path: str | Path) -> Image.Image:
    """加载图像，失败时打印错误并退出。"""
    path = Path(path)
    if not path.exists():
        print(f"[错误] 图像文件不存在: {path}")
        sys.exit(1)

    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[错误] 无法打开图像 {path}: {e}")
        sys.exit(1)

    return img


# ---------------------------------------------------------------------------
# JSON 输出
# ---------------------------------------------------------------------------
def print_json(results: list[dict], output_path: str | Path | None = None):
    """将结果输出为漂亮 JSON，可选的保存到文件。"""
    text = json.dumps(results, ensure_ascii=False, indent=2)
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        print(f"[完成] 结果已保存至 {output_path}")
    else:
        print(text)
