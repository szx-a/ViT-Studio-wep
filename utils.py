# utils.py - 工具函数：标签加载、图像读取、结果格式化
"""辅助函数，供 model.py / inference.py / train.py 共用。"""

import json
import sys
from pathlib import Path
from PIL import Image

import config


# ---------------------------------------------------------------------------
# 标签（单语）
# ---------------------------------------------------------------------------
_labels_cache: dict[int, str] | None = None


def load_labels() -> dict[int, str]:
    """加载英文标签，返回 {class_id: class_name}。"""
    global _labels_cache
    if _labels_cache is not None:
        return _labels_cache

    label_file = config.LABEL_FILE
    if not label_file.exists():
        print(f"[警告] 标签文件不存在: {label_file}，使用数字类别名。")
        _labels_cache = {i: f"class_{i}" for i in range(config.NUM_CLASSES)}
        return _labels_cache

    text = label_file.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        raw = json.loads(text)
        _labels_cache = {int(k): v for k, v in raw.items()}
    else:
        _labels_cache = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    _labels_cache[int(parts[0])] = parts[1]
                except ValueError:
                    pass
    return _labels_cache


# ---------------------------------------------------------------------------
# 双语标签
# ---------------------------------------------------------------------------
_bilingual_cache: dict[int, dict] | None = None


def load_bilingual_labels() -> dict[int, dict]:
    """加载双语标签，返回 {class_id: {en: str, zh: str}}。"""
    global _bilingual_cache
    if _bilingual_cache is not None:
        return _bilingual_cache

    bilingual_file = config.BILINGUAL_LABEL_FILE
    if not bilingual_file.exists():
        # fallback 到单语标签
        en_map = load_labels()
        _bilingual_cache = {k: {"en": v, "zh": v} for k, v in en_map.items()}
        return _bilingual_cache

    _bilingual_cache = {}
    for line in bilingual_file.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                idx = int(parts[0])
                _bilingual_cache[idx] = {"en": parts[1], "zh": parts[2]}
            except ValueError:
                pass
        elif len(parts) == 2:
            try:
                idx = int(parts[0])
                _bilingual_cache[idx] = {"en": parts[1], "zh": parts[1]}
            except ValueError:
                pass

    return _bilingual_cache


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
