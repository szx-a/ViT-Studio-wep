# model.py - ViT 模型加载与推理
"""封装 timm 的 ViT 加载逻辑，支持多个微调模型缓存与按需切换。"""

import torch
import timm
from torchvision import transforms
from PIL import Image
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# 全局缓存：预处理管线 + 多模型缓存
# ---------------------------------------------------------------------------
_transform = None
_model_cache: dict[str, tuple[torch.nn.Module, list[str]]] = {}


def get_transform() -> transforms.Compose:
    """返回 ViT 标准预处理管线（复用缓存）。"""
    global _transform
    if _transform is None:
        _transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    return _transform


def _resolve_checkpoint_path(model_key: str) -> Path:
    """把模型 key 解析为 checkpoint 路径。"""
    if model_key in ("_last_train", "_temp"):
        return config.TEMP_CHECKPOINT
    p = Path(model_key)
    if p.exists():
        return p
    return config.CHECKPOINT_DIR / f"{model_key}.pth"


def _build_imagenet_model() -> tuple[torch.nn.Module, list[str]]:
    print(f"[模型] 加载 {config.MODEL_NAME} (ImageNet-1K, num_classes={config.NUM_CLASSES}) ...")
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES,
    )
    model.to(config.DEVICE)
    model.eval()
    print("[模型] 加载完成。")
    return model, [str(i) for i in range(config.NUM_CLASSES)]


def _build_finetuned_model(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    classes = list(ckpt.get("classes") or [])

    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    if not classes and "head.weight" in state_dict:
        num_classes = state_dict["head.weight"].shape[0]
        classes = [f"class_{i}" for i in range(num_classes)]
    num_classes = len(classes)

    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    return model, classes


def get_model_for_key(model_key: str = None) -> tuple[torch.nn.Module, list[str]]:
    """按 key 获取模型和类别列表；未命中时加载并缓存。"""
    model_key = (model_key or config.BUILTIN_MODEL_KEY).strip()
    if model_key == config.BUILTIN_MODEL_KEY:
        cache_key = config.BUILTIN_MODEL_KEY
    else:
        cache_key = str(_resolve_checkpoint_path(model_key).resolve())

    if cache_key not in _model_cache:
        if cache_key == config.BUILTIN_MODEL_KEY:
            _model_cache[cache_key] = _build_imagenet_model()
        else:
            path = _resolve_checkpoint_path(model_key)
            if not path.exists():
                raise FileNotFoundError(f"微调模型不存在: {path}")
            _model_cache[cache_key] = _build_finetuned_model(path)
    return _model_cache[cache_key]


def get_label_map_for_key(model_key: str = None) -> dict[int, dict]:
    """返回 {class_id: {en, zh}}。内置模型用双语标签，微调模型用类别名。"""
    model_key = (model_key or config.BUILTIN_MODEL_KEY).strip()
    if model_key == config.BUILTIN_MODEL_KEY:
        from utils import load_bilingual_labels
        return load_bilingual_labels()

    _, classes = get_model_for_key(model_key)
    label_map = {}
    for i, cls in enumerate(classes):
        zh = config.FINETUNED_LABELS_ZH.get(cls, cls)
        label_map[i] = {"en": cls, "zh": zh}
    return label_map


def load_model(pretrained: bool = None) -> torch.nn.Module:
    """兼容旧接口：返回内置 ImageNet-1K 模型（训练时也用它）。"""
    model, _ = get_model_for_key(config.BUILTIN_MODEL_KEY)
    return model


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(image: Image.Image, top_k: int = None, model_key: str = None) -> list[dict]:
    """对一张 PIL 图像做推理，返回 Top-K 预测列表。

    返回格式: [{"rank", "class_id", "class_name", "class_name_zh", "confidence"}, ...]
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    model_key = (model_key or config.BUILTIN_MODEL_KEY).strip()
    model, classes = get_model_for_key(model_key)
    label_map = get_label_map_for_key(model_key)
    transform = get_transform()

    tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    num_classes = len(classes)
    values, indices = torch.topk(probs, k=min(top_k, num_classes))

    results = []
    for i, (idx, conf) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        info = label_map.get(idx, {"en": f"class_{idx}", "zh": f"类别_{idx}"})
        results.append({
            "rank": i,
            "class_id": idx,
            "class_name": info["en"],
            "class_name_zh": info["zh"],
            "confidence": round(conf, 6),
        })
    return results