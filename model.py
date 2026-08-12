# model.py - ViT 模型加载与推理
"""封装 timm 的 ViT 加载逻辑，对外暴露简洁的加载/预测接口。"""

import torch
import timm
from torchvision import transforms
from PIL import Image

import config


# ---------------------------------------------------------------------------
# 全局缓存：模型 & 预处理管线只加载一次
# ---------------------------------------------------------------------------
_model = None
_transform = None


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


def load_model(pretrained: bool = None) -> torch.nn.Module:
    """加载 ViT 模型（缓存复用）。"""
    global _model
    if _model is not None:
        return _model

    if pretrained is None:
        pretrained = config.PRETRAINED

    print(f"[模型] 加载 {config.MODEL_NAME} (pretrained={pretrained}, num_classes={config.NUM_CLASSES}) ...")
    _model = timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
    )
    _model.to(config.DEVICE)
    _model.eval()
    print("[模型] 加载完成。")
    return _model


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(image: Image.Image, top_k: int = None) -> list[dict]:
    """对一张 PIL 图像做推理，返回 Top-K 预测列表。

    返回格式: [{"class_id": int, "class_name": str, "confidence": float}, ...]
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    model = load_model()
    transform = get_transform()

    # 预处理 & 增加 batch 维度
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # 推理
    logits = model(tensor)            # (1, num_classes)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    # 取 Top-K
    values, indices = torch.topk(probs, k=min(top_k, config.NUM_CLASSES))

    # 加载标签映射
    from utils import load_labels
    label_map = load_labels()

    results = []
    for i, (idx, conf) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        class_id = idx
        class_name = label_map.get(idx, f"class_{idx}")
        results.append({
            "rank": i,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(conf, 6),
        })

    return results
