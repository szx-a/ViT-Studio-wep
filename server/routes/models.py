# server/routes/models.py - 模型管理与评分 API

import sys
import json
import shutil
import threading
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast

from fastapi import APIRouter, Form

import config
from model import get_model_for_key, get_transform
from dataset import ImageFolderDataset

router = APIRouter(prefix="/api/models", tags=["models"])

DATASET_ROOT = Path(config.BASE_DIR) / "datasets"
CHECKPOINT_DIR = config.CHECKPOINT_DIR
META_FILE = config.MODEL_META_FILE

_eval_state = {
    "running": False,
    "message": "空闲",
    "progress": 0.0,
}
_eval_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------
def _load_meta() -> dict:
    if not META_FILE.exists():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, "_")
    return name


def _read_checkpoint_meta(path: Path) -> dict:
    """读取 checkpoint 的 classes/val_acc，失败返回空。"""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        classes = list(ckpt.get("classes") or [])
        val_acc = ckpt.get("val_acc")
        return {"classes": classes, "val_acc": val_acc}
    except Exception:
        return {"classes": [], "val_acc": None}


# ---------------------------------------------------------------------------
# 列表
# ---------------------------------------------------------------------------
@router.get("")
async def list_models():
    models = [{
        "key": config.BUILTIN_MODEL_KEY,
        "name": config.BUILTIN_MODEL_NAME,
        "builtin": True,
        "num_classes": config.NUM_CLASSES,
        "classes": [],
        "val_acc": None,
        "size_mb": config.BUILTIN_MODEL_SIZE_MB,
        "dataset": None,
        "created": None,
        "pending": False,
    }]

    meta = _load_meta()
    for p in sorted(CHECKPOINT_DIR.glob("*.pth")):
        if p.name == config.TEMP_CHECKPOINT.name:
            continue
        key = p.stem
        m = meta.get(key, {})
        if not m:
            ck = _read_checkpoint_meta(p)
            m = {
                "key": key,
                "name": key,
                "classes": ck["classes"],
                "val_acc": ck["val_acc"],
                "dataset": None,
                "created": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            }
            meta[key] = m
            _save_meta(meta)
        models.append({
            "key": key,
            "name": m.get("name", key),
            "builtin": False,
            "num_classes": len(m.get("classes", [])),
            "classes": m.get("classes", []),
            "val_acc": m.get("val_acc"),
            "size_mb": m.get("size_mb", round(p.stat().st_size / 1024 / 1024, 2)),
            "dataset": m.get("dataset"),
            "created": m.get("created"),
            "pending": False,
        })

    if config.TEMP_CHECKPOINT.exists():
        tmp = _read_checkpoint_meta(config.TEMP_CHECKPOINT)
        models.append({
            "key": "_last_train",
            "name": "未命名（训练结果）",
            "builtin": False,
            "num_classes": len(tmp["classes"]),
            "classes": tmp["classes"],
            "val_acc": tmp.get("val_acc"),
            "size_mb": round(config.TEMP_CHECKPOINT.stat().st_size / 1024 / 1024, 2),
            "dataset": None,
            "created": None,
            "pending": True,
        })

    return {"models": models}


@router.get("/finetuned")
async def list_finetuned():
    """返回可用于识别的微调模型 key 列表。"""
    data = await list_models()
    keys = [m["key"] for m in data["models"] if not m["builtin"]]
    return {"models": keys}


# ---------------------------------------------------------------------------
# 保存 / 删除
# ---------------------------------------------------------------------------
@router.post("/save")
async def save_model(name: str = Form(...)):
    name = _safe_name(name)
    if not name or name == config.BUILTIN_MODEL_KEY:
        return {"status": "error", "message": "模型名不合法"}

    src = config.TEMP_CHECKPOINT
    if not src.exists():
        return {"status": "error", "message": "没有待处理的训练模型"}

    dst = CHECKPOINT_DIR / f"{name}.pth"
    if dst.exists():
        return {"status": "error", "message": f"模型名已存在: {name}"}

    shutil.copyfile(src, dst)
    ck = _read_checkpoint_meta(dst)

    meta = _load_meta()
    meta[name] = {
        "key": name,
        "name": name,
        "classes": ck["classes"],
        "val_acc": ck.get("val_acc"),
        "dataset": None,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "size_mb": round(dst.stat().st_size / 1024 / 1024, 2),
    }
    _save_meta(meta)
    src.unlink(missing_ok=True)
    return {"status": "ok", "model": meta[name]}


@router.delete("/{name}")
async def delete_model(name: str):
    if name == config.BUILTIN_MODEL_KEY:
        return {"status": "error", "message": "内置 ImageNet-1K 模型不可删除"}

    if name == "_last_train":
        config.TEMP_CHECKPOINT.unlink(missing_ok=True)
        return {"status": "ok", "name": name}

    path = CHECKPOINT_DIR / f"{name}.pth"
    if not path.exists():
        return {"status": "not_found", "name": name}

    path.unlink(missing_ok=True)
    meta = _load_meta()
    meta.pop(name, None)
    _save_meta(meta)
    return {"status": "ok", "name": name}


@router.post("/discard")
async def discard_model():
    config.TEMP_CHECKPOINT.unlink(missing_ok=True)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------
@router.post("/evaluate")
def evaluate_model(
    model_key: str = Form(...),
    dataset_name: str = Form(...),
    batch_size: int = Form(32),
):
    global _eval_state
    with _eval_lock:
        if _eval_state["running"]:
            return {"status": "error", "message": "已有评分任务在运行"}
        _eval_state["running"] = True
        _eval_state["message"] = "正在加载模型与数据..."
        _eval_state["progress"] = 0.0

    try:
        device = config.DEVICE
        data_dir = DATASET_ROOT / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"数据集不存在: {data_dir}")

        with _eval_lock:
            _eval_state["progress"] = 0.02
            _eval_state["message"] = "加载数据集..."
        transform = get_transform()
        full_dataset = ImageFolderDataset(data_dir, transform=transform)
        classes = full_dataset.classes

        with _eval_lock:
            _eval_state["progress"] = 0.08
            _eval_state["message"] = "加载模型..."
        model, model_classes = get_model_for_key(model_key)
        if len(model_classes) != len(classes):
            raise ValueError(
                f"模型类别数({len(model_classes)})与数据集类别数({len(classes)})不一致"
            )

        val_size = max(1, int(len(full_dataset) * config.EVAL_VAL_SPLIT))
        train_size = len(full_dataset) - val_size
        _, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.EVAL_SEED),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )

        n = len(classes)
        confusion = torch.zeros(n, n, dtype=torch.long)
        correct = 0
        total = 0
        total_batches = max(1, len(val_loader))
        model.eval()

        with torch.no_grad():
            for bi, (images, labels) in enumerate(val_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                if device == "cuda":
                    with autocast("cuda", dtype=torch.float16):
                        outputs = model(images)
                else:
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    confusion[t, p] += 1
                with _eval_lock:
                    _eval_state["progress"] = round(0.1 + 0.9 * bi / total_batches, 4)
                    _eval_state["message"] = f"评分中 {bi}/{total_batches} 批"

        accuracy = correct / total if total else 0.0
        per_class = []
        macro_p = macro_r = macro_f1 = 0.0
        weighted_p = weighted_r = weighted_f1 = 0.0
        support_sum = 0

        for i in range(n):
            tp = confusion[i, i].item()
            fp = int(confusion[:, i].sum().item()) - tp
            fn = int(confusion[i, :].sum().item()) - tp
            support = int(confusion[i, :].sum().item())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)
            per_class.append({
                "class": classes[i],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            })
            macro_p += precision
            macro_r += recall
            macro_f1 += f1
            weighted_p += precision * support
            weighted_r += recall * support
            weighted_f1 += f1 * support
            support_sum += support

        macro_p /= n if n else 1
        macro_r /= n if n else 1
        macro_f1 /= n if n else 1
        weighted_p /= support_sum if support_sum else 1
        weighted_r /= support_sum if support_sum else 1
        weighted_f1 /= support_sum if support_sum else 1

        with _eval_lock:
            _eval_state["message"] = "评分完成"
            _eval_state["progress"] = 1.0

        return {
            "status": "ok",
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_p, 4),
            "weighted_recall": round(weighted_r, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
            "classes": classes,
        }
    except Exception as e:
        with _eval_lock:
            _eval_state["message"] = f"评分出错: {e}"
        return {"status": "error", "message": str(e)}
    finally:
        with _eval_lock:
            _eval_state["running"] = False


@router.get("/evaluate/status")
async def evaluate_status():
    with _eval_lock:
        return dict(_eval_state)