# server/routes/train.py - 训练 API（后台线程 + 实时进度）

import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

from fastapi import APIRouter, Form
import config
from model import load_model, get_transform
from dataset import ImageFolderDataset

router = APIRouter(prefix="/api/train", tags=["train"])

DATASET_ROOT = Path(config.BASE_DIR) / "datasets"

# ---- 全局训练状态 ----
training_state: dict = {
    "running": False,
    "stop_requested": False,
    "epoch": 0,
    "total_epochs": 0,
    "batch": 0,
    "total_batches": 0,
    "train_loss": 0.0,
    "train_acc": 0.0,
    "val_loss": 0.0,
    "val_acc": 0.0,
    "best_acc": 0.0,
    "message": "空闲",
    "num_classes": 0,
    "classes": [],
    "history": [],
}
_lock = threading.Lock()


def _run_training(data_dir: str, epochs: int, batch_size: int,
                  lr: float, freeze_backbone: bool):
    global training_state

    with _lock:
        training_state["running"] = True
        training_state["stop_requested"] = False
        training_state["epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["batch"] = 0
        training_state["total_batches"] = 0
        training_state["message"] = "正在加载数据..."
        training_state["history"] = []
        training_state["best_acc"] = 0.0

    try:
        device = config.DEVICE
        use_amp = (device == "cuda") and config.USE_AMP

        transform = get_transform()
        full_dataset = ImageFolderDataset(data_dir, transform=transform)
        num_classes = len(full_dataset.classes)

        with _lock:
            training_state["num_classes"] = num_classes
            training_state["classes"] = full_dataset.classes
            training_state["message"] = f"数据集: {num_classes} 类, {len(full_dataset)} 张"

        val_size = max(1, int(len(full_dataset) * 0.2))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )

        model = load_model(pretrained=True)
        if hasattr(model, "head"):
            in_features = (model.head.in_features
                           if hasattr(model.head, "in_features")
                           else model.head.weight.shape[1])
            model.head = nn.Linear(in_features, num_classes)
        model.to(device)

        if freeze_backbone:
            for name, param in model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        )
        scaler = GradScaler("cuda", enabled=use_amp)

        with _lock:
            training_state["message"] = "训练开始..."

        for epoch in range(1, epochs + 1):
            with _lock:
                if training_state["stop_requested"]:
                    training_state["message"] = "训练已手动停止"
                    break
                training_state["batch"] = 0
                training_state["total_batches"] = len(train_loader)

            # ---- 训练 ----
            model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for images, labels in train_loader:
                with _lock:
                    if training_state["stop_requested"]:
                        break
                    training_state["batch"] += 1

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with autocast("cuda", dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                t_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                t_correct += (preds == labels).sum().item()
                t_total += labels.size(0)

            train_loss = t_loss / t_total if t_total else 0
            train_acc = t_correct / t_total if t_total else 0

            # ---- 验证 ----
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with autocast("cuda", dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    v_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)

            val_loss = v_loss / v_total if v_total else 0
            val_acc = v_correct / v_total if v_total else 0

            if val_acc > training_state["best_acc"]:
                save_path = config.CHECKPOINT_DIR / "best_web.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "classes": full_dataset.classes,
                    "val_acc": val_acc,
                }, save_path)

            history_entry = {
                "epoch": epoch, "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            }
            with _lock:
                training_state["epoch"] = epoch
                training_state["train_loss"] = round(train_loss, 4)
                training_state["train_acc"] = round(train_acc, 4)
                training_state["val_loss"] = round(val_loss, 4)
                training_state["val_acc"] = round(val_acc, 4)
                training_state["best_acc"] = max(training_state["best_acc"], val_acc)
                training_state["history"].append(history_entry)
                training_state["message"] = (
                    f"Epoch {epoch}/{epochs} | "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc: {train_acc:.4f}/{val_acc:.4f}"
                )

        with _lock:
            if not training_state["stop_requested"]:
                training_state["message"] = f"训练完成！最佳准确率: {training_state['best_acc']:.4f}"

    except Exception as e:
        with _lock:
            training_state["message"] = f"训练出错: {e}"
    finally:
        with _lock:
            training_state["running"] = False


@router.get("/status")
async def get_status():
    with _lock:
        return dict(training_state)


@router.get("/history")
async def get_history():
    with _lock:
        return {"history": list(training_state["history"])}


@router.post("/start")
async def start_training(
    dataset_name: str = Form(...),
    epochs: int = Form(10),
    batch_size: int = Form(32),
    lr: float = Form(1e-4),
    freeze_backbone: bool = Form(True),
):
    with _lock:
        if training_state["running"]:
            return {"status": "error", "message": "已有训练正在运行"}

    data_dir = DATASET_ROOT / dataset_name
    if not data_dir.exists() or not any(data_dir.iterdir()):
        return {"status": "error", "message": f"数据集不存在: {data_dir}"}

    thread = threading.Thread(
        target=_run_training,
        args=(str(data_dir), epochs, batch_size, lr, freeze_backbone),
        daemon=True,
    )
    thread.start()
    return {"status": "ok", "message": "训练已启动"}


@router.post("/stop")
async def stop_training():
    with _lock:
        if not training_state["running"]:
            return {"status": "error", "message": "当前没有训练在运行"}
        training_state["stop_requested"] = True
        return {"status": "ok", "message": "已发送停止信号"}
