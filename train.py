# train.py - 微调训练脚本（支持 fp16 混合精度）
"""在自定义数据集上微调 ViT 模型。

用法:
    python train.py --data_dir ./my_data --epochs 10 --lr 1e-4 --batch_size 32

目录结构:
    my_data/
      ├── cat/    (cat001.jpg, ...)
      ├── dog/    (dog001.jpg, ...)
      ...

默认启用 fp16 混合精度（GPU），可通过 --no_amp 关闭。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast

import config
from model import load_model, get_transform
from dataset import ImageFolderDataset


def parse_args():
    parser = argparse.ArgumentParser(description="ViT 微调训练")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="训练数据根目录（子文件夹=类别）")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS,
                        help=f"训练轮数（默认 {config.DEFAULT_EPOCHS}）")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE,
                        help=f"批次大小（默认 {config.DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE,
                        help=f"学习率（默认 {config.DEFAULT_LEARNING_RATE}）")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="验证集比例（默认 0.2）")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="冻结 backbone，仅训练分类头（默认开启）")
    parser.add_argument("--unfreeze", action="store_true",
                        help="解冻 backbone 进行全模型微调")
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用 fp16 混合精度（默认在 GPU 上启用）")
    parser.add_argument("--output", type=str, default=None,
                        help="模型保存路径（默认 vit_src/checkpoints/best.pth）")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
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

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()

    # ---- 设备检测 ----
    device = config.DEVICE
    use_amp = (device == "cuda") and (not args.no_amp) and config.USE_AMP
    if use_amp:
        print("[训练] fp16 混合精度已启用")
    else:
        print("[训练] 使用 fp32 全精度")

    # ---- 数据集 ----
    transform = get_transform()
    full_dataset = ImageFolderDataset(args.data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"[训练] 检测到 {num_classes} 个类别: {full_dataset.classes}")

    # 训练/验证拆分
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=config.DEFAULT_NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=config.DEFAULT_NUM_WORKERS, pin_memory=(device == "cuda"),
    )

    # ---- 模型 ----
    model = load_model(pretrained=True)

    # 替换分类头
    if hasattr(model, "head"):
        in_features = model.head.in_features if hasattr(model.head, "in_features") else model.head.weight.shape[1]
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("模型结构不匹配：找不到 head 层，请检查模型名称")

    model.to(device)

    # 冻结 backbone（可选）
    if args.freeze_backbone and not args.unfreeze:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[训练] Backbone 已冻结，仅训练分类头（可训参数: {trainable:,} / {total:,}）")
    else:
        print("[训练] 全模型微调")

    # ---- 优化器 & 混合精度 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- 训练循环 ----
    best_acc = 0.0
    save_path = args.output or str(config.CHECKPOINT_DIR / "best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
        )

        # 估算显存占用
        if device == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            + (f" | GPU 峰值显存: {mem_mb:.0f} MB" if device == "cuda" else "")
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "classes": full_dataset.classes,
                    "val_acc": val_acc,
                },
                save_path,
            )
            print(f"  -> 保存最佳模型: {save_path}")

    print(f"\n[完成] 最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()

