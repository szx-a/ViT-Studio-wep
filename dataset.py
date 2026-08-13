# dataset.py - 自定义数据集
"""按子文件夹组织的图像分类数据集（两级目录：数据集/类别/图片）。

目录结构:
    data/
      ├── cat/
      │     ├── cat001.jpg
      │     └── cat002.jpg
      ├── dog/
      │     └── dog001.jpg
      ...

子文件夹名 -> 类别标签（按字母排序）；空类别文件夹会被自动忽略。
"""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """ImageFolder 风格的分类数据集。"""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.classes = []
        self.class_to_idx = {}
        self.samples: list[tuple[Path, int]] = []

        for cls_dir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            images = [p for p in cls_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in image_exts]
            if not images:
                continue
            idx = len(self.classes)
            self.classes.append(cls_dir.name)
            self.class_to_idx[cls_dir.name] = idx
            for img_path in images:
                self.samples.append((img_path, idx))

        if not self.classes:
            raise ValueError(f"数据目录下未找到类别子文件夹: {self.root}")

        print(f"[数据集] 类别数={len(self.classes)}, 样本数={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label