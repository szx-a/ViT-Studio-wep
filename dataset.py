# dataset.py - 自定义数据集
"""按子文件夹组织的图像分类数据集。

目录结构:
    data/
      ├── cat/
      │     ├── cat001.jpg
      │     └── cat002.jpg
      ├── dog/
      │     └── dog001.jpg
      ...

子文件夹名 → 类别标签（按字母排序）"""

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

        # 收集所有类别（子文件夹）
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        if not self.classes:
            raise ValueError(f"数据目录下未找到类别子文件夹: {self.root}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 收集所有图像路径
        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))
                # 大小写不敏感
                for img_path in cls_dir.glob(ext.upper()):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        print(f"[数据集] 类别数={len(self.classes)}, 样本数={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
