# server/routes/dataset.py - 数据集管理 API

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import shutil
from fastapi import APIRouter, UploadFile, File, Form

import config

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATASET_ROOT = Path(config.BASE_DIR) / "datasets"
DATASET_ROOT.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images_in(directory: Path):
    if not directory.exists():
        return []
    return sorted(
        f.name for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


@router.get("")
async def list_datasets():
    """列出所有数据集（两级：数据集 -> 类别），兼容根目录直接放图片的旧结构。"""
    datasets = []
    if not DATASET_ROOT.exists():
        return {"datasets": datasets}

    for d in sorted(DATASET_ROOT.iterdir()):
        if not d.is_dir():
            continue

        root_images = _images_in(d)
        classes = []
        for c in sorted(d.iterdir()):
            if not c.is_dir():
                continue
            class_images = _images_in(c)
            classes.append({
                "name": c.name,
                "count": len(class_images),
                "images": class_images,
            })

        total = len(root_images) + sum(c["count"] for c in classes)
        datasets.append({
            "name": d.name,
            "count": total,
            "root_images": root_images,
            "classes": classes,
        })

    return {"datasets": datasets}


def _category_dir(dataset: str, category: str) -> Path:
    """解析类别目录。dataset 为空时兼容旧的一级结构。"""
    if dataset:
        return DATASET_ROOT / dataset / category
    return DATASET_ROOT / category


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(...),
    dataset: str = Form(""),
):
    """上传图片到指定数据集下的类别。"""
    cls_dir = _category_dir(dataset, category)
    cls_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".jpg"
    save_path = cls_dir / file.filename

    contents = await file.read()
    save_path.write_bytes(contents)

    return {
        "status": "ok",
        "dataset": dataset,
        "category": category,
        "path": str(save_path.relative_to(config.BASE_DIR)),
    }


@router.post("/category")
async def create_category(
    category: str = Form(...),
    dataset: str = Form(""),
):
    """创建新类别文件夹。"""
    cls_dir = _category_dir(dataset, category)
    cls_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "dataset": dataset, "category": category}


@router.delete("/category")
async def delete_category(
    category: str,
    dataset: str = "",
):
    """删除类别及其下所有图片。"""
    cls_dir = _category_dir(dataset, category)
    if cls_dir.exists():
        shutil.rmtree(cls_dir)
        return {"status": "ok", "dataset": dataset, "category": category}
    return {"status": "not_found", "dataset": dataset, "category": category}


@router.delete("/image")
async def delete_image(
    category: str = Form(...),
    filename: str = Form(...),
    dataset: str = Form(""),
):
    """删除单张图片。"""
    img_path = _category_dir(dataset, category) / filename
    if img_path.exists():
        img_path.unlink()
        return {"status": "ok"}
    return {"status": "not_found"}
