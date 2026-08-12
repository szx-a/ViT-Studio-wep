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


@router.get("")
async def list_datasets():
    """列出所有数据集：类别名 + 图片数。"""
    if not DATASET_ROOT.exists():
        return {"datasets": []}

    categories = []
    for d in sorted(DATASET_ROOT.iterdir()):
        if d.is_dir():
            images = [f.name for f in d.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
            categories.append({
                "name": d.name,
                "count": len(images),
                "images": sorted(images),
            })
    return {"datasets": categories}


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(...),
):
    """上传图片到指定类别。"""
    cls_dir = DATASET_ROOT / category
    cls_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".jpg"
    save_path = cls_dir / file.filename

    contents = await file.read()
    save_path.write_bytes(contents)

    return {"status": "ok", "path": str(save_path.relative_to(config.BASE_DIR))}


@router.post("/category")
async def create_category(category: str = Form(...)):
    """创建新类别文件夹。"""
    cls_dir = DATASET_ROOT / category
    cls_dir.mkdir(parents=True, exist_ok=True)
    return {"status": "ok", "category": category}


@router.delete("/category/{category}")
async def delete_category(category: str):
    """删除类别及其下所有图片。"""
    cls_dir = DATASET_ROOT / category
    if cls_dir.exists():
        shutil.rmtree(cls_dir)
        return {"status": "ok", "category": category}
    return {"status": "not_found", "category": category}


@router.delete("/image")
async def delete_image(category: str = Form(...), filename: str = Form(...)):
    """删除单张图片。"""
    img_path = DATASET_ROOT / category / filename
    if img_path.exists():
        img_path.unlink()
        return {"status": "ok"}
    return {"status": "not_found"}
