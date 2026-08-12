# server/routes/predict.py - 图片识别 API

import sys
from pathlib import Path

# 确保 vit_src 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io

from model import predict as model_predict

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):
    """上传图片，返回 Top-K 分类结果。"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model_predict(image, top_k=top_k)
    return {"filename": file.filename, "results": results}
