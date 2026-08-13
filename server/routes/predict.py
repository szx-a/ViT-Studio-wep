# server/routes/predict.py - 图片识别 API

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io

import config
from model import predict as model_predict

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    model_key: str = Form(config.BUILTIN_MODEL_KEY),
):
    """上传图片，返回 Top-K 分类结果。"""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model_predict(image, top_k=top_k, model_key=model_key)
    return {"filename": file.filename, "model_key": model_key, "results": results}