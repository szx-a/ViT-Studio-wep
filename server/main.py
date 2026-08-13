# server/main.py - FastAPI 入口

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.routes import predict, dataset, train, models

app = FastAPI(title="ViT 图像识别中台", version="1.2")

app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(train.router)
app.include_router(models.router)

base = Path(__file__).resolve().parent.parent
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/datasets", StaticFiles(directory=str(base / "datasets")), name="datasets")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)