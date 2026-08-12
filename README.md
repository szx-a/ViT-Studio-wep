# ViT 图像识别中台 — 项目完整文档

> **项目路径**: `D:\PythonProject  cv\vit\`  
> **Python 环境**: `D:\PythonProject  cv\.venv`  
> **硬件**: RTX 5060 Laptop GPU (8GB), CUDA 12.8  
> **最后更新**: 2026-08-12

---

## 总览：两个计划的合流

本项目由两个独立计划逐步合并而成：

```
计划一：CLI 图像识别工具          计划二：Web 中台
    │                                │
    ├─ ViT-B/16 模型加载              ├─ FastAPI 后端
    ├─ CLI 推理入口                   ├─ 三 Tab 前端（识别/数据集/训练）
    ├─ CLI 训练脚本                   ├─ 后台线程训练
    ├─ ImageNet-1K 标签               ├─ 实时进度 + Chart.js
    ├─ CPU → GPU 切换                ├─ 中英双语展示
    ├─ fp16 混合精度训练               ├─ 数据集管理界面
    └─ vit-inference Skill            └─ 双击启动器
                    ↘                ↙
                   最终形态：完整 Web 中台
                   CLI 后端 + Web 前端共用同一套 model.py / config.py
```

---

## 计划一：CLI 图像识别工具

### 设计目标
- 模型可独立使用（本地推理/训练）
- Agent 可调用（vit-inference skill）
- 支持微调训练

### 初始架构

```
vit_src/
├── config.py          # 全局配置中心
├── model.py           # ViT 模型加载 + 推理（缓存复用）
├── inference.py       # CLI: python inference.py <图片> --top-k 5
├── train.py           # CLI: python train.py --data_dir ./data --epochs 10
├── dataset.py         # ImageFolder 数据集类
├── utils.py           # 工具函数
└── labels/
    └── imagenet_label.txt   # ImageNet-1K 英文标签
```

### 迭代记录

| # | 改进 | 内容 |
|:--|------|------|
| 1 | 模型选型 | ViT-B/16 (8600万参数), ImageNet-1K 1000 类 |
| 2 | GPU 切换 | CPU 版 PyTorch → CUDA 12.8 (RTX 5060)，驱动 573.22 兼容 cu128 |
| 3 | 版本降级 | 从 2.13.0+cu130 降至 2.11.0+cu128（驱动兼容） |
| 4 | fp16 训练 | `torch.amp.autocast("cuda")` + `GradScaler("cuda")` |
| 5 | 标签 | 生成 ImageNet-1K 英文标签（从 torchvision weights meta） |
| 6 | Skill | 创建 `vit-inference` skill，Agent 可调用 CLI 推理 |

---

## 计划二：Web 中台

### 设计目标
- 浏览器访问，三 Tab 单页应用
- 训练支持后台线程 + 实时进度
- 中英双语结果

### 架构

```
vit_src/
└── server/
    ├── main.py                  # FastAPI 入口 + 静态文件挂载
    ├── routes/
    │   ├── predict.py           # POST /api/predict (图片上传→识别)
    │   ├── dataset.py           # 数据集 CRUD API
    │   └── train.py             # 训练 API (后台线程 + 状态共享)
    └── static/
        ├── index.html           # 三 Tab 单页
        ├── style.css            # 深色玻璃态主题
        └── app.js               # 交互逻辑 + Chart.js
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/predict` | 上传图片 → Top-K 分类 |
| GET | `/api/datasets` | 列出所有数据集 |
| POST | `/api/datasets/upload` | 上传到指定类别 |
| POST | `/api/datasets/category` | 创建类别 |
| DELETE | `/api/datasets/category/{name}` | 删除类别 |
| DELETE | `/api/datasets/image` | 删除单张图片 |
| POST | `/api/train/start` | 启动后台训练 |
| GET | `/api/train/status` | 实时进度 (epoch/batch/loss/acc) |
| GET | `/api/train/history` | 训练历史 |
| POST | `/api/train/stop` | 停止训练 |

### 前端三模块

| Tab | 功能 | 技术细节 |
|-----|------|---------|
| 🖼️ 图片识别 | 拖拽上传 → 分类 → 中英双语 + 置信度条形图 + 识别进度动画 | Chart.js 横向 bar |
| 📦 数据集 | 创建类别 / 批量上传 / 图片网格查看 / 删除 | 图片载入 `/datasets/{类别}/{文件名}` |
| 🏋️ 训练中心 | 选数据集 → 填参数 → 开始 → Epoch+Batch 双进度条 + 实时 Loss/Acc 折线图 | `setInterval` 1秒轮询 + Chart.js line |

### 训练后台机制

```
前端 POST /api/train/start
        ↓
FastAPI 启动 threading.Thread
        ↓
_train_thread() 每 epoch 更新 training_state (共享 dict)
        ↓
前端 setInterval 1秒轮询 GET /api/train/status
        ↓
状态字段: running, epoch, total_epochs, batch, total_batches,
          train_loss/acc, val_loss/acc, best_acc, message, history[]
```

---

## 合并后：共用层

两个计划共用同一套核心，没有重复代码：

```
        ┌──────────────────┐
        │    config.py      │  ← 参数中控
        ├──────────────────┤
        │    model.py       │  ← 模型加载/推理
        ├──────────────────┤
        │    utils.py       │  ← 标签/图像/JSON
        ├──────────────────┤
        │    dataset.py     │  ← 数据集类
        └──────┬───────────┘
               │
    ┌──────────┴──────────┐
    │                     │
 CLI 入口              Web 入口
inference.py          server/main.py
train.py              server/routes/*
```

---

## 改进全景时间线

| # | 所属计划 | 改进 |
|:--|:--:|------|
| 1 | 一 | 基础 ViT 项目搭建 (config/model/utils/inference/dataset/train) |
| 2 | 一 | ImageNet-1K 标签生成 |
| 3 | 一 | CPU → GPU 切换 (CUDA 12.8, PyTorch cu128) |
| 4 | 一 | PyTorch 版本降级 2.13→2.11 (驱动兼容) |
| 5 | 一 | fp16 混合精度训练 (autocast + GradScaler) |
| 6 | 一 | vit-inference Agent Skill |
| 7 | 二 | 安装 FastAPI + uvicorn + python-multipart |
| 8 | 二 | 创建 server/routes/predict.py |
| 9 | 二 | 创建 server/routes/dataset.py |
| 10 | 二 | 创建 server/routes/train.py (后台线程+状态共享) |
| 11 | 二 | 创建 server/main.py + 静态文件挂载 |
| 12 | 二 | 创建前端 index.html + style.css + app.js |
| 13 | 合并 | 中英双语标签 (1000类), utils.py 支持, model.py 返回 `class_name_zh` |
| 14 | 合并 | 20 条缺失翻译补齐 (class 480-499) |
| 15 | 二 | Epoch + Batch 双进度条 |
| 16 | 二 | 识别进度动画 (indeterminate bar) |
| 17 | 合并 | 双击启动器 (launch.py + bat + 桌面快捷方式) |
| 18 | 合并 | PyCharm 旧进程端口冲突修复 (清缓存, 关旧服务) |
| 19 | 合并 | 双语标签全量扫描 (1000条, 0缺失) |
| 20 | 合并 | 虚拟因子关联写入 AGENTS.md |

---

## 启动方式

| 方式 | 操作 |
|------|------|
| 桌面快捷方式 | 双击 `ViT中台` |
| bat 文件 | 双击 `vit_src/启动中台.bat` |
| Python | `python vit_src/launch.py` |
| PyCharm | 右键 `launch.py` → Run |

---

## 训练数据格式

```
datasets/
├── 猫/  (cat001.jpg, cat002.png, ...)
├── 狗/  (dog001.jpg, ...)
└── 鸟/  (bird001.jpg, ...)
```

支持格式: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## 关联项目：虚拟因子

虚拟因子 (Virtual Factor) 是一个物理第一性的建造交互系统，将数字实体（"因子"）投影到物理沙盘，通过触觉/手势操控，内嵌实时物理引擎。

- **核心理念**: 物理第一性——几何从物理状态派生，不是物理属性的容器
- **四层架构**: 投影层 / 操控层 / 物理层 / 交互层
- **AGI 定位**: 不是工具，而是 AGI 理解物理世界的"母语环境"
- **原型**: `F:\UNLAY\virtual_factor\`
- **论文**: `D:\文档\vf\`

**ViT 结合点**: 校准质量检测、因子状态快照分类、无人机地形预处理

---

## 依赖清单

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
timm==1.0.28
fastapi==0.141.1
uvicorn==0.52.1
python-multipart==0.0.32
aiofiles==25.1.0
```
