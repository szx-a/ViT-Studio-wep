# inference.py - CLI 推理入口
"""命令行图像分类工具。

用法:
    python inference.py <图片路径> [--top-k 5] [--model imagenet|微调模型名] [--output result.json]

示例:
    python inference.py cat.jpg
    python inference.py cat.jpg --top-k 3 --model eurosat
"""

import argparse

import config
from model import predict
from utils import load_image, print_json


def main():
    parser = argparse.ArgumentParser(description="ViT-B/16 图像分类推理")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K,
                        help=f"返回前 K 个预测结果（默认 {config.DEFAULT_TOP_K}）")
    parser.add_argument("--model", type=str, default=config.BUILTIN_MODEL_KEY,
                        help="模型 key：imagenet 或 checkpoints 下微调模型名（不带 .pth）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="结果保存为 JSON 文件（可选）")

    args = parser.parse_args()

    img = load_image(args.image)
    results = predict(img, top_k=args.top_k, model_key=args.model)
    print_json(results, output_path=args.output)


if __name__ == "__main__":
    main()