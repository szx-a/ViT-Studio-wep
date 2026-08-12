import ssl, urllib.request, zipfile, shutil, os
from pathlib import Path

# 禁用 SSL 验证（仅针对这个特定下载）
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
dest = Path(r"D:\PythonProject  cv\vit\vit_src\datasets")
dest.mkdir(parents=True, exist_ok=True)
zip_path = dest / "EuroSAT.zip"

print("下载 EuroSAT (2GB, 请耐心等待)...")
try:
    with urllib.request.urlopen(url, context=ctx, timeout=30) as resp:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded / 1024 / 1024
                    print(f"\r  进度: {mb:.0f}MB / {total/1024/1024:.0f}MB ({pct:.1f}%)", end="")
    print("\n下载完成, 解压中...")
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest / "_temp")
    
    # 找到 2750 目录
    src = dest / "_temp" / "2750"
    eurosat = dest / "eurosat"
    if src.exists():
        for cls_dir in src.iterdir():
            if cls_dir.is_dir():
                shutil.copytree(cls_dir, eurosat / cls_dir.name, dirs_exist_ok=True)
    else:
        # 搜一下
        for root, dirs, files in os.walk(str(dest / "_temp")):
            for d in dirs:
                if d == "2750":
                    src = Path(root) / d
                    for cls_dir in src.iterdir():
                        if cls_dir.is_dir():
                            shutil.copytree(cls_dir, eurosat / cls_dir.name, dirs_exist_ok=True)
                    break
    
    shutil.rmtree(dest / "_temp", ignore_errors=True)
    zip_path.unlink()
    
    cats = sorted([d.name for d in eurosat.iterdir() if d.is_dir()])
    total = sum(1 for f in eurosat.rglob("*") if f.suffix.lower() in (".jpg",".jpeg",".png"))
    print(f"完成! {total} 张图, {len(cats)} 类:")
    for c in cats:
        count = len(list((eurosat / c).glob("*")))
        print(f"  {c}: {count} 张")
        
except Exception as e:
    print(f"下载失败: {e}")
    print("可能需要手动下载: https://madm.dfki.de/files/sentinel/EuroSAT.zip")
    print(f"解压后把 2750/ 下所有文件夹复制到: {dest / 'eurosat'}")
