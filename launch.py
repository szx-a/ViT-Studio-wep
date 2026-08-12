"""ViT 中台启动器 —— 双击启动服务 + 自动打开浏览器。"""
import sys
import socket
from pathlib import Path

# 确保能导入项目模块
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import os
os.system("title ViT 图像识别中台")  # 控制台标题


def is_port_in_use(port: int) -> bool:
    """检查端口是否已被占用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def open_browser():
    """延迟打开浏览器（等服务器就绪）。"""
    import webbrowser
    import threading
    def _open():
        import time
        time.sleep(1.5)  # 等 uvicorn 完全启动
        webbrowser.open("http://localhost:8000")
    threading.Thread(target=_open, daemon=True).start()


def main():
    PORT = 8000

    if is_port_in_use(PORT):
        print(f"⚠ 端口 {PORT} 已被占用，跳过启动，直接打开浏览器...")
        open_browser()
        import time
        time.sleep(3)
        return

    print(f"🚀 正在启动 ViT 中台服务...")
    print(f"   浏览器将自动打开 http://localhost:{PORT}")
    print(f"   关闭此窗口即可停止服务")
    print("-" * 50)

    open_browser()

    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="127.0.0.1",
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
