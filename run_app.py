"""
One-Click System Launcher for E2H-ViT Diagnostic Workspace
Starts the FastAPI backend on http://127.0.0.1:8000 and automatically
opens the interactive clinical interface in the default web browser.
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def open_browser(url: str, delay: float = 1.5):
    """Opens browser after server starts."""
    time.sleep(delay)
    print(f"\n[+] Launching interactive clinical web workspace at {url}...")
    webbrowser.open(url)


def main():
    print("=" * 72)
    print("  E2H-ViT: Efficient Explainable Hybrid Vision Transformer")
    print("  Clinical AI Diagnostic Working System Launcher")
    print("=" * 72)

    # 1. Check & pre-warm PyTorch
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"[*] PyTorch Version : {torch.__version__} ({device} mode)")

    from e2h_vit import build_e2h_vit_nano
    print("[*] Pre-warming E2H-ViT neural network...")
    model = build_e2h_vit_nano()
    params = model.count_parameters()
    print(f"[+] Loaded E2H-ViT-Nano ({params:,d} parameters ready)")

    # 2. Host & Port
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}/#diagnostic-system"

    # 3. Open browser in background thread
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # 4. Start Uvicorn Server
    print(f"\n[+] Starting FastAPI REST server at http://{host}:{port}...")
    print("    - REST API Docs: http://127.0.0.1:8000/docs")
    print("    - Press Ctrl+C to terminate the working system.")
    print("=" * 72)

    uvicorn.run("server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
