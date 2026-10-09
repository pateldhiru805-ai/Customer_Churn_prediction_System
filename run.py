"""Customer Churn Prediction System - Master Entry Point.

Run this script directly from the project root:
    python run.py
"""

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODEL_PATH

def ensure_model_exists():
    """Verify production model pipeline exists; train if not present."""
    if not MODEL_PATH.exists():
        print("[INFO] Model binary not found. Training model pipeline first...")
        from src.train import train_and_evaluate_all
        train_and_evaluate_all()
        print("[INFO] Model trained successfully.")

def open_browser(port):
    """Wait 1.5 seconds and open the web browser automatically."""
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[INFO] Opening application in browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[NOTE] Could not open browser automatically: {e}")

def main():
    print("=" * 65)
    print("      Customer Churn Prediction System - AI Workbench")
    print("=" * 65)

    ensure_model_exists()

    from app.app import app
    port = int(os.getenv("PORT", 5000))

    # Auto-open browser in background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    print(f"\n[READY] Server starting at: http://127.0.0.1:{port}")
    print("[NOTE]  Press CTRL+C to stop the server anytime.\n")

    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    main()
