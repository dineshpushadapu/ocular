"""
MediScan Entry Point Mirror.
Delegates execution to root app.py for portability across working directory invocations.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import main if "main" in globals() else None

if __name__ == "__main__":
    import os
    os.system(f"streamlit run {ROOT_DIR / 'app.py'}")
