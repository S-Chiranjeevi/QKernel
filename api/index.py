"""Vercel serverless entry point (repo root api/index.py).

Vercel's Python runtime executes this file from the repo root, so
sys.path already contains the repo root. We just import the FastAPI app.
"""
import sys
import os
from pathlib import Path

# Make sure the repo root is on sys.path so `backend.app.main` resolves.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set VERCEL so main.py redirects writes to /tmp
os.environ.setdefault("VERCEL", "1")

from backend.app.main import app  # noqa: F401
