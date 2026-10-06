"""Vercel serverless entry point — repo root api/index.py.

Adds the repo root to sys.path so the backend package resolves,
then re-exports the FastAPI app instance.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("VERCEL", "1")

from backend.app.main import app  # noqa: F401
