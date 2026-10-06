# Vercel serverless entry point.
# Vercel looks for a callable named `app` in api/index.py.
from backend.app.main import app  # noqa: F401 — re-exported for Vercel
