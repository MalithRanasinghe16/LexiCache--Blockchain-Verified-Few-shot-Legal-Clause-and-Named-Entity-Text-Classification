"""Compatibility entrypoint for FastAPI app."""

from src.api import main as _api_main


if __name__ != "__main__":
    # Expose the canonical module object so attribute monkeypatching on
    import sys as _sys

    _sys.modules[__name__] = _api_main
else:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
