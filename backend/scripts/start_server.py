# !/usr/bin/env python
"""Start the LexiCache backend server with hot-reload."""
import uvicorn

if __name__ == "__main__":
    print("="*60)
    print(" Starting LexiCache Backend Server")
    print("="*60)

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
