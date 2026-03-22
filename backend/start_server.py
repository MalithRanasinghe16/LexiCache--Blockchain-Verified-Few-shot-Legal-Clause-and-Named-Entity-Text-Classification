"""Compatibility launcher for backend server."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.start_server", run_name="__main__")
