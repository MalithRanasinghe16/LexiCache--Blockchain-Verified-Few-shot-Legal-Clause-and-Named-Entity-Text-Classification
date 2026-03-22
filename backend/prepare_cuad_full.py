"""Compatibility wrapper for CUAD full dataset preparation script."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.data.prepare_cuad_full", run_name="__main__")
