"""Compatibility wrapper for CUAD train/test split script."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.data.split_cuad", run_name="__main__")
