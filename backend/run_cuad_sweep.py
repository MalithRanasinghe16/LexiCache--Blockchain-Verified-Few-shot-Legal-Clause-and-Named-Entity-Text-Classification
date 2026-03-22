"""Compatibility wrapper for CUAD sweep script."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.evaluation.run_cuad_sweep", run_name="__main__")
