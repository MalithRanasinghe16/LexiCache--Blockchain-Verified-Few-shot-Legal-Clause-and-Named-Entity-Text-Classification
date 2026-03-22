"""Compatibility wrapper for CUAD evaluation script."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.evaluation.evaluate_cuad_test", run_name="__main__")
