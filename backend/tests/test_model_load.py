# test_model_load.py
from src.ml_model import LexiCacheModel

print("Testing final model load...")
model = LexiCacheModel()
print("Model loaded successfully!")
print(f"Projection device: {model.projection.weight.device}")