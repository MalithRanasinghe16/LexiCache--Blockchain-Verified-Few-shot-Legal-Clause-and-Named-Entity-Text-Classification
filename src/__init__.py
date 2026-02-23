# src/__init__.py
"""
LexiCache Source Package

Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification
Final Year Project - Informatics Institute of Technology (IIT)
In collaboration with University of Westminster

Student: T.M.M.S. Ranasinghe (Malith)
Supervisor: Mr. Jihan Jeeth

Modules:
    - data                : normalization & dataset utilities
    - modeling            : Legal-BERT + Prototypical Network
    - experiments         : few-shot evaluation & meta-training
    - ml_model            : unified final model (CUAD primary)
    - cuad_fewshot        : CUAD-specific few-shot logic
    - deduplication       : SHA-256 + Redis cache (Goal 3)
    - prototype_demo      : interactive supervisor demo

Tagline: "One Upload = One Immutable Proof"
"""

# Optional: explicit exports (cleaner imports)
__all__ = [
    'normalize_text',           # from data.py
    'PrototypicalNetwork',      # from modeling.py
    'LexiCacheModel',           # from ml_model.py
    'CUADFewShot',              # from cuad_fewshot.py
    # 'compute_document_hash',  # from deduplication.py – add later
    # 'check_cache',            # from deduplication.py
]