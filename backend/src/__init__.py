# src/__init__.py
"""
LexiCache Source Package

Few-shot Legal Clause and Named Entity Text Classification.
Final Year Project -- IIT x University of Westminster.

Modules:
    data          -- Text normalization and dataset utilities
    modeling      -- Legal-BERT encoder and Prototypical Network
    experiments   -- Few-shot evaluation and meta-training
    ml_model      -- Unified adaptive model (CUAD primary)
    fine_tune_multi -- Multi-task fine-tuning script
"""

__all__ = [
    'normalize_text',
    'PrototypicalNetwork',
    'LexiCacheModel',
]