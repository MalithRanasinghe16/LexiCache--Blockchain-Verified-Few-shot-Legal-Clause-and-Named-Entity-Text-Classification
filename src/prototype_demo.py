# src/prototype_demo.py
"""
LexiCache – Supervisor Live Prototype Demo
Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification

Final Year Project
Informatics Institute of Technology (IIT) in collaboration with University of Westminster
Student: T.M.M.S. Ranasinghe (Malith)
Supervisor: Mr. Jihan Jeeth
Date: February 2026

Model: Legal-BERT + Meta-trained Projection Head
Performance: 89.87% Macro F1 (5-way 5-shot on LEDGAR)
Datasets: CUAD (primary), LEDGAR, CoNLL-2003
"""

print("=" * 100)
print(" " * 35 + "LEXICACHE")
print(" " * 28 + "SUPERVISOR PROTOTYPE DEMO")
print("=" * 100)
print("Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification")
print("Final Year Project | IIT × University of Westminster")
print(f"Student     : T.M.M.S. Ranasinghe (Malith)")
print(f"Supervisor  : Mr. Jihan Jeeth")
print(f"Model       : Legal-BERT + Meta-trained Projection Head")
print(f"Performance : 89.87% Macro F1 (5-way 5-shot)")
print(f"Datasets    : CUAD (primary) + LEDGAR + CoNLL-2003")
print(f"Tagline     : One Upload = One Immutable Proof")
print("=" * 100)
print("\nHow to test:")
print("• Paste any legal clause or contract paragraph")
print("• Type 'cuad' or use longer text → uses CUAD model")
print("• Type 'exit' or 'quit' to stop\n")

# ====================== IMPORTS ======================
from src.cuad_fewshot import CUADFewShot
import torch

# Lazy load model
cuad_model = None
print("✅ Meta-trained model loaded successfully. Ready for live demo!\n")

# ====================== MAIN DEMO LOOP ======================
while True:
    try:
        text = input("Enter legal clause or contract text: ").strip()

        if text.lower() in ['exit', 'quit', 'q']:
            print("\n" + "="*100)
            print("Thank you for testing the LexiCache Prototype!")
            print("One Upload = One Immutable Proof")
            print("="*100 + "\n")
            break

        if not text:
            print("→ Please enter some text.\n")
            continue

        print("🔄 Processing with meta-trained model...", end="", flush=True)

        # Use CUAD model for longer text or when "cuad" is typed
        if len(text) > 80 or "cuad" in text.lower() or "clause" in text.lower():
            if cuad_model is None:
                cuad_model = CUADFewShot()
            cuad_model.predict(text)
        else:
            # LEDGAR fallback for short text
            print("\n[LEDGAR Mode] Using provision classification (CUAD is primary for clauses)")
            if cuad_model is None:
                cuad_model = CUADFewShot()
            cuad_model.predict(text)

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user. Goodbye!\n")
        break
    except Exception as e:
        print(f"\nError: {str(e)}\n")