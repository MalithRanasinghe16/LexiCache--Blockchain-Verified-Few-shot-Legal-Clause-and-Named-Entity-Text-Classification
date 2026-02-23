# src/prototype_demo.py
"""
LexiCache – Final Supervisor Prototype Demo
Uses finalized LexiCacheModel
"""

print("=" * 100)
print(" " * 35 + "LEXICACHE")
print(" " * 28 + "FINAL SUPERVISOR DEMO")
print("=" * 100)
print("Final Year Project | IIT × University of Westminster")
print(f"Student     : T.M.M.S. Ranasinghe (Malith)")
print(f"Supervisor  : Mr. Jihan Jeeth")
print(f"Model       : Finalized LexiCacheModel (CUAD primary)")
print(f"Performance : 89.87% Macro F1")
print("=" * 100)

from src.ml_model import LexiCacheModel

model = LexiCacheModel()

print("\n✅ Prototype ready! Enter legal text below.\n")

while True:
    try:
        text = input("Enter legal clause or contract text: ").strip()
        
        if text.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for testing LexiCache!\n")
            break
            
        if not text:
            print("→ Please enter some text.\n")
            continue

        print("🔄 Processing with final model...", end="", flush=True)

        # ─── This is where you put the line ─────────────────────────────────
        result = model.predict_cuad(text)
        # ────────────────────────────────────────────────────────────────────

        # Now show the result nicely
        print("\r" + " " * 40 + "\r", end="")  # clear the processing line
        print(f"\n{'='*80}")
        print("🧠 FINAL CUAD PREDICTION RESULT")
        print(f"{'='*80}")
        
        if isinstance(result, dict):
            print(f"Clause Type   : {result.get('clause_type', 'Unknown')}")
            print(f"Confidence    : {result.get('confidence', 0)*100:.1f}%")
            print(f"Extracted Span: {result.get('span', text[:100] + '...')}")
        elif isinstance(result, list):
            for i, (clause, span, conf) in enumerate(result, 1):
                print(f"\n{i}. Clause Type   : {clause}")
                print(f"   Extracted Span : {span}")
                print(f"   Confidence     : {conf*100:.1f}%")
        else:
            print("Result:", result)

        print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user. Goodbye!\n")
        break
    except Exception as e:
        print(f"\nError during prediction: {str(e)}\n")