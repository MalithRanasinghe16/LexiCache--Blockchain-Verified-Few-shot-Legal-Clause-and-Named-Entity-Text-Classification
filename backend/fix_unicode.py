"""Replace all non-ASCII characters in evaluate_cuad_document_level.py with ASCII equivalents."""
path = "scripts/evaluation/evaluate_cuad_document_level.py"
content = open(path, "r", encoding="utf-8").read()

replacements = [
    ("\u2550", "="),   # ═  double horizontal
    ("\u2500", "-"),   # ─  single horizontal
    ("\u2265", ">="),  # ≥  greater-than-or-equal
    ("\u2192", "->"),  # →  right arrow
    ("\u2026", "..."), # …  ellipsis
    ("\u2248", "~"),   # ≈  approximately equal
    ("\u2717", "x"),   # ✗  ballot x
    ("\u2014", "--"),  # —  em dash
    ("\u2013", "-"),   # –  en dash
]

for old, new in replacements:
    content = content.replace(old, new)

open(path, "w", encoding="utf-8").write(content)

bad = [(i, ch) for i, ch in enumerate(content) if ord(ch) > 127]
if bad:
    for i, ch in bad[:20]:
        line_no = content[:i].count("\n") + 1
        print(f"  Still present  Line {line_no}: U+{ord(ch):04X} {repr(ch)}")
else:
    print("SUCCESS: No non-ASCII characters remain in the file.")
