# src/check_datasets.py
from datasets import load_dataset

print("Checking dataset status...\n")

for name, config in [
    ("cuad", None),
    ("lex_glue", "ledgar"),
    ("conll2003", None)
]:
    print(f"→ Loading {name} {'(' + config + ')' if config else ''}...")
    try:
        ds = load_dataset(name, config, trust_remote_code=True) if config else load_dataset(name, trust_remote_code=True)
        print(f"   Status: LOADED SUCCESSFULLY")
        print(f"   Splits: {list(ds.keys())}")
        print(f"   Train size: {len(ds.get('train', [])):,} examples")
        if 'validation' in ds:
            print(f"   Validation size: {len(ds['validation']):,} examples")
        if 'test' in ds:
            print(f"   Test size: {len(ds['test']):,} examples")
        print(f"   First few columns: {list(ds['train'].column_names)[:6]} ...\n")
    except Exception as e:
        print(f"   FAILED: {str(e)}\n")


print("\n=== SAMPLE EXAMPLES ===\n")

# LEDGAR example
ledgar = load_dataset("lex_glue", "ledgar")
print("LEDGAR sample (provision text → label):")
print("Text:", ledgar['train'][0]['text'][:150], "...")
print("Label:", ledgar['train'][0]['label'])
print()

# CUAD example
# cuad = load_dataset("cuad")
# print("CUAD sample (contract context + question + answer spans):")
# print("Context (short):", cuad['train'][0]['context'][:150], "...")
# print("Question:", cuad['train'][0]['question'])
# print("Answer spans:", cuad['train'][0]['answers'])
# print()

# CoNLL-2003 example
# conll = load_dataset("conll2003")
# print("CoNLL-2003 sample (token + NER tags):")
# print("Tokens:", conll['train'][0]['tokens'][:10])
# print("NER tags:", conll['train'][0]['ner_tags'][:10])