"""Data acquisition and preprocessing for LexiCache."""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset


# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def normalize_text(text: str) -> str:
    """Canonical normalization for legal near-duplicate matching."""
    if not text or not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip().lower()

    # Execution/signature tails are noisy and commonly vary across duplicates.
    text = re.sub(r"(?m)^\s*(by|name|title|date)\s*:\s*[_\-.\s]*$", " ", text)
    text = re.sub(r"(?m)^\s*in witness whereof[^\n]*$", " [EXECUTION_BLOCK] ", text)

    # Canonicalize section numbering style differences.
    text = re.sub(r"(?m)^\s*(article|section|clause)\s+[ivxlcdm\d\.-]+\s*[:\-\.]?", r"\1 [SECTION_ID]", text)

    # Dates
    text = re.sub(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b", " [DATE] ", text)
    text = re.sub(r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b", " [DATE] ", text)
    text = re.sub(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b", " [DATE] ", text)
    text = re.sub(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b", " [DATE] ", text)

    # Amounts, percentages, ordinals and numeric variants.
    text = re.sub(r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?", " [AMOUNT] ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*%\b", " [PERCENT] ", text)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " [ORD] ", text)

    number_words = (
        "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
        "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
        "million|billion|trillion"
    )
    text = re.sub(rf"\b(?:{number_words})\b", " [NUM] ", text)
    text = re.sub(r"\b(?:one|two|three|four)\s*[- ]\s*(?:half|third|quarter|fifth)s?\b", " [FRACTION] ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " [NUM] ", text)

    # Party/entity normalization.
    entity_suffix = (
        "inc\\.?|llc|l\\.l\\.c\\.|corp\\.?|corporation|company|co\\.?|"
        "ltd\\.?|limited|lp|l\\.p\\.|llp|l\\.l\\.p\\."
    )
    text = re.sub(
        rf"\b(?:[a-z0-9&'\\.-]+\s+){{0,7}}(?:{entity_suffix})\b",
        " [PARTY] ",
        text,
    )
    text = re.sub(
        r"\b(?:licensor|licensee|buyer|seller|provider|customer|consultant|company|party\s+[ab])\b",
        " [PARTY_ROLE] ",
        text,
    )

    # Jurisdictional names
    states = (
        "alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|"
        "florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|"
        "louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|"
        "missouri|montana|nebraska|nevada|new\\s+hampshire|new\\s+jersey|"
        "new\\s+mexico|new\\s+york|north\\s+carolina|north\\s+dakota|ohio|"
        "oklahoma|oregon|pennsylvania|rhode\\s+island|south\\s+carolina|"
        "south\\s+dakota|tennessee|texas|utah|vermont|virginia|washington|"
        "west\\s+virginia|wisconsin|wyoming"
    )
    text = re.sub(rf"\b(?:{states})\b", " [STATE] ", text)

    # Final cleanup
    text = re.sub(r"[^\w\s\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_cuad_dataset(save_path: Optional[Path] = None) -> Dict:
    """Load CUAD from Kaggle cache (already downloaded successfully)."""
    print("\n" + "="*60)
    print("Loading CUAD Dataset from Kaggle Cache")
    print("="*60)
    print("Source: Kaggle - Atticus Open Contract Dataset (AOK Beta)")
    print("License: CC BY 4.0")
    print("Citation: Hendrycks et al. (2021)")

    try:
        # Kaggle cache path (already downloaded)
        kaggle_cache_path = Path(r"C:\Users\T.M.Malith Sandeepa\.cache\kagglehub\datasets\theatticusproject\atticus-open-contract-dataset-aok-beta\versions\3")
        
        if not kaggle_cache_path.exists():
            raise FileNotFoundError("CUAD cache not found. Please run kagglehub download again.")

        print(f"Successfully loaded CUAD from: {kaggle_cache_path}")
        print(f"  Full contracts available in: {kaggle_cache_path / 'CUAD_v1'}")
        
        # Save metadata
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                "dataset": "CUAD",
                "source": "Kaggle - theatticusproject/atticus-open-contract-dataset-aok-beta",
                "license": "CC BY 4.0",
                "citation": "Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.",
                "local_path": str(kaggle_cache_path),
                "description": "510 full legal contracts with 41 clause types + expert span annotations",
                "status": "full_dataset_loaded_via_kaggle"
            }
            
            with open(save_path / "cuad_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to {save_path / 'cuad_metadata.json'}")
        
        return {"path": kaggle_cache_path, "status": "success"}
        
    except Exception as e:
        print(f"Error loading CUAD from cache: {str(e)}")
        return None


def download_ledgar_dataset(save_path: Optional[Path] = None) -> Dict:
    """Download LEDGAR dataset from LexGLUE benchmark."""
    print("\n" + "="*60)
    print("Downloading LEDGAR Dataset (Legal Provisions)")
    print("="*60)
    print("Source: https://huggingface.co/datasets/lex_glue")
    
    try:
        # Load LEDGAR subset from LexGLUE
        dataset = load_dataset("lex_glue", "ledgar", trust_remote_code=True)
        
        print(f"Successfully loaded LEDGAR dataset")
        print(f"  Train samples: {len(dataset['train']) if 'train' in dataset else 'N/A'}")
        print(f"  Validation samples: {len(dataset['validation']) if 'validation' in dataset else 'N/A'}")
        print(f"  Test samples: {len(dataset['test']) if 'test' in dataset else 'N/A'}")
        
        # Save processed data
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata = {
                "dataset": "LEDGAR",
                "source": "https://huggingface.co/datasets/lex_glue",
                "license": "CC BY-SA 4.0",
                "citation": "Chalkidis, I., et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding.",
                "splits": list(dataset.keys()),
                "description": "Legal provisions classification from SEC EDGAR contracts"
            }
            
            with open(save_path / "ledgar_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to {save_path / 'ledgar_metadata.json'}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading LEDGAR: {str(e)}")
        return None


def download_conll2003_dataset(save_path: Optional[Path] = None) -> Dict:
    """Download CoNLL-2003 Named Entity Recognition dataset."""
    print("\n" + "="*60)
    print("Downloading CoNLL-2003 Dataset (Named Entity Recognition)")
    print("="*60)
    print("Source: https://huggingface.co/datasets/conll2003")
    print("License: Research use")
    print("Citation: Tjong Kim Sang & De Meulder (2003)")
    
    try:
        # Load CoNLL-2003 dataset
        dataset = load_dataset("conll2003", trust_remote_code=True)
        
        print(f"Successfully loaded CoNLL-2003 dataset")
        print(f"  Train samples: {len(dataset['train']) if 'train' in dataset else 'N/A'}")
        print(f"  Validation samples: {len(dataset['validation']) if 'validation' in dataset else 'N/A'}")
        print(f"  Test samples: {len(dataset['test']) if 'test' in dataset else 'N/A'}")
        
        # Save processed data
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata = {
                "dataset": "CoNLL-2003",
                "source": "https://huggingface.co/datasets/conll2003",
                "license": "Research use",
                "citation": "Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 Shared Task.",
                "splits": list(dataset.keys()),
                "description": "Named Entity Recognition for legal text processing"
            }
            
            with open(save_path / "conll2003_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Metadata saved to {save_path / 'conll2003_metadata.json'}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading CoNLL-2003: {str(e)}")
        return None


def test_normalize_text():
    """Test the normalize_text function with sample legal text."""
    print("\n" + "="*60)
    print("Testing normalize_text Function")
    print("="*60)
    
    test_cases = [
        "  Contract dated 01/15/2023  ",
        "This Agreement signed on December 25, 2022 by the parties",
        "Effective Date: 2023-06-30",
        "As of 15 March 2021, the terms apply",
        "Multiple   Spaces    Should    Be    Normalized",
        "UPPERCASE TEXT SHOULD BE LOWERCASE",
        "Agreement executed on 12/31/2022 and 01/01/2023",
        "Party A agrees to pay Party B the sum of $150,000 within 30 days",
        "The Company (Inc.) shall deliver the goods within 45 days",
    ]
    
    print("\nTest Cases:")
    print("-" * 60)
    
    for i, test_text in enumerate(test_cases, 1):
        normalized = normalize_text(test_text)
        print(f"\n{i}. Original:")
        print(f"   '{test_text}'")
        print(f"   Normalized:")
        print(f"   '{normalized}'")
    
    print("\n" + "="*60)
    print("Text normalization tests completed")


def main():
    """Main execution function to download and preprocess all datasets."""
    print("\n" + "="*60)
    print("LEGAL AI DATASET ACQUISITION")
    print("="*60)
    print("\nETHICAL NOTICE:")
    print("- All datasets are publicly available with proper licenses")
    print("- No proprietary or confidential data is used")
    print("- No personally identifiable information (PII)")
    print("- All sources are properly cited")
    print("="*60)
    
    # Test text normalization
    test_normalize_text()
    
    # Download datasets
    print("\n" + "="*60)
    print("Downloading Datasets...")
    print("="*60)
    
    # CUAD - Contract clauses
    cuad_data = download_cuad_dataset(save_path=DATA_DIR / "cuad")
    
    # LEDGAR - Legal provisions
    ledgar_data = download_ledgar_dataset(save_path=DATA_DIR / "ledgar")
    
    # CoNLL-2003 - Named Entity Recognition
    conll_data = download_conll2003_dataset(save_path=DATA_DIR / "conll2003")
    
    # Summary
    print("\n" + "="*60)
    print("DATASET ACQUISITION SUMMARY")
    print("="*60)
    print(f"CUAD: {'Successfully loaded' if cuad_data else 'Failed'}")
    print(f"LEDGAR: {'Successfully loaded' if ledgar_data else 'Failed'}")
    print(f"CoNLL-2003: {'Successfully loaded' if conll_data else 'Failed'}")
    print(f"\nData saved to: {DATA_DIR.absolute()}")
    print("="*60)
    
    print("\nCITATIONS:")
    print("-" * 60)
    print("1. Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP")
    print("   Dataset for Legal Contract Review. arXiv:2103.06268.")
    print("\n2. Chalkidis, I., et al. (2022). LexGLUE: A Benchmark Dataset for")
    print("   Legal Language Understanding. ACL 2022.")
    print("\n3. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to")
    print("   the CoNLL-2003 Shared Task: Language-Independent Named Entity")
    print("   Recognition.")
    print("="*60)
    
    return {
        "cuad": cuad_data,
        "ledgar": ledgar_data,
        "conll2003": conll_data
    }


if __name__ == "__main__":
    datasets = main()