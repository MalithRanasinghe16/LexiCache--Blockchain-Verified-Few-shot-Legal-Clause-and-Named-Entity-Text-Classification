# app/prototype_streamlit.py
import streamlit as st
import torch
import numpy as np
import fitz  # PyMuPDF – pip install pymupdf
import re
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from typing import List, Tuple

st.set_page_config(page_title="LexiCache - Clause Highlight Demo", page_icon="⚖️", layout="wide")

st.markdown("""
# LexiCache – Clause Highlighting Prototype  
**Blockchain Verified Few-shot Legal Clause Classification**

**Student:** T.M.M.S. Ranasinghe | **Supervisor:** Mr. Jihan Jeeth  
IIT × University of Westminster | February 2026  

**Model:** Meta-trained Legal-BERT | **Performance:** 89.87% macro F1 (5-way 5-shot on LEDGAR)
""")

st.info("Upload a PDF → see clauses automatically detected & highlighted by type.")

# ─── Full LEDGAR Label Names + Colors ───────────────────────────────────────
LEDGAR_LABEL_NAMES = {  # abbreviated – add more if needed
    2: "Applicable Law / Governing Law",
    8: "Confidentiality",
    13: "Governing Law",
    14: "Indemnification",
    17: "Limitation of Liability",
    24: "Payment Terms",
    # ... (use your full 100 if you want)
    99: "Miscellaneous"
}

# Simple color map (expand as needed)
CLAUSE_COLORS = {
    "Governing Law": "#ADD8E6",      # light blue
    "Confidentiality": "#90EE90",     # light green
    "Payment Terms": "#FFD700",       # gold/yellow
    "Indemnification": "#FFB6C1",     # light pink
    "Limitation of Liability": "#FFA07A",  # light salmon
    "Miscellaneous": "#D3D3D3",       # light gray
    "Unknown": "#E0E0E0"
}

# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrototypicalNetwork()
    projection = torch.nn.Linear(768, 768).to(device)
    projection.load_state_dict(torch.load("projection_head.pth", map_location=device))
    projection.eval()
    dataset = load_dataset("lex_glue", "ledgar")
    return model, projection, dataset, device


class PrototypicalNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    def forward(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_emb.cpu())
        return torch.cat(embeddings, dim=0)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = ' '.join(text.split())
    return text


def predict_single_clause(clause: str, model, projection, dataset):
    clause_norm = normalize_text(clause)
    # Quick 1-shot dummy support (for speed in highlight mode)
    # For demo we use a fixed small support set to avoid long runtime
    support_texts = [
        "This Agreement is governed by the laws of England.",
        "The parties agree to keep all information confidential.",
        "Payment is due within 30 days of invoice.",
        "Each party shall indemnify the other for breaches.",
        "Liability is limited to direct damages only."
    ]
    support_labels = [13, 8, 24, 14, 17]  # corresponding IDs

    with torch.no_grad():
        s_emb = model(support_texts, batch_size=8)
        q_emb = model([clause_norm], batch_size=1)
        s_proj = projection(s_emb.to(model.device))
        q_proj = projection(q_emb.to(model.device))

    prototypes = []
    unique = np.unique(support_labels)
    for lbl in unique:
        mask = np.array(support_labels) == lbl
        prototypes.append(s_proj[mask].mean(dim=0))
    prototypes = torch.stack(prototypes)

    dists = torch.cdist(q_proj, prototypes)
    pred_idx = torch.argmin(dists).item()
    confidence = torch.softmax(-dists, dim=1)[0, pred_idx].item()

    pred_id = unique[pred_idx]
    pred_name = LEDGAR_LABEL_NAMES.get(pred_id, f"Type {pred_id}")
    color = CLAUSE_COLORS.get(pred_name, "#E0E0E0")

    return pred_name, confidence, color


def highlight_clauses(full_text: str, model, projection, dataset):
    # Simple clause splitting (improve later with better regex/NLP)
    clauses = re.split(r'(?=\b(?:Section|Article|\d+\.|\([a-z]\)|[A-Z][a-z]+\s+Clause)\b|\n\s*\n)', full_text)
    clauses = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 20]

    highlighted = ""
    for clause in clauses:
        name, conf, color = predict_single_clause(clause, model, projection, dataset)
        highlighted += f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px;" title="Confidence: {conf:.1%}">{clause}</span> '
        highlighted += f'<small style="color:#555;">({name})</small><br>'

    return highlighted


# ─── Load once ───────────────────────────────────────────────────────────────
model, projection, dataset, device = load_components()

# ─── UI ──────────────────────────────────────────────────────────────────────
st.subheader("Upload Contract PDF")
pdf_file = st.file_uploader("Choose PDF", type="pdf")

if pdf_file is not None:
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n\n"
        doc.close()

        st.success("PDF processed!")

        tab1, tab2 = st.tabs(["Highlighted Clauses", "Raw Text"])

        with tab1:
            with st.spinner("Detecting & classifying clauses..."):
                highlighted_html = highlight_clauses(full_text, model, projection, dataset)
            st.markdown(highlighted_html, unsafe_allow_html=True)

        with tab2:
            with st.expander("Full Extracted Text"):
                st.text(full_text[:5000] + "..." if len(full_text) > 5000 else full_text)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a PDF contract to see clause-by-clause highlighting.")