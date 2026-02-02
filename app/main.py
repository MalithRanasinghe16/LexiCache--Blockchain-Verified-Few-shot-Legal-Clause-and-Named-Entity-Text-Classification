import streamlit as st
import fitz  # PyMuPDF for PDF
from docx import Document  # For DOCX

# Title with project name and tagline
st.title("LexiCache: Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification")
st.subheader('Tagline: "One Upload = One Immutable Proof"')

# File uploader for PDF/DOC/DOCX
uploaded_file = st.file_uploader("Upload a Legal Contract (PDF, DOC, or DOCX)", type=["pdf", "doc", "docx"])

if uploaded_file is not None:
    try:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            # Use PyMuPDF for PDF text extraction
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"  # Preserve page breaks
            doc.close()
        elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            # Use python-docx for DOC/DOCX
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error("Unsupported file type. Please upload PDF, DOC, or DOCX.")
            text = ""

        if text:
            st.success("Document uploaded and text extracted successfully!")
            st.subheader("Extracted Text Preview (First 1000 characters):")
            st.text(text[:1000])  # Limit preview to avoid overwhelming the UI
            # TODO: Pass 'text' to next steps (normalization, deduplication, etc.)

    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")

# Security note in code: Never process or store sensitive data without anonymization (per ethics section)