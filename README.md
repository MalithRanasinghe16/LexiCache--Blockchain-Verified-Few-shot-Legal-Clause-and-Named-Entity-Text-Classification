# LexiCache

## Overview

LexiCache is a legal document analysis tool that uses few-shot meta-learning to classify contract clauses. It combines a FastAPI backend with a Next.js frontend to provide an interactive document analysis experience.

**Key Features:**

- Upload PDF or DOCX contracts and get clause-by-clause classification
- Hybrid classification using keyword matching and prototypical networks (Legal-BERT)
- Adaptive online learning: teach the system new clause types through the UI
- Supports 40+ CUAD clause types plus custom user-defined types
- Visual clause highlighting with color-coded overlays on the document

## Architecture

- **Backend:** Python 3.10, FastAPI, PyTorch, Transformers (Legal-BERT), PyMuPDF, python-docx
- **Frontend:** Next.js (TypeScript), React-PDF, Tailwind CSS
- **ML Model:** Prototypical networks with a meta-trained projection head on CUAD/LEDGAR/CoNLL-2003

## Setup

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- pip and npm package managers

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Running the Application

### Start the backend (port 8000)

```bash
cd backend
python start_server.py
```

### Start the frontend (port 3000)

```bash
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## Project Structure

```
LexiCache/
  backend/
    main.py                 # FastAPI server with all API endpoints
    start_server.py         # Uvicorn startup script
    requirements.txt        # Python dependencies
    src/
      __init__.py           # Package exports
      ml_model.py           # Adaptive meta-learning model (segmentation, classification, online learning)
      modeling.py           # PrototypicalNetwork with Legal-BERT encoder
      data.py               # Dataset acquisition and text normalization
      experiments.py        # Few-shot experiment runner and meta-training
      fine_tune_multi.py    # Multi-task fine-tuning on CUAD + LEDGAR + CoNLL-2003
    examples/
      online_learning_demo.py  # Demo script for adaptive learning API
    tests/
      test_pipeline.py      # Unit tests for segmentation, keywords, merge logic
    data/                   # Dataset metadata
  frontend/
    src/
      app/
        page.tsx            # Main page with upload, analysis, and clause interaction
        layout.tsx          # Root layout
        types.ts            # TypeScript type definitions
        components/
          AppHeader.tsx     # Application header
          UploadForm.tsx    # File upload form
          DocumentViewer.tsx # PDF/DOCX document display
          PdfViewer.tsx     # PDF rendering with clause highlight overlays
          DocxViewer.tsx    # DOCX text rendering
          ResultsSidebar.tsx # Clause results panel
          ClauseList.tsx    # Individual clause cards
          FilterPanel.tsx   # Clause type and confidence filters
          SearchBar.tsx     # Text search in document
          ColorLegend.tsx   # Clause color legend with customization
          RenameModal.tsx   # Modal to teach the system new clause types
```

## Datasets

All datasets are publicly available and properly licensed:

1. **CUAD** - Contract Understanding Atticus Dataset (CC BY 4.0)
2. **LEDGAR** - Legal provisions from LexGLUE (CC BY-SA 4.0)
3. **CoNLL-2003** - Named Entity Recognition (Research use)

## License

[Add your license information here]
