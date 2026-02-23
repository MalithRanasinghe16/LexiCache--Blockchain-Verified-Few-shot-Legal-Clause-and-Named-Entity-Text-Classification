# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from docx import Document
from src.ml_model import LexiCacheModel

app = FastAPI(
    title="LexiCache API",
    description="Blockchain Verified Few-shot Legal Clause & NER Classification",
    version="0.1.0"
)

# Allow Next.js frontend (localhost:3000) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # update to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = LexiCacheModel()  # loads your final_projection_head.pth

class TextRequest(BaseModel):
    text: str

@app.post("/predict-text")
async def predict_text(request: TextRequest):
    """Predict clause types from pasted text"""
    try:
        result = model.predict_cuad(request.text)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload PDF/DOCX → extract text → predict"""
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX allowed")

    content = await file.read()

    try:
        if file.filename.lower().endswith('.pdf'):
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        else:
            doc = Document(content)
            text = "\n".join([p.text for p in doc.paragraphs])

        result = model.predict_cuad(text[:4000])  # limit length for demo
        return {"status": "success", "extracted_text_preview": text[:500] + "...", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}