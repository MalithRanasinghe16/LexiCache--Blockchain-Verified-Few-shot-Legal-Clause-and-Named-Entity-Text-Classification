# backend/main.py
from typing import Optional
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

class FeedbackRequest(BaseModel):
    clause_text: str
    correct_label: str
    original_prediction: Optional[str] = None
    confidence: Optional[float] = None

class RenameUnknownRequest(BaseModel):
    contract_text: str  # Full contract text for re-classification
    unknown_span: str   # The span text that was marked as Unknown
    new_type_name: str  # User's name for this clause type
    color: Optional[str] = None   # Optional: user-chosen color (hex)

class UpdateColorRequest(BaseModel):
    clause_type: str
    color: str  # Hex color code

@app.get("/")
async def root():
    """API root - system info"""
    stats = model.get_statistics()
    return {
        "name": "LexiCache API",
        "version": "0.1.0",
        "description": "Adaptive Few-shot Legal Clause Classification",
        "model_stats": stats
    }

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
    """Upload PDF/DOCX → extract text → predict with page tracking"""
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX allowed")

    content = await file.read()

    try:
        page_texts = []  # Track text per page for position mapping
        text: str = ""  # Will hold extracted text
        
        if file.filename.lower().endswith('.pdf'):
            doc = fitz.open(stream=content, filetype="pdf")
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                page_texts.append({
                    'page': page_num,
                    'text': page_text,
                    'start_char': len(text),
                    'end_char': len(text) + len(page_text)
                })
                text += page_text + "\n"
            doc.close()
        else:
            doc = Document(content)
            text = "\n".join([p.text for p in doc.paragraphs])
            # For DOCX, treat as single page
            page_texts.append({
                'page': 1,
                'text': text,
                'start_char': 0,
                'end_char': len(text)
            })

        # Analyze full document (remove 4000 char limit for production)
        result = model.predict_cuad(text)
        
        # Add page numbers to each clause
        for clause in result:
            clause_start = clause.get('start_idx', 0)
            # Find which page this clause is on
            for page_info in page_texts:
                if page_info['start_char'] <= clause_start < page_info['end_char']:
                    clause['page_number'] = page_info['page']
                    break
            if 'page_number' not in clause:
                clause['page_number'] = 1  # Default to page 1
        
        return {
            "status": "success", 
            "extracted_text": text,  # Full text for highlighting
            "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,  # type: ignore[index]
            "page_count": len(page_texts),
            "page_texts": page_texts,  # For frontend text position mapping
            "result": result,
            "file_type": "pdf" if file.filename.lower().endswith('.pdf') else "docx"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/statistics")
async def get_statistics():
    """Get model statistics and learning progress"""
    try:
        stats = model.get_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback to improve the model.
    The model learns from corrections in real-time.
    
    Example:
    {
        "clause_text": "This agreement shall be governed by the laws of California",
        "correct_label": "Governing Law",
        "original_prediction": "General Provision",
        "confidence": 0.45
    }
    """
    try:
        success = model.learn_from_feedback(
            clause_text=request.clause_text,
            correct_label=request.correct_label
        )
        
        if success:
            stats = model.get_statistics()
            return {
                "status": "success",
                "message": f"Learned '{request.correct_label}' successfully",
                "model_stats": stats
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to learn from feedback")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

@app.get("/clause-types")
async def get_clause_types():
    """Get all known clause types (CUAD standard + learned)"""
    try:
        from src.ml_model import CLAUSE_KEYWORDS_WEIGHTED
        stats = model.get_statistics()
        
        return {
            "status": "success",
            "cuad_types": list(CLAUSE_KEYWORDS_WEIGHTED.keys()),
            "learned_types": list(stats['label_distribution'].keys()),
            "total_known_types": len(set(list(CLAUSE_KEYWORDS_WEIGHTED.keys()) + list(stats['label_distribution'].keys())))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/clause-types-with-colors")
async def get_clause_types_with_colors():
    """Get all clause types with their assigned colors for legend"""
    try:
        types_colors = model.get_all_clause_types_with_colors()
        return {
            "status": "success",
            "clause_types": types_colors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rename-unknown")
async def rename_unknown_clause(request: RenameUnknownRequest):
    """
    Rename an 'Unknown clause' to a user-defined type and re-classify.
    This teaches the model the new clause type.
    
    Example:
    {
        "contract_text": "...full contract text...",
        "unknown_span": "The parties agree to escrow...",
        "new_type_name": "Escrow Provision",
        "color": "#FF5733"
    }
    """
    try:
        # Teach the model
        success = model.learn_from_feedback(
            clause_text=request.unknown_span,
            correct_label=request.new_type_name,
            color=request.color
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to learn new clause type")
        
        # Re-classify the contract with new knowledge
        updated_results = model.predict_cuad(request.contract_text)
        stats = model.get_statistics()
        
        return {
            "status": "success",
            "message": f"Successfully learned '{request.new_type_name}'",
            "updated_results": updated_results,
            "model_stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rename operation failed: {str(e)}")

@app.post("/update-color")
async def update_clause_color(request: UpdateColorRequest):
    """
    Update the color for a specific clause type.
    
    Example:
    {
        "clause_type": "Termination",
        "color": "#FF5733"
    }
    """
    try:
        success = model.update_clause_color(request.clause_type, request.color)
        
        if success:
            return {
                "status": "success",
                "message": f"Updated color for '{request.clause_type}'"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update color")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Color update failed: {str(e)}")