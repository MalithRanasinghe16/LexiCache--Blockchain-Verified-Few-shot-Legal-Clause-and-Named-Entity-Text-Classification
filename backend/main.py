from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from docx import Document
from src.ml_model import LexiCacheModel
from src.deduplication import compute_doc_hash, get_cached_result, store_result

app = FastAPI(
    title="LexiCache API",
    description="Few-shot Legal Clause and NER Classification",
    version="0.1.0"
)

# CORS: allow the Next.js frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = LexiCacheModel()

class TextRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    clause_text: str
    correct_label: str
    original_prediction: Optional[str] = None
    confidence: Optional[float] = None

class RenameUnknownRequest(BaseModel):
    contract_text: str
    unknown_span: str
    new_type_name: str
    color: Optional[str] = None

class UpdateColorRequest(BaseModel):
    clause_type: str
    color: str

@app.get("/")
async def root():
    """Return API information and current model statistics."""
    stats = model.get_statistics()
    return {
        "name": "LexiCache API",
        "version": "0.1.0",
        "description": "Adaptive Few-shot Legal Clause Classification",
        "model_stats": stats
    }

@app.post("/predict-text")
async def predict_text(request: TextRequest):
    """Classify clause types from plain text input."""
    try:
        result = model.predict_cuad(request.text)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a PDF or DOCX file, extract text, and classify clauses with page tracking.

    Redis deduplication: the document is fingerprinted by SHA-256 of its
    normalised text.  If an identical document was analysed before (within
    90 days) the cached result is returned immediately without re-running
    the model.  If Redis is unavailable the endpoint degrades gracefully
    and always runs the full model pipeline.
    """
    if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF, DOC, DOCX allowed")

    content = await file.read()

    try:
        # ── Step 1: Extract raw text from the uploaded file ──────────────────
        page_texts = []
        text: str = ""
        is_pdf = file.filename.lower().endswith('.pdf')

        if is_pdf:
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
            docx_doc = Document(content)
            text = "\n".join([p.text for p in docx_doc.paragraphs])
            page_texts.append({
                'page': 1,
                'text': text,
                'start_char': 0,
                'end_char': len(text)
            })

        file_type = "pdf" if is_pdf else "docx"

        # ── Step 2: Compute content hash & check Redis cache ─────────────────
        doc_hash = compute_doc_hash(text)
        print(f"[upload-file] doc_hash={doc_hash[:16]}... file={file.filename!r} ({len(text)} chars)")

        cached = get_cached_result(doc_hash)
        if cached is not None:
            # Cache HIT — rebuild the full response from stored data and return
            # immediately.  The stored page_texts and extracted_text are used
            # verbatim so the frontend receives an identical payload.
            cached_clauses = cached.get("clauses", [])
            cached_page_texts = cached.get("page_texts", page_texts)
            cached_extracted_text = cached.get("extracted_text", text)
            cached_file_type = cached.get("file_type", file_type)
            analyzed_at = cached.get("analyzed_at", "")

            return {
                "status": "cache_hit",
                "cached_at": analyzed_at,
                "extracted_text": cached_extracted_text,
                "extracted_text_preview": (
                    cached_extracted_text[:500] + "..."
                    if len(cached_extracted_text) > 500
                    else cached_extracted_text
                ),
                "page_count": len(cached_page_texts),
                "page_texts": cached_page_texts,
                "result": cached_clauses,
                "file_type": cached_file_type,
            }

        # ── Step 3: Cache MISS — run the full model pipeline ─────────────────
        # IMPORTANT: raw text is passed to predict_cuad without normalisation.
        # Normalisation strips legal capitalisation and punctuation that the
        # keyword heuristics depend on (e.g. "INDEMNIFICATION", "GOVERNING LAW").
        result = model.predict_cuad(text)

        # Map each clause to its page number using char-offset ranges
        for clause in result:
            clause_start = clause.get('start_idx', 0)
            for page_info in page_texts:
                if page_info['start_char'] <= clause_start < page_info['end_char']:
                    clause['page_number'] = page_info['page']
                    break
            if 'page_number' not in clause:
                clause['page_number'] = 1

        # ── Step 4: Persist result to Redis (non-fatal on failure) ───────────
        store_result(
            doc_hash=doc_hash,
            clauses=result,
            page_texts=page_texts,
            extracted_text=text,
            file_type=file_type,
        )

        return {
            "status": "success",
            "extracted_text": text,
            "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,  # type: ignore[index]
            "page_count": len(page_texts),
            "page_texts": page_texts,
            "result": result,
            "file_type": file_type,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/statistics")
async def get_statistics():
    """Return model statistics and learning progress."""
    try:
        stats = model.get_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback so the model can learn from corrections in real-time."""
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
    """Return all known clause types (CUAD standard + user-learned)."""
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
    """Return all clause types with their assigned display colors."""
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
    """Rename an unknown clause to a user-defined type, teach the model, and re-classify."""
    try:
        success = model.learn_from_feedback(
            clause_text=request.unknown_span,
            correct_label=request.new_type_name,
            color=request.color
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to learn new clause type")
        
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
    """Update the display color for a specific clause type."""
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