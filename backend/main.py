from io import BytesIO
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz  # PyMuPDF
from docx import Document
from web3 import Web3
from src.ml_model import LexiCacheModel
from src.deduplication import (
    can_user_verify,
    compute_doc_hash,
    create_verification_attempt,
    discard_document_data,
    get_cached_result,
    get_verification_history,
    get_verification_state,
    has_verification_history,
    push_history_entry,
    record_user_teach,
    register_upload,
    store_result,
)

# .env example (do not commit real secrets):
# PRIVATE_KEY=0xyour_wallet_private_key
# SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/your_project_id
load_dotenv()

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
    doc_hash: Optional[str] = None
    user_id: Optional[str] = None


class VerifyRequest(BaseModel):
    doc_hash: str
    user_id: str
    clauses: List[Dict[str, Any]]


class DiscardRequest(BaseModel):
    doc_hash: str
    user_id: Optional[str] = None

class UpdateColorRequest(BaseModel):
    clause_type: str
    color: str


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _send_sepolia_verification_tx(
    doc_hash: str,
    clause_types: List[str],
    analysis_hash: str,
) -> Dict[str, Any]:
    rpc_url = (os.getenv("SEPOLIA_RPC_URL") or "").strip()
    private_key = (os.getenv("PRIVATE_KEY") or "").strip()

    if not rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL is not set in .env")
    if not private_key:
        raise RuntimeError("PRIVATE_KEY is not set in .env")

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise RuntimeError("Unable to connect to Sepolia RPC endpoint")

    account = w3.eth.account.from_key(private_key)
    timestamp = int(time.time())

    chain_id = w3.eth.chain_id
    if chain_id != 11155111:
        raise RuntimeError(f"Connected chain_id={chain_id}, expected Sepolia (11155111)")

    onchain_payload = {
        "doc_hash": doc_hash,
        "clause_types": clause_types,
        "timestamp": timestamp,
        "analysis_hash": analysis_hash,
    }
    data_hex = w3.to_hex(text=json.dumps(onchain_payload, sort_keys=True, separators=(",", ":")))

    nonce = w3.eth.get_transaction_count(account.address, "pending")
    latest_block = w3.eth.get_block("latest")
    base_fee = int(latest_block.get("baseFeePerGas", w3.eth.gas_price))
    priority_fee = w3.to_wei(2, "gwei")
    max_fee = (base_fee * 2) + priority_fee

    tx = {
        "from": account.address,
        "to": account.address,
        "value": 0,
        "data": data_hex,
        "nonce": nonce,
        "chainId": 11155111,
        "type": 2,
        "gas": 150000,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": priority_fee,
    }

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if int(receipt.status) != 1:
        raise RuntimeError("Transaction mined but failed (status=0)")

    tx_hash_hex = tx_hash.hex()
    return {
        "tx_hash": tx_hash_hex,
        "explorer_link": f"https://sepolia.etherscan.io/tx/{tx_hash_hex}",
        "timestamp": timestamp,
        "payload": onchain_payload,
    }

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
async def upload_file(file: UploadFile = File(...), user_id: str = Form("anonymous")):
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
            docx_doc = Document(BytesIO(content))
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

        register_upload(doc_hash, user_id)
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

            unknown_count = len([
                c for c in cached_clauses
                if str(c.get("clause_type", "")) == "Unknown clause"
            ])
            verification = get_verification_state(doc_hash, user_id, unknown_count)
            history = get_verification_history(doc_hash)

            return {
                "status": "cache_hit",
                "cached_at": analyzed_at,
                "doc_hash": doc_hash,
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
                "verification": verification,
                "history": history,
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

        unknown_count = len([r for r in result if r.get("clause_type") == "Unknown clause"])
        verification = get_verification_state(doc_hash, user_id, unknown_count)
        history = get_verification_history(doc_hash)

        return {
            "status": "success",
            "doc_hash": doc_hash,
            "extracted_text": text,
            "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,  # type: ignore[index]
            "page_count": len(page_texts),
            "page_texts": page_texts,
            "result": result,
            "file_type": file_type,
            "verification": verification,
            "history": history,
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

        if request.doc_hash and request.user_id:
            record_user_teach(request.doc_hash, request.user_id)

            # Keep dedup cache in sync with newly taught results.
            cached = get_cached_result(request.doc_hash) or {}
            store_result(
                doc_hash=request.doc_hash,
                clauses=updated_results,
                page_texts=cached.get("page_texts", []),
                extracted_text=cached.get("extracted_text", request.contract_text),
                file_type=cached.get("file_type", "unknown"),
            )

        unknown_count = len([
            c for c in updated_results if c.get("clause_type") == "Unknown clause"
        ])
        verification = None
        history = []
        if request.doc_hash and request.user_id:
            verification = get_verification_state(request.doc_hash, request.user_id, unknown_count)
            history = get_verification_history(request.doc_hash)

        stats = model.get_statistics()
        
        return {
            "status": "success",
            "message": f"Successfully learned '{request.new_type_name}'",
            "updated_results": updated_results,
            "model_stats": stats,
            "verification": verification,
            "history": history,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rename operation failed: {str(e)}")


@app.post("/verify")
@app.post("/verify-document")
async def verify_document(request: VerifyRequest):
    """
    Create an immutable verification attempt for the current document analysis.

    Rules:
    - First uploader can verify any time while unknowns remain.
    - Later users can verify only after teaching at least one unknown.
    - If unknown_count is zero, verification is no longer needed.
    """
    try:
        cached = get_cached_result(request.doc_hash)
        if not cached:
            raise HTTPException(
                status_code=404,
                detail="Document analysis not found in cache. Please upload the document again.",
            )

        # Authoritative verification source: server-side cached analysis.
        # Do not trust client-provided clauses for permission checks or proof.
        clauses = cached.get("clauses", [])
        unknown_count = len([
            c for c in clauses
            if str(c.get("clause_type", "")) == "Unknown clause"
        ])

        allowed, reason = can_user_verify(request.doc_hash, request.user_id, unknown_count)
        if not allowed:
            raise HTTPException(status_code=403, detail=reason)

        clause_types = sorted({
            str(c.get("clause_type", "Unknown clause"))
            for c in clauses
            if c.get("clause_type")
        })
        analysis_hash = _sha256_json(cached)

        try:
            tx_result = _send_sepolia_verification_tx(
                doc_hash=request.doc_hash,
                clause_types=clause_types,
                analysis_hash=analysis_hash,
            )
        except Exception as tx_exc:
            raise HTTPException(
                status_code=502,
                detail=f"Blockchain transaction failed: {str(tx_exc)}",
            )

        history_entry = {
            "doc_hash": request.doc_hash,
            "verified_by": request.user_id,
            "clause_types": clause_types,
            "timestamp": tx_result["timestamp"],
            "analysis_hash": analysis_hash,
            "tx_hash": tx_result["tx_hash"],
            "explorer_link": tx_result["explorer_link"],
        }
        if not push_history_entry(request.doc_hash, history_entry):
            raise HTTPException(
                status_code=500,
                detail="Blockchain transaction succeeded but Redis history write failed",
            )

        attempt = create_verification_attempt(
            doc_hash=request.doc_hash,
            user_id=request.user_id,
            clauses=clauses,
            unknown_count=unknown_count,
            tx_hash=tx_result["tx_hash"],
            blockchain_link=tx_result["explorer_link"],
            snapshot_hash=analysis_hash,
        )
        if attempt is None:
            raise HTTPException(status_code=500, detail="Failed to persist verification attempt")

        history = get_verification_history(request.doc_hash)
        verification = get_verification_state(request.doc_hash, request.user_id, unknown_count)
        return {
            "status": "verified",
            "tx_hash": tx_result["tx_hash"],
            "explorer_link": tx_result["explorer_link"],
            "message": "Document verified on Sepolia testnet.",
            "record": attempt,
            "history": history,
            "verification": verification,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.get("/document-history/{doc_hash}")
async def document_history(doc_hash: str):
    """Return verification history for a document hash."""
    try:
        history = get_verification_history(doc_hash)
        return {
            "status": "success",
            "doc_hash": doc_hash,
            "history": history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History lookup failed: {str(e)}")


@app.post("/discard-document")
async def discard_document(request: DiscardRequest):
    """
    Discard cached document data when a user leaves before verification.

    Safety rule: if any verification attempt already exists, keep data/history.
    """
    try:
        if has_verification_history(request.doc_hash):
            return {
                "status": "kept",
                "message": "Document already verified; history preserved.",
            }

        deleted = discard_document_data(request.doc_hash)
        if deleted:
            return {
                "status": "discarded",
                "message": "Unverified document data discarded.",
            }
        return {
            "status": "skipped",
            "message": "Redis unavailable or document not found.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discard failed: {str(e)}")

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