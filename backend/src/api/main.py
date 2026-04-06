from io import BytesIO
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
import urllib.request
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz  # PyMuPDF
from docx import Document
from src.ml_model import LexiCacheModel
from src.deduplication import (
    add_pending_teach,
    can_user_verify,
    clear_pending_teaches_for_user,
    compute_doc_fingerprints,
    compute_doc_hash,
    create_verification_attempt,
    discard_document_data,
    get_cached_result,
    get_changed_fields_meta,
    get_pending_teaches,
    get_verification_history,
    get_verification_state,
    has_verification_history,
    has_open_verification_cycle,
    push_history_entry,
    record_user_teach,
    register_upload,
    rollback_open_cycle_data,
    seed_verification_baseline,
    should_discard_on_leave,
    store_changed_fields_meta,
    store_result,
)
from src.history_store import (
    append_verification_attempt as append_mongo_verification_attempt,
    get_verification_history as get_mongo_verification_history,
)

# .env example (do not commit real secrets):
# PRIVATE_KEY=0xyour_wallet_private_key
# SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/your_project_id
# CONTRACT_ADDRESS=0xYourDeployedContractAddress
load_dotenv()

# ABI for LexiCacheVerifier.sol — keeps the backend decoupled from the build artefact
LEXICACHE_VERIFIER_ABI = [
    {
        "inputs": [
            {"internalType": "string",   "name": "docHash",      "type": "string"},
            {"internalType": "string[]", "name": "clauseTypes",  "type": "string[]"},
            {"internalType": "uint256",  "name": "timestamp",    "type": "uint256"},
            {"internalType": "string",   "name": "analysisHash", "type": "string"},
            {"internalType": "string",   "name": "cid",          "type": "string"},
        ],
        "name": "storeVerification",
        "outputs": [{"internalType": "uint256", "name": "recordId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "analysisHash", "type": "string"}],
        "name": "isLogged",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "recordId", "type": "uint256"}],
        "name": "getVerification",
        "outputs": [
            {"internalType": "string",   "name": "docHash",      "type": "string"},
            {"internalType": "string[]", "name": "clauseTypes",  "type": "string[]"},
            {"internalType": "uint256",  "name": "timestamp",    "type": "uint256"},
            {"internalType": "string",   "name": "analysisHash", "type": "string"},
            {"internalType": "string",   "name": "cid",          "type": "string"},
            {"internalType": "address",  "name": "verifier",     "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalRecords",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "recordId",     "type": "uint256"},
            {"indexed": False, "internalType": "string",  "name": "docHash",      "type": "string"},
            {"indexed": True,  "internalType": "address", "name": "verifier",     "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",    "type": "uint256"},
            {"indexed": False, "internalType": "string",  "name": "analysisHash", "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "cid",          "type": "string"},
        ],
        "name": "VerificationStored",
        "type": "event",
    },
]

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

model = LexiCacheModel(use_multilabel=True, kw_weight=0.35, model_weight=0.65)

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


class VerifyGeoLocation(BaseModel):
    latitude: float
    longitude: float
    accuracy_m: Optional[float] = None
    captured_at: Optional[str] = None


class VerifyRequest(BaseModel):
    doc_hash: str
    user_id: str
    clauses: List[Dict[str, Any]]
    geolocation: Optional[VerifyGeoLocation] = None


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


def _normalize_span_key(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _candidate_span_keys(clause: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for field in ("span", "span_exact", "span_display"):
        value = str(clause.get(field, "")).strip()
        if not value:
            continue
        normalized = _normalize_span_key(value)
        if normalized and normalized not in keys:
            keys.append(normalized)
    return keys


def _apply_pending_teaches_to_results(
    results: List[Dict[str, Any]],
    pending_teaches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply staged teach labels to matching spans without committing model weights."""
    span_to_label: Dict[str, str] = {}
    for teach in pending_teaches:
        span = str(teach.get("span", "")).strip()
        label = str(teach.get("label", "")).strip()
        if span and label:
            span_to_label[_normalize_span_key(span)] = label

    if not span_to_label:
        return results

    patched: List[Dict[str, Any]] = []
    for clause in results:
        entry = dict(clause)
        forced_label = None
        for span_key in _candidate_span_keys(entry):
            forced_label = span_to_label.get(span_key)
            if forced_label:
                break
        if forced_label:
            entry["clause_type"] = forced_label
            try:
                existing_conf = float(entry.get("confidence", 0.0))
            except Exception:
                existing_conf = 0.0
            entry["confidence"] = max(existing_conf, 0.95)
            entry["source"] = "pending_feedback"
            entry["is_staged"] = True
            entry["needs_review"] = False
        patched.append(entry)

    return patched


def _compute_raw_hash(text: str) -> str:
    """SHA-256 of the raw (un-normalised) document text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_changed_fields(old_text: str, new_text: str) -> List[str]:
    """
    Return which normalisation placeholder types differ between the two raw texts.
    Used to record what changed in a template-variant re-upload.
    """
    import re
    changed: List[str] = []

    date_pat = re.compile(
        r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b'
        r'|\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b'
        r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        r'|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
        re.IGNORECASE,
    )
    if set(date_pat.findall(old_text.lower())) != set(date_pat.findall(new_text.lower())):
        changed.append("DATE")

    entity_suffix = r'(?:inc\.?|llc|l\.l\.c\.|corp\.?|corporation|company|co\.?|ltd\.?|limited|lp|l\.p\.|llp|l\.l\.p\.)'
    party_pat = re.compile(rf'\b(?:[a-z0-9&\'\\.-]+\s+){{0,7}}(?:{entity_suffix})\b', re.IGNORECASE)
    if set(party_pat.findall(old_text.lower())) != set(party_pat.findall(new_text.lower())):
        changed.append("PARTY")

    amount_pat = re.compile(r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?')
    if set(amount_pat.findall(old_text)) != set(amount_pat.findall(new_text)):
        changed.append("AMOUNT")

    pct_pat = re.compile(r'\b\d+(?:\.\d+)?\s*%\b')
    if set(pct_pat.findall(old_text)) != set(pct_pat.findall(new_text)):
        changed.append("PERCENT")

    return changed


def _remap_clause_offsets(
    clauses: List[Dict[str, Any]],
    new_text: str,
    new_page_texts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Try to find each cached clause span in the new document text.

    - If found: update start_idx / end_idx and page_number.
    - If not found (span contains changed text like a date or party name):
      clear offsets to -1 so the frontend knows the position is approximate.
    """
    remapped: List[Dict[str, Any]] = []
    for clause in clauses:
        c = dict(clause)
        span = c.get("span", "")
        if span:
            idx = new_text.find(span)
            if idx >= 0:
                c["start_idx"] = idx
                c["end_idx"] = idx + len(span)
                c["display_start_idx"] = idx
                c["display_end_idx"] = idx + len(span)
                for page_info in new_page_texts:
                    if page_info["start_char"] <= idx < page_info["end_char"]:
                        c["page_number"] = page_info["page"]
                        break
            else:
                c["start_idx"] = -1
                c["end_idx"] = -1
                c["display_start_idx"] = -1
                c["display_end_idx"] = -1
        remapped.append(c)
    return remapped


def _pin_to_ipfs(
    doc_hash: str,
    analysis_hash: str,
    clause_types: List[str],
    clause_count: int,
    unknown_count: int,
    verified_at: str,
) -> str:
    """
    Pin a privacy-preserving analysis summary to IPFS via Pinata.

    Only metadata is pinned — no document text, no PII.
    Returns the IPFS CID (Content Identifier) on success, or empty
    string if Pinata is unavailable (non-fatal — verification continues).

    The CID creates a two-part proof:
      - analysisHash (on-chain)  → proves WHAT was found (tamper-proof)
      - CID (on-chain + IPFS)    → proves the full analysis is retrievable
    """
    pinata_jwt = (os.getenv("PINATA_JWT") or "").strip()
    if not pinata_jwt:
        print("[IPFS] PINATA_JWT not set — skipping IPFS pin.")
        return ""

    # Build the analysis summary — no raw text, no PII
    payload = {
        "pinataContent": {
            "doc_hash":     doc_hash,
            "analysis_hash": analysis_hash,
            "clause_types": clause_types,
            "clause_count": clause_count,
            "unknown_count": unknown_count,
            "verified_at":  verified_at,
            "system":       "LexiCache v0.1.0",
        },
        "pinataMetadata": {
            "name": f"lexicache-{doc_hash[:16]}",
        },
    }

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        "https://api.pinata.cloud/pinning/pinJSONToIPFS",
        data    = body,
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {pinata_jwt}",
        },
        method = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            cid = result.get("IpfsHash", "")
            print(f"[IPFS] Pinned analysis → CID: {cid}")
            return cid
    except Exception as exc:
        print(f"[IPFS] Pin failed (non-fatal): {exc}")
        return ""


def _send_sepolia_verification_tx(
    doc_hash: str,
    clause_types: List[str],
    geo_hash: Optional[str] = None,
    cid: str = "",
) -> Dict[str, Any]:
    try:
        from web3 import Web3
    except Exception as exc:
        raise RuntimeError(
            "web3 runtime dependency is unavailable in this Python environment. "
            "Use a Python 3.10/3.11 environment with compatible web3 dependencies."
        ) from exc

    rpc_url = (os.getenv("SEPOLIA_RPC_URL") or "").strip()
    private_key = (os.getenv("PRIVATE_KEY") or "").strip()
    contract_address = (
        os.getenv("CONTRACT_ADDRESS") or "0x9B29820FEc9B0497b91175205C454FB06c576777"
    ).strip()

    if not rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL is not set in .env")
    if not private_key:
        raise RuntimeError("PRIVATE_KEY is not set in .env")

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        raise RuntimeError("Unable to connect to Sepolia RPC endpoint")

    account = w3.eth.account.from_key(private_key)
    chain_id = w3.eth.chain_id
    if chain_id != 11155111:
        raise RuntimeError(f"Connected chain_id={chain_id}, expected Sepolia (11155111)")

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=LEXICACHE_VERIFIER_ABI,
    )

    # Build analysis hash: deterministic digest of doc_hash + sorted clause types + geo
    payload_seed = doc_hash + json.dumps(clause_types, sort_keys=True, ensure_ascii=False)
    if geo_hash:
        payload_seed += geo_hash
    analysis_hash = hashlib.sha256(payload_seed.encode("utf-8")).hexdigest()
    timestamp = int(time.time())

    nonce = w3.eth.get_transaction_count(account.address)

    # ABI-encode and send to the deployed LexiCacheVerifier contract.
    # The contract's idempotency guard will revert if analysisHash was already logged.
    tx = contract.functions.storeVerification(
        doc_hash,
        clause_types,
        timestamp,
        analysis_hash,
        cid,
    ).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "chainId": 11155111,
        "gasPrice": w3.eth.gas_price,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if int(receipt.status) != 1:
        raise RuntimeError("Transaction mined but failed (status=0)")

    tx_hash_hex = receipt.transactionHash.hex()
    return {
        "tx_hash": tx_hash_hex,
        "explorer_link": f"https://sepolia.etherscan.io/tx/{tx_hash_hex}",
    }


def _get_effective_verification_history(doc_hash: str) -> List[Dict[str, Any]]:
    """Prefer Redis history; fall back to Mongo durable history when needed."""
    redis_history = get_verification_history(doc_hash)
    if redis_history:
        return redis_history
    return get_mongo_verification_history(doc_hash)


def _pending_teaches_for_user(
    doc_hash: str,
    user_id: str,
) -> List[Dict[str, Any]]:
    user = (user_id or "anonymous").strip() or "anonymous"
    pending_teaches = get_pending_teaches(doc_hash)
    return [
        teach
        for teach in pending_teaches
        if isinstance(teach, dict) and str(teach.get("user_id", "")).strip() == user
    ]


def _build_geo_audit_payload(
    geolocation: Optional[VerifyGeoLocation],
) -> Dict[str, Optional[str]]:
    if geolocation is None:
        return {
            "geo_hash": None,
            "geo_summary": None,
        }

    lat = float(geolocation.latitude)
    lon = float(geolocation.longitude)
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail="Invalid geolocation coordinates")

    accuracy = None
    if geolocation.accuracy_m is not None:
        accuracy = max(0.0, float(geolocation.accuracy_m))

    captured_at = str(geolocation.captured_at or "").strip()
    geo_payload: Dict[str, Any] = {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "accuracy_m": round(accuracy, 1) if accuracy is not None else None,
        "captured_at": captured_at or None,
    }
    geo_hash = _sha256_json(geo_payload)

    lat_display = round(lat, 3)
    lon_display = round(lon, 3)
    if accuracy is not None:
        geo_summary = f"{lat_display}, {lon_display} (+/- {round(accuracy)}m)"
    else:
        geo_summary = f"{lat_display}, {lon_display}"

    return {
        "geo_hash": geo_hash,
        "geo_summary": geo_summary,
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
        # Step 1: extract raw text
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

        # Step 2: compute hash and check cache
        fingerprints = compute_doc_fingerprints(text)
        doc_hash = str(fingerprints.get("primary_hash", "")) or compute_doc_hash(text)
        raw_hash = _compute_raw_hash(text)
        print(f"[upload-file] doc_hash={doc_hash[:16]}... file={file.filename!r} ({len(text)} chars)")

        register_upload(doc_hash, user_id)
        durable_history = _get_effective_verification_history(doc_hash)
        seed_verification_baseline(doc_hash, user_id, durable_history)
        cached = get_cached_result(fingerprints)
        if cached is not None:
            cached_clauses = cached.get("clauses", [])
            cached_page_texts = cached.get("page_texts", page_texts)
            cached_extracted_text = cached.get("extracted_text", text)
            cached_file_type = cached.get("file_type", file_type)
            analyzed_at = cached.get("analyzed_at", "")
            cached_raw_hash = cached.get("raw_hash")

            # Check exact match vs template variant
            is_exact = (cached_raw_hash == raw_hash) or (
                cached_raw_hash is None and cached_extracted_text == text
            )

            if is_exact:
                # Exact hit: return cached payload
                cache_match_type = "exact"
                response_clauses = cached_clauses
                response_page_texts = cached_page_texts
                response_extracted_text = cached_extracted_text
                changed_fields: List[str] = []
            else:
                # Template variant: remap offsets for new text
                cache_match_type = "template_variant"
                changed_fields = _detect_changed_fields(cached_extracted_text, text)
                response_clauses = _remap_clause_offsets(cached_clauses, text, page_texts)
                response_page_texts = page_texts
                response_extracted_text = text
                store_changed_fields_meta(doc_hash, changed_fields)
                print(
                    f"[upload-file] Template variant detected for doc:{doc_hash[:16]}... "
                    f"changed={changed_fields}"
                )

            unknown_count = len([
                c for c in response_clauses
                if str(c.get("clause_type", "")) == "Unknown clause"
            ])
            verification = get_verification_state(doc_hash, user_id, unknown_count)
            history = _get_effective_verification_history(doc_hash)

            return {
                "status": "cache_hit",
                "cache_match_type": cache_match_type,
                "cached_at": analyzed_at,
                "changed_fields": changed_fields,
                "doc_hash": doc_hash,
                "extracted_text": response_extracted_text,
                "extracted_text_preview": (
                    response_extracted_text[:500] + "..."
                    if len(response_extracted_text) > 500
                    else response_extracted_text
                ),
                "page_count": len(response_page_texts),
                "page_texts": response_page_texts,
                "result": response_clauses,
                "file_type": cached_file_type,
                "verification": verification,
                "history": history,
            }

        # Step 3: cache miss, run full model pipeline
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

        # Step 4: store result in Redis
        store_result(
            doc_hash=fingerprints,
            clauses=result,
            page_texts=page_texts,
            extracted_text=text,
            file_type=file_type,
            raw_hash=raw_hash,
        )

        unknown_count = len([r for r in result if r.get("clause_type") == "Unknown clause"])
        verification = get_verification_state(doc_hash, user_id, unknown_count)
        history = _get_effective_verification_history(doc_hash)

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
    """Stage unknown-clause feedback temporarily and re-classify view output."""
    try:
        if not request.doc_hash or not request.user_id:
            raise HTTPException(status_code=400, detail="doc_hash and user_id are required for staged teaching")

        staged = add_pending_teach(
            doc_hash=request.doc_hash,
            user_id=request.user_id,
            span=request.unknown_span,
            label=request.new_type_name,
            color=request.color,
        )
        if not staged:
            raise HTTPException(status_code=500, detail="Failed to stage feedback")

        record_user_teach(request.doc_hash, request.user_id)

        pending_teaches = _pending_teaches_for_user(request.doc_hash, request.user_id)

        # Use cached results instead of re-running the entire model pipeline.
        # This reduces teach latency from ~5-15s to <100ms.
        cached = get_cached_result(request.doc_hash) or {}
        cached_clauses = cached.get("clauses", [])

        if cached_clauses:
            updated_results = _apply_pending_teaches_to_results(cached_clauses, pending_teaches)
        else:
            # Fallback: cache miss (should not happen in normal flow)
            updated_results = model.predict_cuad(request.contract_text)
            updated_results = _apply_pending_teaches_to_results(updated_results, pending_teaches)

        unknown_count = len([
            c for c in updated_results if c.get("clause_type") == "Unknown clause"
        ])
        verification = None
        history = []
        verification = get_verification_state(request.doc_hash, request.user_id, unknown_count)
        history = _get_effective_verification_history(request.doc_hash)

        stats = model.get_statistics()
        
        return {
            "status": "success",
            "message": f"Staged '{request.new_type_name}'. Click Verify to commit learning.",
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
    - First uploader can verify while verify-cycle is open.
    - Later users can verify only after teaching at least one unknown.
    - Teaching can reopen verification even if unknown_count becomes zero.
    """
    try:
        cached = get_cached_result(request.doc_hash)
        if not cached:
            raise HTTPException(
                status_code=404,
                detail="Document analysis not found in cache. Please upload the document again.",
            )

        pending_teaches = _pending_teaches_for_user(request.doc_hash, request.user_id)

        # Authoritative verification source: server-side cached analysis.
        # Do not trust client-provided clauses for permission checks or proof.
        clauses = cached.get("clauses", [])
        clauses = _apply_pending_teaches_to_results(clauses, pending_teaches)
        unknown_count = len([
            c for c in clauses
            if str(c.get("clause_type", "")) == "Unknown clause"
        ])

        allowed, reason = can_user_verify(request.doc_hash, request.user_id, unknown_count)
        if not allowed:
            raise HTTPException(status_code=403, detail=reason)

        # Commit staged teaches to support set only at verify time.
        if pending_teaches:
            for teach in pending_teaches:
                span = str(teach.get("span", "")).strip()
                label = str(teach.get("label", "")).strip()
                color = teach.get("color")
                if not span or not label:
                    continue
                committed = model.learn_from_feedback(
                    clause_text=span,
                    correct_label=label,
                    color=str(color) if color else None,
                )
                if not committed:
                    raise HTTPException(status_code=500, detail="Failed to commit staged feedback during verify")

            extracted_text = str(cached.get("extracted_text", ""))
            if extracted_text:
                clauses = model.predict_cuad(extracted_text)
                clauses = _apply_pending_teaches_to_results(clauses, pending_teaches)
                store_result(
                    doc_hash=request.doc_hash,
                    clauses=clauses,
                    page_texts=cached.get("page_texts", []),
                    extracted_text=extracted_text,
                    file_type=str(cached.get("file_type", "unknown")),
                    raw_hash=cached.get("raw_hash"),
                )

            clear_pending_teaches_for_user(request.doc_hash, request.user_id)

            unknown_count = len([
                c for c in clauses
                if str(c.get("clause_type", "")) == "Unknown clause"
            ])

        clause_types = sorted({
            str(c.get("clause_type", "Unknown clause"))
            for c in clauses
            if c.get("clause_type")
        })
        analysis_hash = _sha256_json(cached)
        verified_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"Sending tx for doc {request.doc_hash}")

        geo_audit = _build_geo_audit_payload(request.geolocation)
        geo_hash = geo_audit.get("geo_hash")
        geo_summary = geo_audit.get("geo_summary")

        # Step 1 — Pin analysis summary to IPFS (non-fatal if Pinata unavailable).
        # CID links the on-chain hash to a retrievable off-chain analysis record.
        cid = _pin_to_ipfs(
            doc_hash=request.doc_hash,
            analysis_hash=analysis_hash,
            clause_types=clause_types,
            clause_count=len(clauses),
            unknown_count=unknown_count,
            verified_at=verified_at_iso,
        )

        # Step 2 — Write immutable proof to Sepolia with the CID embedded.
        try:
            tx_result = _send_sepolia_verification_tx(
                doc_hash=request.doc_hash,
                clause_types=clause_types,
                geo_hash=geo_hash,
                cid=cid,
            )
        except Exception as tx_exc:
            error_text = str(tx_exc)
            if "insufficient funds" in error_text.lower():
                error_text = "Insufficient funds"
            elif "private key" in error_text.lower() or "invalid" in error_text.lower():
                error_text = "Invalid key"
            raise HTTPException(
                status_code=502,
                detail=f"Blockchain transaction failed: {error_text}",
            )

        existing_history = _get_effective_verification_history(request.doc_hash)
        attempt_count = len(existing_history) + 1
        history_entry = {
            "attempt": attempt_count,
            "date": verified_at_iso,
            "tx_hash": tx_result["tx_hash"],
            "clause_summary": json.dumps(clause_types, ensure_ascii=False),
            "geo_hash": geo_hash,
            "geo_summary": geo_summary,
            "ipfs_cid": cid,
            "ipfs_url": f"https://gateway.pinata.cloud/ipfs/{cid}" if cid else None,
        }
        if not push_history_entry(request.doc_hash, history_entry):
            raise HTTPException(
                status_code=500,
                detail="Blockchain transaction succeeded but Redis history write failed",
            )

        changed_fields = get_changed_fields_meta(request.doc_hash)
        attempt = create_verification_attempt(
            doc_hash=request.doc_hash,
            user_id=request.user_id,
            clauses=clauses,
            unknown_count=unknown_count,
            attempt_number=attempt_count,
            tx_hash=tx_result["tx_hash"],
            blockchain_link=tx_result["explorer_link"],
            snapshot_hash=analysis_hash,
            geo_hash=geo_hash,
            geo_summary=geo_summary,
            changed_fields=changed_fields,
        )
        if attempt is None:
            raise HTTPException(status_code=500, detail="Failed to persist verification attempt")

        append_mongo_verification_attempt(request.doc_hash, attempt)

        history = _get_effective_verification_history(request.doc_hash)
        verification = get_verification_state(request.doc_hash, request.user_id, unknown_count)
        return {
            "status": "verified",
            "tx_hash": tx_result["tx_hash"],
            "explorer_link": tx_result["explorer_link"],
            "ipfs_cid": cid,
            "ipfs_url": f"https://gateway.pinata.cloud/ipfs/{cid}" if cid else None,
            "history": history,
            "message": "Document verified on Sepolia testnet.",
            "record": attempt,
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
        history = _get_effective_verification_history(doc_hash)
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
    Discard cached document data when a user leaves with an open verify cycle.

    Rules:
    - If document was never verified, discard.
    - If user taught after a prior verification and did not verify again,
      discard that open cycle data as well.
    - Keep only fully verified/closed cycles.
    """
    try:
        has_history = len(_get_effective_verification_history(request.doc_hash)) > 0
        cycle_open = has_open_verification_cycle(request.doc_hash)

        if has_history and not cycle_open:
            return {
                "status": "kept",
                "message": "Document already verified; history preserved.",
            }

        if not should_discard_on_leave(request.doc_hash):
            return {
                "status": "skipped",
                "message": "Redis unavailable or document not found.",
            }

        # Previously verified docs: roll back only unverified cycle changes,
        # preserve verified baseline metadata and history.
        if has_history and cycle_open:
            rolled_back = rollback_open_cycle_data(request.doc_hash)
            if rolled_back:
                return {
                    "status": "discarded",
                    "message": "Open/unverified cycle data discarded; verified history preserved.",
                }
            return {
                "status": "skipped",
                "message": "Redis unavailable or document not found.",
            }

        # Never-verified docs: delete all cycle keys.
        deleted = discard_document_data(request.doc_hash)
        if deleted:
            return {
                "status": "discarded",
                "message": "Open/unverified document cycle data discarded.",
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