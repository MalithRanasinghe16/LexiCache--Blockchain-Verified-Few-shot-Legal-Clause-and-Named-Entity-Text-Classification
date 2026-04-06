# LexiCache

LexiCache is an adaptive legal document analysis platform that classifies clauses in contracts, lets users teach unknown clauses, and records verification proofs on Sepolia blockchain.

## What It Does

- Upload PDF or DOCX contracts and extract clause predictions.
- Use hybrid classification (keyword rules + Legal-BERT based model).
- Teach unknown clauses from the UI with rename flow.
- Stage teaches first, then commit them only when verifying.
- Verify document analysis on-chain and optionally pin metadata to IPFS.
- Reuse previous analysis with Redis cache and template-variant handling.

## Architecture

- Backend: FastAPI + Python ML stack.
- Frontend: Next.js + React + react-pdf.
- Model: Prototypical/few-shot style pipeline in `backend/src/ml_model.py`.
- Cache and metadata: Redis (90-day TTL).
- Durable verification history (optional): MongoDB.
- Blockchain verification: Sepolia via Web3.

## Requirements

- Python 3.10+
- Node.js 18+
- npm
- Redis running on `localhost:6379` (recommended)

Optional:

- MongoDB (durable verification history)
- Pinata JWT (IPFS pinning)

## Setup

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env`:

```env
# Required for blockchain verify
PRIVATE_KEY=0xyour_wallet_private_key
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/your_project_id

# Optional override
CONTRACT_ADDRESS=0x9B29820FEc9B0497b91175205C454FB06c576777

# Optional durable history
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=lexicache
MONGODB_HISTORY_COLLECTION=document_history

# Optional IPFS pinning
PINATA_JWT=your_pinata_jwt
```

### Frontend

```bash
cd frontend
npm install
```

## Run

### Start Backend

```bash
cd backend
python start_server.py
```

Alternative:

```bash
cd backend
python -m scripts.start_server
```

Backend API: `http://127.0.0.1:8000`

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend: `http://localhost:3000`

### Quick Health Check

```bash
curl http://localhost:8000/health
```

Expected:

```json
{ "status": "healthy", "model_loaded": true }
```

## API Overview

Core:

- `GET /` API info
- `GET /health` health status
- `GET /statistics` model stats
- `GET /clause-types`
- `GET /clause-types-with-colors`

Document flow:

- `POST /upload-file` (multipart: `file`, `user_id`)
- `POST /rename-unknown` (stages teach for unknown clause)
- `POST /verify` or `POST /verify-document` (commits staged teaches + blockchain proof)
- `GET /document-history/{doc_hash}`
- `POST /discard-document`
- `POST /update-color`
- `POST /predict-text`
- `POST /feedback`

## Cache and Variant Behavior

When uploading, backend computes normalized fingerprint:

- Exact cache hit: returns cached analysis immediately.
- Template variant hit: same normalized structure but different raw values (for example date/party/amount changes).

For template variants, backend:

- marks response as `cache_match_type: "template_variant"`
- reports `changed_fields`
- remaps clause offsets to the new text when possible

If no cache match, backend runs full model inference and stores result in Redis.

## Verification Rules

Verification uses a cycle gate:

- First uploader can verify while cycle is open.
- Other users must teach at least one unknown clause for that document.
- After verification, cycle closes until new teaching occurs.

On successful verify:

- pending teaches are committed to the model
- transaction is sent to Sepolia contract
- verification history is recorded
- optional IPFS metadata CID is attached

## Testing

Run targeted verify-flow tests from backend directory:

```bash
cd backend
.venv\Scripts\python.exe -m pytest tests/test_verify_feature.py -q
```

Also available:

- `backend/tests/test_pipeline.py`
- `backend/tests/test_infura.py`

## Scripts

### Data

- `backend/scripts/data/prepare_cuad_full.py`
- `backend/scripts/data/split_cuad.py`

### Training

- `backend/scripts/training/train_cuad_multilabel_finetune.py`
- `backend/scripts/training/fine_tune_multi.py`
- `backend/scripts/training/experiments.py`

### Evaluation

- `backend/scripts/evaluation/evaluate_cuad_test.py`
- `backend/scripts/evaluation/evaluate_cuad_document_level.py`
- `backend/scripts/evaluation/evaluate_cuad_hybrid_ablation.py`
- `backend/scripts/evaluation/evaluate_cuad_multilabel_finetuned.py`
- `backend/scripts/evaluation/evaluate_conll2003_test.py`
- `backend/scripts/evaluation/evaluate_ledgar_test.py`
- `backend/scripts/evaluation/evaluate_fewshot_teaching.py`
- `backend/scripts/evaluation/evaluate_cache_blockchain.py`
- `backend/scripts/evaluation/generate_eval_report.py`
- `backend/scripts/evaluation/run_cuad_sweep.py`

## Project Layout

```text
LexiCache/
  backend/
    main.py
    start_server.py
    requirements.txt
    src/
      api/main.py
      ml_model.py
      deduplication.py
      history_store.py
      data.py
      modeling.py
    scripts/
      data/
      training/
      evaluation/
    tests/
  frontend/
    package.json
    src/app/
      page.tsx
      types.ts
      components/
```

## Notes

- Redis or blockchain failure should not crash the app; endpoints degrade with clear errors.
- DOCX and PDF are both supported, but rendering/highlighting paths differ.
- Verification history can come from MongoDB fallback even if Redis metadata is missing.

## License

Add your license details here.
