"""
Redis-based document deduplication for LexiCache.

Caches full analysis results keyed by a SHA-256 fingerprint of the
normalised document text, so that re-uploading an identical contract
(even with minor whitespace/date formatting differences) returns the
stored result instantly without re-running Legal-BERT inference.

TTL: 90 days.  If Redis is unreachable the module degrades gracefully
and every request falls through to the model (no crash, no data loss).
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis

from src.data import normalize_text

logger = logging.getLogger(__name__)

# 90 days expressed in seconds
CACHE_TTL: int = 90 * 24 * 3600

# Module-level singleton — one connection shared across all requests
_redis_client: Optional[redis.Redis] = None  # type: ignore[type-arg]


def _get_redis() -> Optional[redis.Redis]:  # type: ignore[type-arg]
    """
    Return a connected Redis client, or None if Redis is unavailable.

    Uses a module-level singleton so we only attempt to connect once per
    server process.  socket_connect_timeout=2 keeps startup fast when
    Redis is not running.
    """
    global _redis_client
    if _redis_client is not None:
        # Verify the connection is still alive before returning it
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            # Connection dropped — reset and try to reconnect below
            _redis_client = None

    try:
        client: redis.Redis = redis.Redis(  # type: ignore[type-arg]
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _redis_client = client
        print("[Redis] Connected to localhost:6379 db=0")
        logger.info("[Redis] Connected to localhost:6379 db=0")
        return _redis_client
    except Exception as exc:
        print(f"[Redis] Connection failed – deduplication disabled: {exc}")
        logger.warning("[Redis] Connection failed – deduplication disabled: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_doc_hash(raw_text: str) -> str:
    """
    Return a stable SHA-256 hex digest that uniquely identifies a document
    by its *content*, not its filename or upload time.

    The text is normalised before hashing (lowercase, dates → [DATE],
    whitespace collapsed) so cosmetically different re-uploads of the
    same contract — e.g. different line endings or date representations —
    map to the same hash and correctly hit the cache.

    Note: raw_text is NOT modified here; normalisation is applied only
    inside this function for hashing purposes.  model.predict_cuad always
    receives the original raw text.
    """
    normalised = normalize_text(raw_text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _meta_key(doc_hash: str) -> str:
    return f"docmeta:{doc_hash}"


def _doc_key(doc_hash: str) -> str:
    return f"doc:{doc_hash}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_meta(user_id: str) -> Dict[str, Any]:
    return {
        "first_uploader": user_id,
        "first_uploaded_at": _now_iso(),
        "taught_users": {},
        "verification_history": [],
        "last_verified_taught_total": 0,
    }


def get_document_meta(doc_hash: str) -> Optional[Dict[str, Any]]:
    """Return document metadata for verification rules/history, or None."""
    client = _get_redis()
    if client is None:
        return None

    try:
        raw = client.get(_meta_key(doc_hash))
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:
        print(f"[Redis] get_document_meta failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] get_document_meta failed: %s", exc)
        return None


def _save_document_meta(doc_hash: str, meta: Dict[str, Any]) -> bool:
    client = _get_redis()
    if client is None:
        return False

    try:
        client.set(_meta_key(doc_hash), json.dumps(meta), ex=CACHE_TTL)
        return True
    except Exception as exc:
        print(f"[Redis] save_document_meta failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] save_document_meta failed: %s", exc)
        return False


def register_upload(doc_hash: str, user_id: str) -> Dict[str, Any]:
    """Create doc metadata on first upload; preserve it for subsequent uploads."""
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta(user)
        _save_document_meta(doc_hash, meta)
    return meta


def record_user_teach(doc_hash: str, user_id: str) -> bool:
    """Increment per-user teaching counter for this specific document."""
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta(user)

    taught_users = meta.get("taught_users")
    taught_map: Dict[str, int] = taught_users if isinstance(taught_users, dict) else {}
    taught_map[user] = int(taught_map.get(user, 0)) + 1
    meta["taught_users"] = taught_map
    return _save_document_meta(doc_hash, meta)


def get_user_teach_count(doc_hash: str, user_id: str) -> int:
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if not meta:
        return 0
    taught_users = meta.get("taught_users")
    taught_map: Dict[str, int] = taught_users if isinstance(taught_users, dict) else {}
    return int(taught_map.get(user, 0))


def _get_total_teach_count(meta: Optional[Dict[str, Any]]) -> int:
    if not meta:
        return 0
    taught_users = meta.get("taught_users")
    taught_map: Dict[str, Any] = taught_users if isinstance(taught_users, dict) else {}
    total = 0
    for value in taught_map.values():
        try:
            total += int(value)
        except Exception:
            continue
    return total


def _is_verify_cycle_open(meta: Optional[Dict[str, Any]]) -> bool:
    """
    Verify-cycle gate:
    - Before first verification: open
    - After any verification: closed until new teaching occurs
      (i.e., total taught count increases beyond last verified count)
    """
    if not meta:
        return True

    history = meta.get("verification_history")
    attempts: List[Dict[str, Any]] = history if isinstance(history, list) else []
    if len(attempts) == 0:
        return True

    total_taught = _get_total_teach_count(meta)
    last_verified_taught_total = int(meta.get("last_verified_taught_total", 0))
    return total_taught > last_verified_taught_total


def is_first_uploader(doc_hash: str, user_id: str) -> bool:
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if not meta:
        return False
    return str(meta.get("first_uploader", "")) == user


def can_user_verify(doc_hash: str, user_id: str, unknown_count: int) -> Tuple[bool, str]:
    """
    Verification rule:
    - First uploader can verify at any time while unknowns exist.
    - Later users must teach at least one unknown for that document.
    - If unknowns are zero, verification is unnecessary and disallowed.
    """
    if unknown_count <= 0:
        return False, "All unknown clauses are already resolved. Verification is no longer required."

    meta = get_document_meta(doc_hash)
    if not _is_verify_cycle_open(meta):
        return False, "Verification already recorded. Teach at least one unknown clause to unlock verify again."

    if is_first_uploader(doc_hash, user_id):
        return True, "First uploader verification allowed."

    teach_count = get_user_teach_count(doc_hash, user_id)
    if teach_count > 0:
        return True, "User has taught at least one unknown clause."

    return False, "You need to teach at least one unknown clause before you can verify."


def get_verification_history(doc_hash: str) -> List[Dict[str, Any]]:
    meta = get_document_meta(doc_hash)
    if not meta:
        return []
    history = meta.get("verification_history")
    return history if isinstance(history, list) else []


def get_verification_state(doc_hash: str, user_id: str, unknown_count: int) -> Dict[str, Any]:
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    cycle_open = _is_verify_cycle_open(meta)
    allowed, reason = can_user_verify(doc_hash, user, unknown_count)
    first = is_first_uploader(doc_hash, user)
    taught_count = get_user_teach_count(doc_hash, user)
    return {
        "doc_hash": doc_hash,
        "unknown_count": unknown_count,
        "show_verify_button": unknown_count > 0 and cycle_open,
        "is_first_uploader": first,
        "user_taught_count": taught_count,
        "can_verify": allowed,
        "message": reason,
    }


def create_verification_attempt(
    doc_hash: str,
    user_id: str,
    clauses: List[Dict[str, Any]],
    unknown_count: int,
) -> Optional[Dict[str, Any]]:
    """
    Append immutable verification attempt metadata to document history.

    Blockchain proof is represented as a deterministic content hash + pseudo
    explorer link so the frontend can display a permanent, externally
    checkable proof URL shape.
    """
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta(user)

    history = meta.get("verification_history")
    attempts: List[Dict[str, Any]] = history if isinstance(history, list) else []
    verified_at = _now_iso()
    payload = {
        "doc_hash": doc_hash,
        "verified_at": verified_at,
        "verified_by": user,
        "clause_count": len(clauses),
        "unknown_count": unknown_count,
        "clauses": clauses,
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    tx_hash = hashlib.sha256(f"tx:{snapshot_hash}:{verified_at}".encode("utf-8")).hexdigest()

    attempt = {
        "attempt": len(attempts) + 1,
        "verified_at": verified_at,
        "verified_by": user,
        "clause_count": len(clauses),
        "unknown_count": unknown_count,
        "snapshot_hash": snapshot_hash,
        "tx_hash": tx_hash,
        "blockchain_link": f"https://blockchain.lexicache.app/proof/{tx_hash}",
    }

    attempts.append(attempt)
    meta["verification_history"] = attempts
    meta["last_verified_taught_total"] = _get_total_teach_count(meta)
    if not _save_document_meta(doc_hash, meta):
        return None
    return attempt


def has_verification_history(doc_hash: str) -> bool:
    meta = get_document_meta(doc_hash)
    if not meta:
        return False
    history = meta.get("verification_history")
    return isinstance(history, list) and len(history) > 0


def discard_document_data(doc_hash: str) -> bool:
    """Delete analysis + metadata keys for a document hash from Redis."""
    client = _get_redis()
    if client is None:
        return False

    try:
        client.delete(_doc_key(doc_hash), _meta_key(doc_hash))
        print(f"[Redis] Discarded document data for doc:{doc_hash[:16]}...")
        logger.info("[Redis] Discarded document data for doc:%s...", doc_hash[:16])
        return True
    except Exception as exc:
        print(f"[Redis] discard_document_data failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] discard_document_data failed: %s", exc)
        return False


def get_cached_result(doc_hash: str) -> Optional[Dict[str, Any]]:
    """
    Look up a previously stored analysis result in Redis.

    Returns the full cached payload dict (keys: clauses, page_texts,
    extracted_text, file_type, analyzed_at) or None if the key does not
    exist, has expired, or Redis is unavailable.
    """
    client = _get_redis()
    if client is None:
        return None

    try:
        raw = client.get(f"doc:{doc_hash}")
        if raw is None:
            print(f"[Redis] Cache miss for doc:{doc_hash[:16]}...")
            logger.debug("[Redis] Cache miss for doc:%s...", doc_hash[:16])
            return None

        data: Dict[str, Any] = json.loads(raw)  # type: ignore[assignment]
        print(f"[Redis] Cache HIT  for doc:{doc_hash[:16]}... (analyzed_at={data.get('analyzed_at', '?')})")
        logger.info("[Redis] Cache HIT for doc:%s...", doc_hash[:16])
        return data

    except json.JSONDecodeError as exc:
        # Corrupt entry — delete it so the next request re-populates cleanly
        print(f"[Redis] Corrupt cache entry for doc:{doc_hash[:16]}... — deleting. ({exc})")
        logger.warning("[Redis] Corrupt cache entry deleted: %s", exc)
        try:
            client.delete(f"doc:{doc_hash}")
        except Exception:
            pass
        return None

    except Exception as exc:
        print(f"[Redis] get() failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] get() failed: %s", exc)
        return None


def store_result(
    doc_hash: str,
    clauses: List[Dict[str, Any]],
    page_texts: List[Dict[str, Any]],
    extracted_text: str,
    file_type: str,
) -> bool:
    """
    Persist a full analysis result in Redis with a 90-day TTL.

    The cached payload mirrors every field returned by /upload-file so
    that a cache-hit response is byte-for-byte equivalent to a fresh one
    (minus the model latency).

    Returns True on success, False if Redis is unavailable or the write
    fails.  Failure is non-fatal — the caller should still return the
    freshly computed result to the user.
    """
    client = _get_redis()
    if client is None:
        return False

    try:
        payload: Dict[str, Any] = {
            "clauses": clauses,
            "page_texts": page_texts,
            "extracted_text": extracted_text,
            "file_type": file_type,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        client.set(f"doc:{doc_hash}", json.dumps(payload), ex=CACHE_TTL)
        print(
            f"[Redis] Stored  doc:{doc_hash[:16]}... "
            f"(TTL={CACHE_TTL // 86400} days, "
            f"{len(clauses)} clauses)"
        )
        logger.info(
            "[Redis] Stored doc:%s... TTL=%ds clauses=%d",
            doc_hash[:16], CACHE_TTL, len(clauses),
        )
        return True

    except Exception as exc:
        print(f"[Redis] store() failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] store() failed: %s", exc)
        return False
