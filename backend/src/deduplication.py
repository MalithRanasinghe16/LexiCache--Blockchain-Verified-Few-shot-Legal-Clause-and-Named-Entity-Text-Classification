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
from typing import Any, Dict, List, Optional

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
