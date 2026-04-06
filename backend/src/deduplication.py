import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import redis

from src.data import normalize_text

logger = logging.getLogger(__name__)

# 90 days in seconds
CACHE_TTL: int = 90 * 24 * 3600

# Shared Redis client
_redis_client: Optional[redis.Redis] = None  # type: ignore[type-arg]


def _get_redis() -> Optional[redis.Redis]:  # type: ignore[type-arg]
    """Return Redis client, or None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        # Reuse current connection if healthy
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            # Reconnect if the old one is stale
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


# Public API

def _tokenize_for_fingerprint(normalized_text: str) -> List[str]:
    stop = {
        "the", "and", "or", "of", "to", "in", "for", "with", "on", "by", "is", "are",
        "this", "that", "shall", "will", "be", "as", "at", "it", "an", "a",
    }
    return [tok for tok in normalized_text.split() if tok and tok not in stop]


def _simhash64(tokens: List[str]) -> int:
    if not tokens:
        return 0
    weights = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            bit = (h >> i) & 1
            weights[i] += 1 if bit else -1
    out = 0
    for i, w in enumerate(weights):
        if w >= 0:
            out |= (1 << i)
    return out


def _bucket_keys_for_simhash(simhash64: int) -> List[str]:
    return [
        f"{(simhash64 >> 0) & 0xFFFF:04x}",
        f"{(simhash64 >> 16) & 0xFFFF:04x}",
        f"{(simhash64 >> 32) & 0xFFFF:04x}",
        f"{(simhash64 >> 48) & 0xFFFF:04x}",
    ]


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_doc_fingerprints(raw_text: str) -> Dict[str, Any]:
    """Compute exact and near-duplicate fingerprints."""
    normalized = normalize_text(raw_text)
    tokens = _tokenize_for_fingerprint(normalized)
    stable_text = " ".join(tokens)
    primary_hash = hashlib.sha256(stable_text.encode("utf-8")).hexdigest()
    simhash64 = _simhash64(tokens)
    buckets = _bucket_keys_for_simhash(simhash64)
    return {
        "primary_hash": primary_hash,
        "simhash64": simhash64,
        "simhash_hex": f"{simhash64:016x}",
        "buckets": buckets,
        "token_count": len(tokens),
        "normalized_text": normalized,
    }

def compute_doc_hash(raw_text: str) -> str:
    """Return stable SHA-256 hash for document content."""
    return str(compute_doc_fingerprints(raw_text)["primary_hash"])


def _meta_key(doc_hash: str) -> str:
    return f"docmeta:{doc_hash}"


def _doc_key(doc_hash: str) -> str:
    return f"doc:{doc_hash}"


def _history_key(doc_hash: str) -> str:
    return f"history:{doc_hash}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_meta(user_id: str) -> Dict[str, Any]:
    return {
        "first_uploader": user_id,
        "first_uploaded_at": _now_iso(),
        "taught_users": {},
        "taught_users_at_last_verify": {},
        "pending_teaches": [],
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


def get_pending_teaches(doc_hash: str) -> List[Dict[str, Any]]:
    """Return staged teach actions waiting for verification commit."""
    meta = get_document_meta(doc_hash)
    if not meta:
        return []
    pending = meta.get("pending_teaches")
    return pending if isinstance(pending, list) else []


def add_pending_teach(
    doc_hash: str,
    user_id: str,
    span: str,
    label: str,
    color: Optional[str] = None,
) -> bool:
    """Stage a teach action in metadata; committed to support set only on verify."""
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta(user)

    pending = meta.get("pending_teaches")
    entries: List[Dict[str, Any]] = pending if isinstance(pending, list) else []

    # Deduplicate by (user, span, label) so repeated clicks don't create duplicates.
    filtered = [
        e for e in entries
        if not (
            isinstance(e, dict)
            and str(e.get("user_id", "")) == user
            and str(e.get("span", "")) == span
            and str(e.get("label", "")) == label
        )
    ]
    filtered.append({
        "user_id": user,
        "span": span,
        "label": label,
        "color": color,
        "staged_at": _now_iso(),
    })
    meta["pending_teaches"] = filtered
    return _save_document_meta(doc_hash, meta)


def clear_pending_teaches(doc_hash: str) -> bool:
    meta = get_document_meta(doc_hash)
    if meta is None:
        return False
    meta["pending_teaches"] = []
    return _save_document_meta(doc_hash, meta)


def clear_pending_teaches_for_user(doc_hash: str, user_id: str) -> bool:
    """Remove staged teaches only for the given user; keep other users' staged data."""
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        return False

    pending = meta.get("pending_teaches")
    entries: List[Dict[str, Any]] = pending if isinstance(pending, list) else []
    filtered = [
        entry
        for entry in entries
        if not (
            isinstance(entry, dict)
            and str(entry.get("user_id", "")).strip() == user
        )
    ]
    meta["pending_teaches"] = filtered
    return _save_document_meta(doc_hash, meta)


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
        """Return True when a new verification is needed."""
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
    """Check if this user can verify now."""
    meta = get_document_meta(doc_hash)
    if not _is_verify_cycle_open(meta):
        return False, "Verification already recorded. Teach at least one unknown clause to unlock verify again."

    if is_first_uploader(doc_hash, user_id):
        return True, "First uploader verification allowed."

    teach_count = get_user_teach_count(doc_hash, user_id)
    if teach_count > 0:
        return True, "User has taught at least one unknown clause."

    if unknown_count > 0:
        return False, "You need to teach at least one unknown clause before you can verify."

    return False, "Teach at least one clause update to unlock verification."


def get_verification_history(doc_hash: str) -> List[Dict[str, Any]]:
    meta = get_document_meta(doc_hash)
    if not meta:
        return []
    history = meta.get("verification_history")
    return history if isinstance(history, list) else []


def seed_verification_baseline(
    doc_hash: str,
    user_id: str,
    history: List[Dict[str, Any]],
) -> bool:
    """Seed Redis metadata from durable verification history."""
    if not history:
        return False

    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta(user)

    existing = meta.get("verification_history")
    existing_attempts: List[Dict[str, Any]] = existing if isinstance(existing, list) else []
    if existing_attempts:
        return False

    taught_users = meta.get("taught_users")
    taught_map: Dict[str, Any] = taught_users if isinstance(taught_users, dict) else {}

    meta["verification_history"] = history
    meta["last_verified_taught_total"] = _get_total_teach_count(meta)
    meta["taught_users_at_last_verify"] = dict(taught_map)
    return _save_document_meta(doc_hash, meta)


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
        "show_verify_button": cycle_open and (unknown_count > 0 or taught_count > 0),
        "is_first_uploader": first,
        "user_taught_count": taught_count,
        "can_verify": allowed,
        "message": reason,
    }


def store_changed_fields_meta(doc_hash: str, changed_fields: List[str]) -> bool:
    """Persist which placeholder types changed for a template-variant upload."""
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta("anonymous")
    meta["last_changed_fields"] = changed_fields
    return _save_document_meta(doc_hash, meta)


def get_changed_fields_meta(doc_hash: str) -> List[str]:
    """Return the stored changed-fields list for a document, or []."""
    meta = get_document_meta(doc_hash)
    if not meta:
        return []
    fields = meta.get("last_changed_fields")
    return fields if isinstance(fields, list) else []


def create_verification_attempt(
    doc_hash: str,
    user_id: str,
    clauses: List[Dict[str, Any]],
    unknown_count: int,
    attempt_number: Optional[int] = None,
    tx_hash: Optional[str] = None,
    blockchain_link: Optional[str] = None,
    snapshot_hash: Optional[str] = None,
    geo_hash: Optional[str] = None,
    geo_summary: Optional[str] = None,
    changed_fields: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Append one verification attempt to history."""
    user = (user_id or "anonymous").strip() or "anonymous"
    meta = get_document_meta(doc_hash)
    if meta is None:
        meta = _default_meta(user)

    history = meta.get("verification_history")
    attempts: List[Dict[str, Any]] = history if isinstance(history, list) else []
    verified_at = _now_iso()
    if snapshot_hash is None:
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

    if tx_hash is None:
        tx_hash = hashlib.sha256(f"tx:{snapshot_hash}:{verified_at}".encode("utf-8")).hexdigest()

    if blockchain_link is None:
        blockchain_link = f"https://blockchain.lexicache.app/proof/{tx_hash}"

    computed_attempt = int(attempt_number) if attempt_number is not None else len(attempts) + 1

    attempt = {
        "attempt": computed_attempt,
        "verified_at": verified_at,
        "verified_by": user,
        "clause_count": len(clauses),
        "unknown_count": unknown_count,
        "snapshot_hash": snapshot_hash,
        "tx_hash": tx_hash,
        "blockchain_link": blockchain_link,
        "geo_hash": geo_hash,
        "geo_summary": geo_summary,
        "changed_fields": changed_fields or [],
    }

    attempts.append(attempt)
    meta["verification_history"] = attempts
    meta["last_verified_taught_total"] = _get_total_teach_count(meta)
    taught_users = meta.get("taught_users")
    taught_map: Dict[str, Any] = taught_users if isinstance(taught_users, dict) else {}
    meta["taught_users_at_last_verify"] = dict(taught_map)
    if not _save_document_meta(doc_hash, meta):
        return None
    return attempt


def rollback_open_cycle_data(doc_hash: str) -> bool:
    """Rollback unverified cycle changes and keep verified baseline."""
    client = _get_redis()
    if client is None:
        return False

    meta = get_document_meta(doc_hash)
    if not meta:
        return False

    try:
        baseline_taught = meta.get("taught_users_at_last_verify")
        baseline_map: Dict[str, Any] = baseline_taught if isinstance(baseline_taught, dict) else {}

        meta["pending_teaches"] = []
        meta["taught_users"] = dict(baseline_map)
        meta["last_verified_taught_total"] = _get_total_teach_count({"taught_users": baseline_map})

        # Clear analysis cache but keep metadata/history
        client.delete(_doc_key(doc_hash), _history_key(doc_hash))
        if not _save_document_meta(doc_hash, meta):
            return False

        print(f"[Redis] Rolled back open cycle data for doc:{doc_hash[:16]}...")
        logger.info("[Redis] Rolled back open cycle data for doc:%s...", doc_hash[:16])
        return True
    except Exception as exc:
        print(f"[Redis] rollback_open_cycle_data failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] rollback_open_cycle_data failed: %s", exc)
        return False


def push_history_entry(doc_hash: str, entry: Dict[str, Any]) -> bool:
    """Append a blockchain verification entry to the Redis history list."""
    client = _get_redis()
    if client is None:
        return False

    try:
        client.rpush(_history_key(doc_hash), json.dumps(entry))
        client.expire(_history_key(doc_hash), CACHE_TTL)
        return True
    except Exception as exc:
        print(f"[Redis] push_history_entry failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] push_history_entry failed: %s", exc)
        return False


def get_history_entries(doc_hash: str) -> List[Dict[str, Any]]:
    """Return parsed history entries from Redis list history:{doc_hash}."""
    client = _get_redis()
    if client is None:
        return []

    try:
        raw_items = client.lrange(_history_key(doc_hash), 0, -1)
        entries: List[Dict[str, Any]] = []
        for item in raw_items:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    entries.append(parsed)
            except Exception:
                continue
        return entries
    except Exception as exc:
        print(f"[Redis] get_history_entries failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] get_history_entries failed: %s", exc)
        return []


def has_verification_history(doc_hash: str) -> bool:
    meta = get_document_meta(doc_hash)
    if not meta:
        return False
    history = meta.get("verification_history")
    return isinstance(history, list) and len(history) > 0


def has_open_verification_cycle(doc_hash: str) -> bool:
    """Return True when a document currently requires a fresh verification."""
    meta = get_document_meta(doc_hash)
    if not meta:
        return False
    return _is_verify_cycle_open(meta)


def should_discard_on_leave(doc_hash: str) -> bool:
    """Return True when unverified changes should be discarded on leave."""
    meta = get_document_meta(doc_hash)
    if not meta:
        return False
    history = meta.get("verification_history")
    attempts: List[Dict[str, Any]] = history if isinstance(history, list) else []
    if len(attempts) == 0:
        return True
    return _is_verify_cycle_open(meta)


def discard_document_data(doc_hash: str) -> bool:
    """Delete analysis + metadata keys for a document hash from Redis."""
    client = _get_redis()
    if client is None:
        return False

    try:
        client.delete(_doc_key(doc_hash), _meta_key(doc_hash), _history_key(doc_hash))
        print(f"[Redis] Discarded document data for doc:{doc_hash[:16]}...")
        logger.info("[Redis] Discarded document data for doc:%s...", doc_hash[:16])
        return True
    except Exception as exc:
        print(f"[Redis] discard_document_data failed for doc:{doc_hash[:16]}...: {exc}")
        logger.warning("[Redis] discard_document_data failed: %s", exc)
        return False


def _load_docmeta(client: redis.Redis, doc_hash: str) -> Optional[Dict[str, Any]]:  # type: ignore[type-arg]
    try:
        raw = client.get(_meta_key(doc_hash))
        if not raw:
            return None
        meta = json.loads(raw)
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def get_cached_result(doc_hash: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return cached analysis result from Redis if available."""
    client = _get_redis()
    if client is None:
        return None

    # Strict hash lookup
    if isinstance(doc_hash, str):
        try:
            raw = client.get(_doc_key(doc_hash))
            if raw is None:
                print(f"[Redis] Cache miss for doc:{doc_hash[:16]}...")
                logger.debug("[Redis] Cache miss for doc:%s...", doc_hash[:16])
                return None

            data: Dict[str, Any] = json.loads(raw)  # type: ignore[assignment]
            print(f"[Redis] Cache HIT  for doc:{doc_hash[:16]}... (analyzed_at={data.get('analyzed_at', '?')})")
            logger.info("[Redis] Cache HIT for doc:%s...", doc_hash[:16])
            return data

        except json.JSONDecodeError as exc:
            print(f"[Redis] Corrupt cache entry for doc:{doc_hash[:16]}... — deleting. ({exc})")
            logger.warning("[Redis] Corrupt cache entry deleted: %s", exc)
            try:
                client.delete(_doc_key(doc_hash))
            except Exception:
                pass
            return None

        except Exception as exc:
            print(f"[Redis] get() failed for doc:{doc_hash[:16]}...: {exc}")
            logger.warning("[Redis] get() failed: %s", exc)
            return None

    # Near-duplicate lookup
    if not isinstance(doc_hash, dict):
        return None

    primary_hash = str(doc_hash.get("primary_hash", "")).strip()
    simhash64 = int(doc_hash.get("simhash64", 0) or 0)
    token_count = int(doc_hash.get("token_count", 0) or 0)
    buckets = doc_hash.get("buckets", [])

    if not primary_hash:
        return None

    # Exact primary hash first.
    exact = get_cached_result(primary_hash)
    if exact is not None:
        exact["cache_match_type"] = "exact"
        return exact

    candidate_hashes: Set[str] = set()
    if isinstance(buckets, list):
        for b in buckets:
            try:
                members = client.smembers(f"docbucket:{b}")
                candidate_hashes.update(str(h) for h in members)
            except Exception:
                continue

    best_hash: Optional[str] = None
    best_dist = 65

    for cand in candidate_hashes:
        meta = _load_docmeta(client, cand)
        if not meta:
            continue

        cand_sim = int(meta.get("simhash64", 0) or 0)
        cand_tokens = int(meta.get("token_count", 0) or 0)

        dist = _hamming64(simhash64, cand_sim)
        if dist > 3:
            continue

        if token_count > 0 and cand_tokens > 0:
            ratio = abs(cand_tokens - token_count) / float(max(cand_tokens, token_count))
            if ratio > 0.12:
                continue

        if dist < best_dist:
            best_dist = dist
            best_hash = cand

    if not best_hash:
        print(f"[Redis] Near-duplicate miss for doc:{primary_hash[:16]}...")
        logger.debug("[Redis] Near-duplicate miss for doc:%s...", primary_hash[:16])
        return None

    try:
        raw = client.get(_doc_key(best_hash))
        if raw is None:
            return None
        data: Dict[str, Any] = json.loads(raw)  # type: ignore[assignment]
        data["cache_match_type"] = "near_duplicate"
        data["cache_hamming_distance"] = best_dist
        print(
            f"[Redis] Near-duplicate HIT for doc:{primary_hash[:16]}... "
            f"via {best_hash[:16]}... (dist={best_dist})"
        )
        logger.info(
            "[Redis] Near-duplicate HIT for doc:%s... via %s... dist=%d",
            primary_hash[:16],
            best_hash[:16],
            best_dist,
        )
        return data

    except Exception as exc:
        print(f"[Redis] near-duplicate get() failed for doc:{primary_hash[:16]}...: {exc}")
        logger.warning("[Redis] near-duplicate get() failed: %s", exc)
        return None


def store_result(
    doc_hash: Union[str, Dict[str, Any]],
    clauses: List[Dict[str, Any]],
    page_texts: List[Dict[str, Any]],
    extracted_text: str,
    file_type: str,
    raw_hash: Optional[str] = None,
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

    if isinstance(doc_hash, str):
        fp = {
            "primary_hash": doc_hash,
            "simhash64": 0,
            "token_count": 0,
            "buckets": [],
        }
    elif isinstance(doc_hash, dict):
        fp = doc_hash
    else:
        return False

    primary_hash = str(fp.get("primary_hash", "")).strip()
    simhash64 = int(fp.get("simhash64", 0) or 0)
    token_count = int(fp.get("token_count", 0) or 0)
    buckets = fp.get("buckets", [])
    if not primary_hash:
        return False

    try:
        payload: Dict[str, Any] = {
            "clauses": clauses,
            "page_texts": page_texts,
            "extracted_text": extracted_text,
            "file_type": file_type,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "raw_hash": raw_hash,
        }

        meta = get_document_meta(primary_hash) or _default_meta("anonymous")
        meta["simhash64"] = simhash64
        meta["token_count"] = token_count
        meta["buckets"] = buckets if isinstance(buckets, list) else []

        client.set(_doc_key(primary_hash), json.dumps(payload), ex=CACHE_TTL)
        client.set(_meta_key(primary_hash), json.dumps(meta), ex=CACHE_TTL)
        if isinstance(buckets, list):
            for bucket in buckets:
                key = f"docbucket:{bucket}"
                client.sadd(key, primary_hash)
                client.expire(key, CACHE_TTL)

        print(
            f"[Redis] Stored  doc:{primary_hash[:16]}... "
            f"(TTL={CACHE_TTL // 86400} days, "
            f"{len(clauses)} clauses)"
        )
        logger.info(
            "[Redis] Stored doc:%s... TTL=%ds clauses=%d",
            primary_hash[:16], CACHE_TTL, len(clauses),
        )
        return True

    except Exception as exc:
        print(f"[Redis] store() failed for doc:{primary_hash[:16]}...: {exc}")
        logger.warning("[Redis] store() failed: %s", exc)
        return False
