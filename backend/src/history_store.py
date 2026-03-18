"""MongoDB-backed durable verification history store.

This module is optional at runtime. If pymongo is unavailable or MONGODB_URI
is not configured, calls degrade gracefully and return empty/False.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_mongo_collection: Any = None
_mongo_init_attempted = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_collection() -> Any:
    global _mongo_collection
    global _mongo_init_attempted

    if _mongo_collection is not None:
        return _mongo_collection

    if _mongo_init_attempted:
        return None

    _mongo_init_attempted = True

    mongo_uri = (os.getenv("MONGODB_URI") or "").strip()
    if not mongo_uri:
        logger.info("[Mongo] MONGODB_URI not set; durable history disabled")
        return None

    try:
        from pymongo import MongoClient
    except Exception as exc:
        logger.warning("[Mongo] pymongo import failed; durable history disabled: %s", exc)
        return None

    db_name = (os.getenv("MONGODB_DB_NAME") or "lexicache").strip()
    collection_name = (os.getenv("MONGODB_HISTORY_COLLECTION") or "document_history").strip()

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        collection = client[db_name][collection_name]
        collection.create_index("doc_hash", unique=True)
        _mongo_collection = collection
        logger.info("[Mongo] Connected to %s.%s", db_name, collection_name)
        return _mongo_collection
    except Exception as exc:
        logger.warning("[Mongo] Connection failed; durable history disabled: %s", exc)
        return None


def append_verification_attempt(doc_hash: str, attempt: Dict[str, Any]) -> bool:
    """Persist a full verification attempt for long-term retention."""
    col = _get_collection()
    if col is None:
        return False

    try:
        now_iso = _now_iso()
        col.update_one(
            {"doc_hash": doc_hash},
            {
                "$setOnInsert": {
                    "doc_hash": doc_hash,
                    "created_at": now_iso,
                },
                "$set": {
                    "updated_at": now_iso,
                },
                "$push": {
                    "verification_history": attempt,
                },
            },
            upsert=True,
        )
        return True
    except Exception as exc:
        logger.warning("[Mongo] append_verification_attempt failed for %s: %s", doc_hash[:16], exc)
        return False


def get_verification_history(doc_hash: str) -> List[Dict[str, Any]]:
    """Fetch durable verification attempts for doc_hash."""
    col = _get_collection()
    if col is None:
        return []

    try:
        doc: Optional[Dict[str, Any]] = col.find_one(
            {"doc_hash": doc_hash},
            {"_id": 0, "verification_history": 1},
        )
        if not doc:
            return []
        history = doc.get("verification_history")
        if not isinstance(history, list):
            return []

        attempts = [h for h in history if isinstance(h, dict)]
        return sorted(attempts, key=lambda h: int(h.get("attempt", 0)))
    except Exception as exc:
        logger.warning("[Mongo] get_verification_history failed for %s: %s", doc_hash[:16], exc)
        return []
