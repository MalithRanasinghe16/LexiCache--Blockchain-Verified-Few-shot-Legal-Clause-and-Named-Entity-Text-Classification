from __future__ import annotations

import importlib
import asyncio
import sys
from types import ModuleType
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest


class DummyModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.learn_from_feedback = Mock(return_value=True)
        self.predict_cuad = Mock(
            return_value=[
                {
                    "clause_type": "Known Clause",
                    "span": "sample span",
                    "confidence": 0.95,
                }
            ]
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {}

    def get_all_clause_types_with_colors(self) -> Dict[str, str]:
        return {}

    def update_clause_color(self, clause_type: str, color: str) -> bool:
        return True


@pytest.fixture()
def main_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Import main.py with a lightweight model constructor to keep tests fast.
    monkeypatch.setattr("src.ml_model.LexiCacheModel", DummyModel)

    if "main" in sys.modules:
        del sys.modules["main"]
    module = importlib.import_module("main")
    return module


def _verify_payload() -> Dict[str, Any]:
    return {
        "doc_hash": "doc_hash_123",
        "user_id": "user_1",
        "clauses": [],
    }


def test_verify_commits_staged_feedback_and_returns_tx(
    main_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "get_cached_result",
        lambda _doc_hash: {
            "clauses": [{"clause_type": "Unknown clause", "span": "sample span"}],
            "extracted_text": "full contract text",
            "page_texts": [],
            "file_type": "pdf",
        },
    )
    monkeypatch.setattr(main_module, "can_user_verify", lambda *_args: (True, "ok"))
    monkeypatch.setattr(
        main_module,
        "get_pending_teaches",
        lambda _doc_hash: [
            {
                "user_id": "user_1",
                "span": "sample span",
                "label": "Custom Clause",
                "color": "#123456",
            }
        ],
    )
    monkeypatch.setattr(
        main_module,
        "_send_sepolia_verification_tx",
        lambda **_kwargs: {
            "tx_hash": "0xabc123",
            "explorer_link": "https://sepolia.etherscan.io/tx/0xabc123",
        },
    )
    monkeypatch.setattr(main_module, "push_history_entry", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        main_module,
        "create_verification_attempt",
        lambda **_kwargs: {
            "attempt": 1,
            "tx_hash": "0xabc123",
            "blockchain_link": "https://sepolia.etherscan.io/tx/0xabc123",
        },
    )

    history_mock = Mock(
        side_effect=[
            [],
            [
                {
                    "attempt": 1,
                    "tx_hash": "0xabc123",
                    "verified_at": "2026-03-18T00:00:00Z",
                    "verified_by": "user_1",
                    "clause_count": 1,
                    "unknown_count": 0,
                    "snapshot_hash": "hash_1",
                    "blockchain_link": "https://sepolia.etherscan.io/tx/0xabc123",
                }
            ],
        ]
    )
    monkeypatch.setattr(main_module, "_get_effective_verification_history", history_mock)

    clear_pending_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "clear_pending_teaches", clear_pending_mock)

    store_result_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "store_result", store_result_mock)

    monkeypatch.setattr(
        main_module,
        "get_verification_state",
        lambda *_args: {
            "doc_hash": "doc_hash_123",
            "unknown_count": 0,
            "show_verify_button": False,
            "is_first_uploader": True,
            "user_taught_count": 1,
            "can_verify": False,
            "message": "Verification already recorded.",
        },
    )

    req = main_module.VerifyRequest(**_verify_payload())
    payload = asyncio.run(main_module.verify_document(req))
    assert payload["status"] == "verified"
    assert payload["tx_hash"] == "0xabc123"
    assert payload["explorer_link"].endswith("0xabc123")

    assert main_module.model.learn_from_feedback.call_count == 1
    assert clear_pending_mock.call_count == 1
    assert store_result_mock.call_count >= 1


def test_verify_fails_when_staged_feedback_commit_fails(
    main_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "get_cached_result",
        lambda _doc_hash: {
            "clauses": [{"clause_type": "Unknown clause", "span": "sample span"}],
            "extracted_text": "full contract text",
            "page_texts": [],
            "file_type": "pdf",
        },
    )
    monkeypatch.setattr(main_module, "can_user_verify", lambda *_args: (True, "ok"))
    monkeypatch.setattr(
        main_module,
        "get_pending_teaches",
        lambda _doc_hash: [
            {
                "user_id": "user_1",
                "span": "sample span",
                "label": "Custom Clause",
            }
        ],
    )

    main_module.model.learn_from_feedback = Mock(return_value=False)

    req = main_module.VerifyRequest(**_verify_payload())
    with pytest.raises(main_module.HTTPException) as exc_info:
        asyncio.run(main_module.verify_document(req))

    assert exc_info.value.status_code == 500
    assert "Failed to commit staged feedback" in str(exc_info.value.detail)


def test_discard_preserves_verified_baseline_for_open_cycle(
    main_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "has_verification_history", lambda _doc_hash: True)
    monkeypatch.setattr(main_module, "has_open_verification_cycle", lambda _doc_hash: True)
    monkeypatch.setattr(main_module, "should_discard_on_leave", lambda _doc_hash: True)

    rollback_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "rollback_open_cycle_data", rollback_mock)

    full_discard_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "discard_document_data", full_discard_mock)

    req = main_module.DiscardRequest(doc_hash="doc_hash_123", user_id="user_1")
    payload = asyncio.run(main_module.discard_document(req))

    assert payload["status"] == "discarded"
    assert "history preserved" in payload["message"].lower()
    assert rollback_mock.call_count == 1
    assert full_discard_mock.call_count == 0


def test_discard_uses_effective_history_when_redis_meta_missing(
    main_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "_get_effective_verification_history",
        lambda _doc_hash: [
            {
                "attempt": 1,
                "verified_at": "2026-03-22T00:00:00Z",
                "verified_by": "user_1",
                "clause_count": 3,
                "unknown_count": 0,
                "snapshot_hash": "hash_1",
                "tx_hash": "0xabc123",
                "blockchain_link": "https://sepolia.etherscan.io/tx/0xabc123",
            }
        ],
    )
    monkeypatch.setattr(main_module, "has_open_verification_cycle", lambda _doc_hash: False)
    monkeypatch.setattr(main_module, "should_discard_on_leave", lambda _doc_hash: True)

    rollback_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "rollback_open_cycle_data", rollback_mock)
    full_discard_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "discard_document_data", full_discard_mock)

    req = main_module.DiscardRequest(doc_hash="doc_hash_123", user_id="user_1")
    payload = asyncio.run(main_module.discard_document(req))

    assert payload["status"] == "kept"
    assert "history preserved" in payload["message"].lower()
    assert rollback_mock.call_count == 0
    assert full_discard_mock.call_count == 0


def test_upload_seeds_verification_baseline_from_durable_history(
    main_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "compute_doc_hash", lambda _text: "doc_hash_123")
    monkeypatch.setattr(main_module, "register_upload", lambda _doc_hash, _user_id: {})
    monkeypatch.setattr(
        main_module,
        "_get_effective_verification_history",
        lambda _doc_hash: [
            {
                "attempt": 1,
                "verified_at": "2026-03-22T00:00:00Z",
                "verified_by": "user_1",
                "clause_count": 2,
                "unknown_count": 0,
                "snapshot_hash": "hash_1",
                "tx_hash": "0xabc123",
                "blockchain_link": "https://sepolia.etherscan.io/tx/0xabc123",
            }
        ],
    )

    seed_mock = Mock(return_value=True)
    monkeypatch.setattr(main_module, "seed_verification_baseline", seed_mock)

    monkeypatch.setattr(main_module, "get_cached_result", lambda _doc_hash: None)
    monkeypatch.setattr(
        main_module.model,
        "predict_cuad",
        Mock(return_value=[{"clause_type": "Known Clause", "span": "sample", "confidence": 0.95, "start_idx": 0, "end_idx": 6}]),
    )
    monkeypatch.setattr(main_module, "store_result", lambda **_kwargs: True)
    monkeypatch.setattr(
        main_module,
        "get_verification_state",
        lambda *_args: {
            "doc_hash": "doc_hash_123",
            "unknown_count": 0,
            "show_verify_button": False,
            "is_first_uploader": True,
            "user_taught_count": 0,
            "can_verify": False,
            "message": "Verification already recorded.",
        },
    )

    class DummyUpload:
        def __init__(self) -> None:
            self.filename = "sample.docx"

        async def read(self) -> bytes:
            # Minimal valid DOCX payload builder for this code path is heavy;
            # we patch Document to avoid parsing.
            return b"dummy"

    class DummyDoc:
        paragraphs = [type("P", (), {"text": "Sample contract text"})()]

    monkeypatch.setattr(main_module, "Document", lambda _stream: DummyDoc())

    payload = asyncio.run(main_module.upload_file(file=DummyUpload(), user_id="user_1"))

    assert payload["status"] == "success"
    assert seed_mock.call_count == 1
