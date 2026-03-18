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
