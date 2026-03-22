"""
Client token store.
Tokens are UUIDs stored as JSON in data/tokens.json.
The in-memory cache is rebuilt on every write so that validation
is a single dict lookup with no disk I/O on the hot path.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

TOKENS_PATH = Path('data/tokens.json')

_lock  = threading.Lock()
_cache: dict[str, dict] | None = None   # token_id -> record; None = not yet loaded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    if not TOKENS_PATH.exists():
        return {'tokens': []}
    with TOKENS_PATH.open() as f:
        return json.load(f)


def _save(data: dict) -> None:
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOKENS_PATH.open('w') as f:
        json.dump(data, f, indent=2)


def _rebuild(data: dict) -> None:
    global _cache
    _cache = {t['id']: t for t in data['tokens']}


def _ensure_loaded() -> dict[str, dict]:
    global _cache
    if _cache is None:
        _rebuild(_load())
    return _cache  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_token(token_value: str) -> dict | None:
    """Return the record if the value is a known, non-revoked token; else None."""
    with _lock:
        record = _ensure_loaded().get(token_value)
        if record and not record.get('revoked', False):
            return record
        return None


def create_token(label: str) -> dict:
    with _lock:
        data = _load()
        record: dict = {
            'id':         str(uuid.uuid4()),
            'label':      label,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'revoked':    False,
        }
        data['tokens'].append(record)
        _save(data)
        _rebuild(data)
        return record


def revoke_token(token_id: str) -> bool:
    with _lock:
        data = _load()
        for t in data['tokens']:
            if t['id'] == token_id:
                t['revoked'] = True
                _save(data)
                _rebuild(data)
                return True
        return False


def list_tokens() -> list[dict]:
    with _lock:
        return list(_ensure_loaded().values())
