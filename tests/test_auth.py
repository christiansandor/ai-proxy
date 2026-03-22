"""
Tests for auth.py — token creation, validation, revocation, persistence.
Each test gets an isolated tmp_path so nothing bleeds between runs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Point auth at a temp file and reset the in-memory cache for every test."""
    import auth
    auth.TOKENS_PATH = tmp_path / 'tokens.json'
    with auth._lock:
        auth._cache = None
    yield


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

class TestCreateToken:

    def test_returns_record_with_sk_prefixed_token(self):
        import auth
        record = auth.create_token('my-app')
        assert re.match(
            r'^sk-[0-9a-f]{64}$',
            record['id'],
        )

    def test_label_preserved(self):
        import auth
        assert auth.create_token('hello')['label'] == 'hello'

    def test_not_revoked_on_creation(self):
        import auth
        assert auth.create_token('fresh')['revoked'] is False

    def test_has_iso_timestamp(self):
        import auth
        record = auth.create_token('ts-check')
        assert 'created_at' in record
        assert 'T' in record['created_at']   # ISO 8601

    def test_multiple_tokens_get_different_ids(self):
        import auth
        a = auth.create_token('a')
        b = auth.create_token('b')
        assert a['id'] != b['id']

    def test_persisted_to_disk(self):
        import auth
        record = auth.create_token('persisted')
        raw = json.loads(auth.TOKENS_PATH.read_text())
        assert record['id'] in [t['id'] for t in raw['tokens']]


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------

class TestValidateToken:

    def test_valid_token_returns_record(self):
        import auth
        record = auth.create_token('valid')
        result = auth.validate_token(record['id'])
        assert result is not None
        assert result['id'] == record['id']

    def test_unknown_token_returns_none(self):
        import auth
        assert auth.validate_token('not-a-real-token') is None

    def test_empty_string_returns_none(self):
        import auth
        assert auth.validate_token('') is None

    def test_revoked_token_returns_none(self):
        import auth
        record = auth.create_token('to-revoke')
        auth.revoke_token(record['id'])
        assert auth.validate_token(record['id']) is None

    def test_valid_token_served_from_cache(self):
        """After creation the cache is warm — validate must not need the file."""
        import auth
        record = auth.create_token('cached')
        auth.TOKENS_PATH.unlink()   # prove we're not hitting disk
        assert auth.validate_token(record['id']) is not None


# ---------------------------------------------------------------------------
# Token revocation
# ---------------------------------------------------------------------------

class TestRevokeToken:

    def test_revoke_existing_returns_true(self):
        import auth
        record = auth.create_token('revoke-me')
        assert auth.revoke_token(record['id']) is True

    def test_revoke_nonexistent_returns_false(self):
        import auth
        assert auth.revoke_token('00000000-0000-0000-0000-000000000000') is False

    def test_revoked_flag_written_to_disk(self):
        import auth
        record = auth.create_token('write-check')
        auth.revoke_token(record['id'])
        raw = json.loads(auth.TOKENS_PATH.read_text())
        entry = next(t for t in raw['tokens'] if t['id'] == record['id'])
        assert entry['revoked'] is True

    def test_other_tokens_unaffected(self):
        import auth
        a = auth.create_token('keep')
        b = auth.create_token('kill')
        auth.revoke_token(b['id'])
        assert auth.validate_token(a['id']) is not None


# ---------------------------------------------------------------------------
# List tokens
# ---------------------------------------------------------------------------

class TestListTokens:

    def test_empty_store_returns_empty_list(self):
        import auth
        assert auth.list_tokens() == []

    def test_includes_revoked_tokens(self):
        import auth
        a = auth.create_token('a')
        b = auth.create_token('b')
        auth.revoke_token(b['id'])
        ids = {t['id'] for t in auth.list_tokens()}
        assert ids == {a['id'], b['id']}

    def test_count_matches_created(self):
        import auth
        for i in range(5):
            auth.create_token(f'token-{i}')
        assert len(auth.list_tokens()) == 5


# ---------------------------------------------------------------------------
# Persistence across cache resets
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_tokens_survive_cache_reset(self):
        import auth
        record = auth.create_token('survivor')
        with auth._lock:
            auth._cache = None
        result = auth.validate_token(record['id'])
        assert result is not None
        assert result['label'] == 'survivor'

    def test_revocation_survives_cache_reset(self):
        import auth
        record = auth.create_token('revived')
        auth.revoke_token(record['id'])
        with auth._lock:
            auth._cache = None
        assert auth.validate_token(record['id']) is None
