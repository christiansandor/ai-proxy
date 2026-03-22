"""
Audit logger.
Writes one JSON object per line to data/audit_logs/log.
Rotates to data/audit_logs/log-1, log-2, … when each file reaches 10 MB.
"""
from __future__ import annotations

import json
import logging.handlers
import os
from datetime import datetime, timezone

_LOG_DIR      = os.path.join('data', 'audit_logs')
_LOG_BASE     = os.path.join(_LOG_DIR, 'log')
_MAX_BYTES    = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 10

os.makedirs(_LOG_DIR, exist_ok=True)

_logger = logging.getLogger('ai_proxy.audit')
_logger.setLevel(logging.INFO)
_logger.propagate = False   # keep audit lines out of stdout

_fh = logging.handlers.RotatingFileHandler(
    _LOG_BASE,
    maxBytes=_MAX_BYTES,
    backupCount=_BACKUP_COUNT,
    encoding='utf-8',
)


def _namer(default_name: str) -> str:
    # RotatingFileHandler produces "…/log.1" — rename to "…/log-1"
    base, sep, ext = default_name.rpartition('.')
    return f'{base}-{ext}' if sep and ext.isdigit() else default_name


_fh.namer = _namer
_logger.addHandler(_fh)


def log(event: str, **fields) -> None:
    entry = {'ts': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    _logger.info(json.dumps(entry))
