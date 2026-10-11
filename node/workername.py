"""Pool worker names. Stdlib only: node/agent.py carries a verbatim copy (tests keep both in step).

Scheme: <prefix><id> where prefix is the provider's 2-3 letter tag (salad sa, vast va) and id is the
machine/instance id lowercased with everything but [a-z0-9] removed, truncated so the whole name is <= MAX_LEN.
Pearl's pool (Kryptex) accepts [A-Za-z0-9]{1,32}; we stay at 30 to leave room for a pool-side suffix.
"""
from __future__ import annotations

import re

MAX_LEN = 30


def make_worker(prefix: str, raw_id: str) -> str:
    ident = re.sub(r"[^a-z0-9]", "", raw_id.lower())
    pre = re.sub(r"[^a-z0-9]", "", prefix.lower())
    if not ident:
        raise ValueError("empty instance id")
    return (pre + ident)[:MAX_LEN]
