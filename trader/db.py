"""
Minimal Supabase (PostgREST) client for the worker.

Uses plain HTTP instead of supabase-py to keep the cron install small. When
SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set, the client runs "offline":
writes are logged instead of sent, and agent state lives in memory. That lets
you dry-run the whole pipeline before the database exists.
"""

import json
import math
from datetime import date, datetime, timezone
from uuid import UUID

import numpy as np
import requests

DEFAULT_STATE = {
    "id": 1, "enabled": True, "halted": False, "halted_reason": None,
    "initial_value": None, "peak_value": None, "trade_step": 0,
    "started_at": None, "last_cycle_at": None,
}


def _json_default(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, UUID):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"Cannot serialize {type(v)}")


def _clean(v):
    """NaN/inf are not valid JSON; store them as null."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


class Database:
    def __init__(self, url: str = None, service_key: str = None, write: bool = True):
        self.online = bool(url and service_key)
        self.write = write
        self._state = dict(DEFAULT_STATE)
        if self.online:
            self.base = url.rstrip("/") + "/rest/v1"
            self.session = requests.Session()
            self.session.headers.update({
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
            })

    def _log_skip(self, verb, table, n=None):
        why = "offline" if not self.online else "dry run"
        print(f"  [db {why}] {verb} {table}" + (f" ({n} rows)" if n is not None else ""))

    def _request(self, method, table, params=None, body=None, prefer=None):
        headers = {"Prefer": prefer} if prefer else {}
        data = json.dumps(_clean(body), default=_json_default) if body is not None else None
        r = self.session.request(method, f"{self.base}/{table}", params=params, data=data,
                                 headers=headers, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase {method} {table} failed ({r.status_code}): {r.text[:500]}")
        return r.json() if r.text else None

    def select(self, table, **params):
        if not self.online:
            return []
        return self._request("GET", table, params=params)

    def upsert(self, table, rows, on_conflict=None, ignore_duplicates=False):
        if not rows:
            return
        if not (self.online and self.write):
            return self._log_skip("upsert", table, len(rows))
        resolution = "ignore-duplicates" if ignore_duplicates else "merge-duplicates"
        params = {"on_conflict": on_conflict} if on_conflict else None
        for i in range(0, len(rows), 500):
            self._request("POST", table, params=params, body=rows[i:i + 500],
                          prefer=f"resolution={resolution},return=minimal")

    def delete(self, table, **filters):
        if not (self.online and self.write):
            return self._log_skip("delete from", table)
        self._request("DELETE", table, params=filters, prefer="return=minimal")

    # agent_state is a single row (id = 1)
    def get_state(self) -> dict:
        if self.online:
            rows = self.select("agent_state", id="eq.1")
            if rows:
                self._state = rows[0]
        return dict(self._state)

    def update_state(self, **fields):
        self._state.update(fields)
        if not (self.online and self.write):
            return
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._request("PATCH", "agent_state", params={"id": "eq.1"}, body=fields, prefer="return=minimal")
