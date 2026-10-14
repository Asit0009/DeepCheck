from __future__ import annotations

import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

VALID_DECISIONS = {"likely_ai", "likely_real", "inconclusive"}
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILENAME_LEN = 255
MAX_THUMBNAIL_BYTES = 60_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    filename            TEXT,
    image_sha256        TEXT NOT NULL,
    decision            TEXT NOT NULL CHECK (decision IN ('likely_ai','likely_real','inconclusive')),
    score               REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    score_is_calibrated INTEGER NOT NULL CHECK (score_is_calibrated IN (0,1)),
    logit               REAL,
    model_id            TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    latency_ms          REAL,
    width               INTEGER,
    height              INTEGER,
    image_format        TEXT,
    thumbnail           BLOB
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
"""


class HistoryError(ValueError):
    pass


def sanitize_filename(name: str | None) -> str | None:
    if not name:
        return None
    name = name.replace("\\", "/").rsplit("/", 1)[-1]  # remove directory path
    name = "".join(ch for ch in name if ch.isprintable())  # remove non-printable characters
    name = name.strip()[:MAX_FILENAME_LEN]  # limit string length
    return name or None


@dataclass
class HistoryRecord:
    id: str
    created_at: str
    filename: str | None
    image_sha256: str
    decision: str
    score: float
    score_is_calibrated: bool
    logit: float | None
    model_id: str
    model_version: str
    latency_ms: float | None
    width: int | None
    height: int | None
    image_format: str | None
    thumbnail: bytes | None = field(default=None, repr=False)


_COLUMNS = (
    "id, created_at, filename, image_sha256, decision, score, score_is_calibrated, logit, model_id, "
    "model_version, latency_ms, width, height, image_format, thumbnail"
)


def _row_to_record(row: sqlite3.Row) -> HistoryRecord:
    return HistoryRecord(
        id=row["id"],
        created_at=row["created_at"],
        filename=row["filename"],
        image_sha256=row["image_sha256"],
        decision=row["decision"],
        score=row["score"],
        score_is_calibrated=bool(row["score_is_calibrated"]),
        logit=row["logit"],
        model_id=row["model_id"],
        model_version=row["model_version"],
        latency_ms=row["latency_ms"],
        width=row["width"],
        height=row["height"],
        image_format=row["image_format"],
        thumbnail=row["thumbnail"],
    )


class HistoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)  # path to sqlite database file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)  # create schema if missing

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def add(
        self,
        *,
        filename: str | None,
        image_sha256: str,
        decision: str,
        score: float,
        score_is_calibrated: bool,
        logit: float | None,
        model_id: str,
        model_version: str,
        latency_ms: float | None = None,
        width: int | None = None,
        height: int | None = None,
        image_format: str | None = None,
        thumbnail: bytes | None = None,
    ) -> str:
        if decision not in VALID_DECISIONS:
            raise HistoryError(f"invalid decision: {decision}")
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
            raise HistoryError("score must be between 0 and 1")
        if not _SHA_RE.match(image_sha256 or ""):
            raise HistoryError("image_sha256 must be a valid 64-char sha256 hash")
        if not model_id or not model_version:
            raise HistoryError("model_id and model_version are required")
        if thumbnail is not None and len(thumbnail) > MAX_THUMBNAIL_BYTES:
            raise HistoryError("thumbnail is too large")

        analysis_id = uuid.uuid4().hex  # generate unique id
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as con:
            con.execute(
                f"INSERT INTO analyses ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    created,
                    sanitize_filename(filename),
                    image_sha256,
                    decision,
                    float(score),
                    int(bool(score_is_calibrated)),
                    None if logit is None else float(logit),
                    model_id,
                    model_version,
                    latency_ms,
                    width,
                    height,
                    image_format,
                    thumbnail,
                ),
            )
        return analysis_id

    def get(self, analysis_id: str) -> HistoryRecord | None:
        if not _ID_RE.match(analysis_id or ""):
            raise HistoryError("invalid analysis id")
        with self._connect() as con:
            row = con.execute(f"SELECT {_COLUMNS} FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return _row_to_record(row) if row else None

    def list(self, limit: int = 50, offset: int = 0) -> list[HistoryRecord]:
        limit, offset = int(limit), int(offset)
        if limit < 1 or limit > 500 or offset < 0:
            raise HistoryError("limit must be between 1 and 500")
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {_COLUMNS} FROM analyses ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])

    def delete(self, analysis_id: str) -> bool:
        if not _ID_RE.match(analysis_id or ""):
            raise HistoryError("invalid analysis id")
        with self._connect() as con:
            cur = con.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return cur.rowcount > 0

    def clear(self, confirm: bool = False) -> int:
        if confirm is not True:
            raise HistoryError("clear() requires confirm=True")
        with self._connect() as con:
            cur = con.execute("DELETE FROM analyses")  # remove all rows
        return cur.rowcount
