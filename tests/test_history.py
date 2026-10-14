import sqlite3

import pytest

from src.storage.history import HistoryError, HistoryStore, sanitize_filename

SHA = "a" * 64


def add(store, **kw):
    args = dict(filename="cat.png", image_sha256=SHA, decision="likely_ai", score=0.91, score_is_calibrated=True,
                logit=2.3, model_id="m1", model_version="1.0.0", latency_ms=420.0, width=100, height=80,
                image_format="PNG", thumbnail=b"\xff\xd8thumb")
    args.update(kw)
    return store.add(**args)


def test_add_and_get(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    rid = add(store)
    rec = store.get(rid)
    assert rec.id == rid and rec.decision == "likely_ai" and rec.score == pytest.approx(0.91)
    assert rec.score_is_calibrated is True and rec.model_id == "m1" and rec.thumbnail == b"\xff\xd8thumb"
    assert rec.created_at.endswith("+00:00")


def test_history_survives_restart(tmp_path):
    path = tmp_path / "h.db"
    first = HistoryStore(path)
    ids = [add(first, filename=f"{i}.png") for i in range(3)]
    del first                                              # "restart": a brand new store on the same file
    second = HistoryStore(path)
    assert second.count() == 3
    assert {r.id for r in second.list()} == set(ids)


def test_list_is_newest_first_and_paginated(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    ids = [add(store, filename=f"{i}.png") for i in range(5)]
    listed = store.list(limit=2)
    assert [r.id for r in listed] == [ids[4], ids[3]]
    assert [r.id for r in store.list(limit=2, offset=2)] == [ids[2], ids[1]]


def test_delete_single_entry(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    a, b = add(store), add(store)
    assert store.delete(a) is True
    assert store.get(a) is None and store.get(b) is not None
    assert store.delete(a) is False


def test_clear_requires_explicit_confirmation(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    add(store)
    add(store)
    with pytest.raises(HistoryError):
        store.clear()
    with pytest.raises(HistoryError):
        store.clear(confirm="yes")
    assert store.count() == 2
    assert store.clear(confirm=True) == 2
    assert store.count() == 0


@pytest.mark.parametrize("bad", [
    dict(decision="maybe"), dict(score=1.5), dict(score=-0.1), dict(image_sha256="xyz"),
    dict(model_id=""), dict(thumbnail=b"x" * 70_000)])
def test_invalid_input_is_rejected(tmp_path, bad):
    store = HistoryStore(tmp_path / "h.db")
    with pytest.raises(HistoryError):
        add(store, **bad)
    assert store.count() == 0


def test_invalid_ids_are_rejected_without_touching_sql(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    add(store)
    for bad in ["1; DROP TABLE analyses", "' OR '1'='1", "", "../etc/passwd"]:
        with pytest.raises(HistoryError):
            store.get(bad)
        with pytest.raises(HistoryError):
            store.delete(bad)
    assert store.count() == 1


def test_sql_injection_in_filename_is_stored_as_plain_text(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    evil = "x'); DROP TABLE analyses; --.png"
    rid = add(store, filename=evil)
    assert store.get(rid).filename == evil
    with sqlite3.connect(tmp_path / "h.db") as con:
        assert con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 1


def test_filename_sanitising():
    assert sanitize_filename("C:\\Users\\me\\secret\\photo.png") == "photo.png"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("a\x00b\x1fc.png") == "abc.png"
    assert sanitize_filename("") is None and sanitize_filename(None) is None
    assert len(sanitize_filename("n" * 600)) == 255


def test_database_constraints_back_up_python_validation(tmp_path):
    HistoryStore(tmp_path / "h.db")
    with sqlite3.connect(tmp_path / "h.db") as con, pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO analyses (id, created_at, image_sha256, decision, score, score_is_calibrated, model_id, "
                    "model_version) VALUES ('a','t','s','bogus',0.5,1,'m','v')")
