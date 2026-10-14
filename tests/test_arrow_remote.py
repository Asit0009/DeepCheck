import numpy as np
import pyarrow as pa
import pytest

from src.data.arrow_remote import ArrowShard, LocalSource


def _write_stream(path, n_batches=4, rows=7, seed=0):
    rng = np.random.default_rng(seed)
    schema = pa.schema([
        ("image_path", pa.string()),
        ("image", pa.binary()),
        ("label", pa.int64()),
        ("generator", pa.int64()),
    ])
    expected = []
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
        k = 0
        for b in range(n_batches):
            paths, blobs, labels, gens = [], [], [], []
            for _ in range(rows):
                paths.append(f"G/test/{'nature' if k < 10 else 'ai'}/img_{k:04d}.png")
                blobs.append(rng.bytes(int(rng.integers(50, 400))))
                labels.append(0 if k < 10 else 1)
                gens.append(3)
                k += 1
            writer.write_batch(pa.record_batch(
                [pa.array(paths), pa.array(blobs, type=pa.binary()), pa.array(labels, pa.int64()),
                 pa.array(gens, pa.int64())], schema=schema))
            expected.append((paths, blobs, labels))
    return expected


@pytest.fixture()
def shard_and_expected(tmp_path):
    path = tmp_path / "shard.arrow"
    expected = _write_stream(path)
    shard = ArrowShard(LocalSource(str(path)))
    shard.build_index()
    return shard, expected


def test_index_finds_every_batch(shard_and_expected):
    shard, expected = shard_and_expected
    assert len(shard.batches) == len(expected)
    assert all(b.num_rows == 7 for b in shard.batches)


def test_columns_match_full_read(shard_and_expected):
    shard, expected = shard_and_expected
    for batch, (paths, blobs, labels) in zip(shard.batches, expected):
        assert shard.read_string_column(batch, "image_path") == paths
        assert list(shard.read_int_column(batch, "label")) == labels
        rows = [0, 3, 6]
        got = shard.read_binary_rows(batch, "image", rows)
        assert [got[r] for r in rows] == [blobs[r] for r in rows]


def test_partial_read_agrees_with_pyarrow_table(shard_and_expected):
    shard, expected = shard_and_expected
    table = shard.read_batch_table(shard.batches[2])
    assert table.num_rows == 7
    assert table.column("image_path").to_pylist() == expected[2][0]


def test_cached_index_roundtrip(shard_and_expected):
    shard, _ = shard_and_expected
    rebuilt = ArrowShard.from_cached(
        shard.source, 0, shard.schema_b64(), [b.to_json() for b in shard.batches])
    assert rebuilt.column_buffers() == shard.column_buffers()
    assert rebuilt.batches[1].buffers == shard.batches[1].buffers
