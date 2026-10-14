"""safe2.seal (SPEC section 10, test 30): encryption round trip, record files, sealed files, the hash
log and `verify` (a modified record and a missing hash-log entry are both detected)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from safe2 import seal
from safe2.tests.helpers import make_key


@pytest.fixture(scope="module")
def keyed(tmp_path_factory):
    root = tmp_path_factory.mktemp("sealroot")
    return str(root), make_key(root)


def test_encrypt_decrypt_round_trip_and_fresh_keys(keyed):
    root, key = keyed
    pub = seal.pub_key(root)
    msg = json.dumps({"eval_return": 12.5, "x": [1, 2, 3]}).encode()
    a, b = seal.encrypt(msg, pub), seal.encrypt(msg, pub)
    assert a != b                                   # a fresh AES key and IV per record
    assert msg not in a and b"12.5" not in a
    assert seal.decrypt(a, key) == msg == seal.decrypt(b, key)
    big = os.urandom(200_000)
    assert seal.decrypt(seal.encrypt(big, pub), key) == big


def test_record_file_truncate_and_verify(tmp_path):
    key = make_key(tmp_path)
    root = str(tmp_path)
    rf = seal.RecordFile(root, "conf", "B5_X_s300")
    for step in (10, 20, 30):
        rf.append(step, {"step": step, "eval_return": step / 10})
    assert [r["step"] for r in seal.read_records(rf.path, key)] == [10, 20, 30]
    assert seal.verify(root) == []
    rf.truncate(20)
    rf.append(30, {"step": 30, "eval_return": -1.0})
    assert [r["eval_return"] for r in seal.read_records(rf.path, key)] == [1.0, 2.0, -1.0]
    assert seal.verify(root) == []                  # the log's LAST entry for index 2 is the new record
    with open(rf.path, "ab") as f:                  # a torn line from a crash is ignored
        f.write(b"40\tAAAA")
    assert len(seal.read_lines(rf.path)) == 3 and seal.verify(root) == []
    # modify one record: verify reports a mismatch
    lines = open(rf.path, "rb").read().split(b"\n")
    step, blob = lines[1].split(b"\t")
    other = seal.encrypt(b'{"step": 20, "eval_return": 99.0}', seal.pub_key(root))
    lines[1] = step + b"\t" + other
    open(rf.path, "wb").write(b"\n".join(lines))
    probs = seal.verify(root)
    assert len(probs) == 1 and "record 1: hash mismatch" in probs[0]
    assert seal.main(["verify", "--root", root]) == 1


def test_sealed_file_and_missing_hash_log_entry(tmp_path):
    key = make_key(tmp_path)
    root = str(tmp_path)
    path = seal.write_file(root, os.path.join("heldout", "run.deployed.json"), {"pooled": {"R": 3.0}})
    assert path == os.path.join(root, "sealed", "heldout", "run.deployed.json.enc")
    assert seal.read_file(path, key) == {"pooled": {"R": 3.0}}
    assert seal.verify(root) == [] and seal.main(["verify", "--root", root]) == 0
    log = os.path.join(root, seal.HASHLOG)
    with open(log) as f:
        entries = f.read().splitlines()
    assert len(entries) == 1 and entries[0].split("\t")[1:4] == ["sealed/heldout/run.deployed.json.enc", "file",
                                                                  "-"]
    open(log, "w").close()                          # drop the entry
    assert seal.verify(root) == ["sealed/heldout/run.deployed.json.enc: no hash-log entry"]


def test_find_root(tmp_path):
    (tmp_path / "results" / "method" / "conf").mkdir(parents=True)
    assert seal.find_root(str(tmp_path / "results" / "method" / "conf")) == \
        os.path.realpath(tmp_path / "results" / "method")
    make_key(tmp_path / "dev")
    (tmp_path / "dev" / "a" / "b").mkdir(parents=True)
    assert seal.find_root(str(tmp_path / "dev" / "a" / "b")) == os.path.realpath(tmp_path / "dev")
    with pytest.raises(FileNotFoundError):
        seal.find_root(str(tmp_path / "elsewhere"))


def test_sealed_sizes_do_not_depend_on_the_outcome(tmp_path):
    """Review (base, finding 1): AES-CTR keeps the plaintext length, and the hash log's n_bytes and the
    files' sizes are public, so rows and files are padded to fixed sizes before sealing."""
    key = make_key(tmp_path)
    root = str(tmp_path)
    rng = np.random.default_rng(0)
    sizes = []
    for lam in (3, 12, 40):
        eps = [{"layout": 70000 + k, "return": float(rng.normal(20, 5)), "cost": float(c)}
               for k, c in enumerate(rng.poisson(lam, 200))]
        path = seal.write_file(root, os.path.join("heldout", f"r{lam}.deployed.json"),
                               {"episodes": eps, "pooled": {"C": float(np.mean([e["cost"] for e in eps]))}})
        sizes.append(os.path.getsize(path))
        assert seal.read_file(path, key)["episodes"] == eps          # the padding is invisible after decryption
    assert len(set(sizes)) == 1
    rf = seal.RecordFile(root, "conf", "B5_X_s300")
    rf.append(10, {"step": 10, "eval_cost": 0.0, "cum_cost": 3.0})
    rf.append(20, {"step": 20, "eval_cost": 105.0, "cum_cost": 123456.0, "eval_violation": 0.35})
    assert len({len(ln) for ln in seal.read_lines(rf.path)}) == 1
    assert [r["step"] for r in seal.read_records(rf.path, key)] == [10, 20]
    with open(os.path.join(root, seal.HASHLOG)) as f:
        n_bytes = {(ln.split("\t")[2], ln.split("\t")[5].strip()) for ln in f}
    assert len({b for k, b in n_bytes if k == "file"}) == 1 and len({b for k, b in n_bytes if k == "record"}) == 1
    with pytest.raises(ValueError) as e:
        seal.write_file(root, os.path.join("heldout", "big.json"), {"x": "y" * (seal.PAD["file"] + 1)})
    assert "y" not in str(e.value) and not any(char.isdigit() for char in str(e.value))


def test_history_lists_every_entry_of_a_path(tmp_path):
    make_key(tmp_path)
    root = str(tmp_path)
    for v in (1.0, 2.0):
        seal.write_file(root, os.path.join("heldout", "a.deployed.json"), {"pooled": {"R": v}})
    h = seal.history(root)
    assert len(h[("sealed/heldout/a.deployed.json.enc", "file", "-")]) == 2 and seal.verify(root) == []
