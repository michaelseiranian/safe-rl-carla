"""Sealing of outcome records and files (prereg_method.md section 10; safe2/SPEC.md section 4).

Every outcome row of a training seed >= 300 and every evaluation on the held-out block is encrypted
to the study's public key before it reaches disk: a fresh AES-256-CTR key and IV per record or file,
wrapped with RSA-OAEP (SHA-256) by the OpenSSL 3 command-line tool. Only `safe2.analyze --unblind`
holds the private key. The SHA-256 of every sealed record and file goes to HASHLOG.tsv, which the
lead pushes daily; `verify` checks every sealed record and file against the log's last entry for it.

Fixed sizes. AES-CTR keeps the plaintext length, and HASHLOG.tsv (pushed) and the files' sizes are
public, so every plaintext is padded with trailing spaces to a fixed size before encryption: PAD["record"]
bytes for an outcome row and PAD["file"] bytes for a whole file (json.loads ignores the padding). A
longer plaintext raises instead of being sealed, so no sealed size depends on an outcome.

    python -m safe2.seal verify [--root results/method]

Layout under the sealing root (results/method for the study; any directory holding keys/seal_pub.pem
for development):
    keys/seal_pub.pem                   public key (committed at method-v0)
    sealed/<stage>/<stem>.rows          one line per outcome row: <step>\\t<encrypt(json)>\\n
    sealed/<kind>/<name>.json.enc       whole files (held-out results, PHS quality, R3)
    HASHLOG.tsv                         utc \\t relpath \\t record|file \\t index \\t sha256 \\t n_bytes
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO, "results", "method")
PUB = os.path.join("keys", "seal_pub.pem")
HASHLOG = "HASHLOG.tsv"
_RECORD = re.compile(rb"^(\d+)\t([A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]*)\n$")
PAD = {"record": 8192, "file": 1 << 20}     # fixed plaintext sizes (bytes) of a sealed row and a sealed file


def openssl() -> str:
    exe = os.environ.get("SAFE2_OPENSSL") or "/opt/homebrew/bin/openssl"
    if not os.path.exists(exe):
        raise FileNotFoundError(f"OpenSSL 3 not found at {exe} (set SAFE2_OPENSSL)")
    return exe


def _run(args: list, data: bytes) -> bytes:
    return subprocess.run([openssl()] + args, input=data, capture_output=True, check=True).stdout


_OAEP = ["-pkeyopt", "rsa_padding_mode:oaep", "-pkeyopt", "rsa_oaep_md:sha256", "-pkeyopt", "rsa_mgf1_md:sha256"]


def encrypt(plaintext: bytes, pub_path: str) -> bytes:
    """base64(RSA-OAEP(k + iv)) + b"." + base64(AES-256-CTR_k,iv(plaintext)); the key never touches disk."""
    k, iv = os.urandom(32), os.urandom(16)
    wrapped = _run(["pkeyutl", "-encrypt", "-pubin", "-inkey", pub_path] + _OAEP, k + iv)
    ct = _run(["enc", "-aes-256-ctr", "-K", k.hex(), "-iv", iv.hex()], plaintext)
    return base64.b64encode(wrapped) + b"." + base64.b64encode(ct)


def padded(plaintext: bytes, kind: str) -> bytes:
    """plaintext + trailing spaces up to PAD[kind] bytes; raises (without any value in the message) when
    the plaintext is longer, so a sealed size never depends on an outcome."""
    size = PAD[kind]
    if len(plaintext) > size:
        raise ValueError(f"sealed {kind} plaintext exceeds its fixed size")
    return plaintext + b" " * (size - len(plaintext))


def decrypt(blob: bytes, key_path: str) -> bytes:
    """Reverse encrypt() with the private key (only analyze --unblind and the tests call this)."""
    w64, c64 = blob.strip().split(b".", 1)
    kiv = _run(["pkeyutl", "-decrypt", "-inkey", key_path] + _OAEP, base64.b64decode(w64))
    k, iv = kiv[:32], kiv[32:48]
    return _run(["enc", "-d", "-aes-256-ctr", "-K", k.hex(), "-iv", iv.hex()], base64.b64decode(c64))


# --- where sealed outputs go -----------------------------------------------------------------------
def find_root(path: str) -> str:
    """The sealing root of a run directory: the enclosing results/method if there is one, else the
    nearest ancestor (inclusive) holding keys/seal_pub.pem. Raises if neither exists."""
    p = os.path.realpath(path)
    parts = p.split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        if parts[i - 1] == "results" and parts[i] == "method":
            return os.sep.join(parts[:i + 1])
    while True:
        if os.path.exists(os.path.join(p, PUB)):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            raise FileNotFoundError(f"no sealing root (results/method or a directory with {PUB}) above {path}")
        p = parent


def pub_key(root: str) -> str:
    path = os.path.join(root, PUB)
    if not os.path.exists(path):
        raise FileNotFoundError(f"public sealing key missing: {path}")
    return path


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def log_hash(root: str, relpath: str, kind: str, index, blob: bytes) -> None:
    """Append one line to HASHLOG.tsv under an exclusive lock, fsynced."""
    line = "\t".join([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), relpath, kind,
                      str(index), hashlib.sha256(blob).hexdigest(), str(len(blob))]) + "\n"
    with open(os.path.join(root, HASHLOG), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class RecordFile:
    """The sealed outcome-row file of one training run: sealed/<stage>/<stem>.rows."""

    def __init__(self, root: str, stage: str, stem: str):
        self.root = root
        self.pub = pub_key(root)
        self.rel = os.path.join("sealed", stage, stem + ".rows")
        self.path = os.path.join(root, self.rel)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def count(self) -> int:
        return len(read_lines(self.path)) if os.path.exists(self.path) else 0

    def append(self, step: int, row: dict) -> None:
        blob = encrypt(padded(json.dumps(row).encode(), "record"), self.pub)
        index = self.count()
        with open(self.path, "ab") as f:
            f.write(str(int(step)).encode() + b"\t" + blob + b"\n")
            f.flush()
            os.fsync(f.fileno())
        log_hash(self.root, self.rel, "record", index, blob)

    def truncate(self, last_step: int) -> None:
        """Keep the records with step <= last_step (their step is in clear); drop torn lines."""
        if not os.path.exists(self.path):
            return
        keep = [ln for ln in read_lines(self.path) if int(_RECORD.match(ln).group(1)) <= last_step]
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(b"".join(keep))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


def read_lines(path: str) -> list[bytes]:
    """The well-formed record lines of a .rows file (a torn last line from a crash is ignored)."""
    with open(path, "rb") as f:
        return [ln for ln in f.readlines() if _RECORD.match(ln)]


def read_records(path: str, key_path: str) -> list[dict]:
    """Decrypt every record of a .rows file (analysis and tests only)."""
    return [json.loads(decrypt(_RECORD.match(ln).group(2), key_path)) for ln in read_lines(path)]


def write_file(root: str, rel: str, obj) -> str:
    """Seal a JSON-serialisable object to <root>/sealed/<rel>.enc atomically; returns the path."""
    relpath = os.path.join("sealed", rel + ".enc")
    path = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = encrypt(padded(json.dumps(obj).encode(), "file"), pub_key(root))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path))
    log_hash(root, relpath, "file", "-", blob)
    return path


def read_file(path: str, key_path: str):
    with open(path, "rb") as f:
        return json.loads(decrypt(f.read(), key_path))


def history(root: str, text: str | None = None) -> dict:
    """{(relpath, record|file, index): [sha256, ...]} in log order, from HASHLOG.tsv (or its text)."""
    if text is None:
        log = os.path.join(root, HASHLOG)
        text = open(log).read() if os.path.exists(log) else ""
    out: dict = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 6:
            out.setdefault((parts[1], parts[2], parts[3]), []).append(parts[4])
    return out


def verify(root: str) -> list[str]:
    """Problems found (empty = verified): every record and sealed file must match the hash log's
    last entry for it."""
    last = {k: v[-1] for k, v in history(root).items()}
    problems = []
    sealed = os.path.join(root, "sealed")
    for path in sorted(glob.glob(os.path.join(sealed, "**", "*.rows"), recursive=True)):
        rel = os.path.relpath(path, root)
        for i, ln in enumerate(read_lines(path)):
            h = hashlib.sha256(_RECORD.match(ln).group(2)).hexdigest()
            want = last.get((rel, "record", str(i)))
            if want is None:
                problems.append(f"{rel} record {i}: no hash-log entry")
            elif want != h:
                problems.append(f"{rel} record {i}: hash mismatch")
    for path in sorted(glob.glob(os.path.join(sealed, "**", "*.enc"), recursive=True)):
        rel = os.path.relpath(path, root)
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        want = last.get((rel, "file", "-"))
        if want is None:
            problems.append(f"{rel}: no hash-log entry")
        elif want != h:
            problems.append(f"{rel}: hash mismatch")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["verify"])
    p.add_argument("--root", default=DEFAULT_ROOT)
    a = p.parse_args(argv)
    problems = verify(a.root)
    for x in problems:
        print(x)
    print("verified" if not problems else f"{len(problems)} problem(s)")
    return int(bool(problems))


if __name__ == "__main__":
    sys.exit(main())
