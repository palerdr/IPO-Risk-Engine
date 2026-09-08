"""Read frozen JSON snapshots and hash their uncompressed content."""

import gzip
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    """Hash decompressed JSON content; hash other files as stored."""
    payload = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path):
    payload = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(payload)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload)
    temporary.replace(path)
