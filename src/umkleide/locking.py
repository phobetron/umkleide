"""Cross-process locks for shared local data operations."""

from hashlib import sha256
from pathlib import Path

from filelock import FileLock


def media_lock(root: Path) -> FileLock:
    """Serialize synchronous catalog, media, and maintenance operations."""
    return FileLock(root / ".media.lock", mode=0o600)


def generation_lock(root: Path, generation_id: str) -> FileLock:
    """Nonblocking poll ownership with a bounded set of persistent lock files."""
    bucket = sha256(generation_id.encode()).hexdigest()[:2]
    return FileLock(root / f".generation-{bucket}.lock", timeout=0, thread_local=False, mode=0o600)
