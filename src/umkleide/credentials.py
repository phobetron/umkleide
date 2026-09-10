"""Small local credential store for the BFL API key."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

BFL_API_KEY_ENV = "BFL_API_KEY"
CREDENTIALS_FILENAME = "credentials.json"
CREDENTIAL_KEY_FIELD = "bfl_api_key"
_MAX_KEY_LENGTH = 4096


class CredentialStoreError(RuntimeError):
    """A sanitized credential-store failure that never includes credential content."""


class BflCredentialSource(str, Enum):
    ENVIRONMENT = "environment"
    STORED = "stored"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ResolvedBflCredential:
    """A normalized key and its active source."""

    key: str | None = field(repr=False)
    source: BflCredentialSource

    def __post_init__(self) -> None:
        if (self.key is None) != (self.source is BflCredentialSource.NONE):
            raise ValueError("credential key and source must agree")


def normalize_credential_key(value: str | None) -> str | None:
    """Normalize a supplied key without ever including it in an error."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise CredentialStoreError("credential value is invalid")
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > _MAX_KEY_LENGTH or any(
        unicodedata.category(character) == "Cc" for character in normalized
    ):
        raise CredentialStoreError("credential value is invalid")
    return normalized


class BflCredentialStore:
    """Read and atomically replace the one local BFL API-key record."""

    def __init__(self, data_root: Path) -> None:
        self.path = Path(data_root) / CREDENTIALS_FILENAME

    def read(self) -> str | None:
        """Return the stored key, or ``None`` when no record exists."""
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CredentialStoreError("credential storage is unavailable") from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise CredentialStoreError("credential storage is invalid")
        if os.name == "posix":
            try:
                self.path.chmod(0o600)
            except OSError as exc:
                raise CredentialStoreError("credential storage is unavailable") from exc
        try:
            payload = self.path.read_text(encoding="utf-8")
            data = json.loads(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CredentialStoreError("credential storage is invalid") from exc
        if (
            not isinstance(data, dict)
            or set(data) != {CREDENTIAL_KEY_FIELD}
            or not isinstance(data.get(CREDENTIAL_KEY_FIELD), str)
        ):
            raise CredentialStoreError("credential storage is invalid")
        key = normalize_credential_key(data[CREDENTIAL_KEY_FIELD])
        if key is None:
            raise CredentialStoreError("credential storage is invalid")
        return key

    def write(self, key: str) -> None:
        """Atomically replace the local record with a normalized nonblank key."""
        normalized = normalize_credential_key(key)
        if normalized is None:
            raise CredentialStoreError("credential value is invalid")
        try:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            if os.name == "posix":
                self.path.parent.chmod(0o700)
        except OSError as exc:
            raise CredentialStoreError("credential storage is unavailable") from exc
        self._reject_unsafe_existing_target()
        temporary_path: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(prefix=".credentials-", dir=self.path.parent)
            temporary_path = Path(raw_path)
            if os.name == "posix":
                os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(
                    json.dumps({CREDENTIAL_KEY_FIELD: normalized}, separators=(",", ":")).encode(
                        "utf-8"
                    )
                    + b"\n"
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        except (OSError, UnicodeError) as exc:
            raise CredentialStoreError("credential storage could not be updated") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def clear(self) -> None:
        """Remove the record when it exists."""
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CredentialStoreError("credential storage is unavailable") from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise CredentialStoreError("credential storage is invalid")
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CredentialStoreError("credential storage could not be updated") from exc

    def _reject_unsafe_existing_target(self) -> None:
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CredentialStoreError("credential storage is unavailable") from exc
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise CredentialStoreError("credential storage is invalid")


def resolve_bfl_credential(
    store: BflCredentialStore, environ: Mapping[str, str]
) -> ResolvedBflCredential:
    """Resolve and, for an environment key, persist the active credential."""
    environment_key = normalize_credential_key(environ.get(BFL_API_KEY_ENV))
    if environment_key is not None:
        store.write(environment_key)
        return ResolvedBflCredential(environment_key, BflCredentialSource.ENVIRONMENT)
    stored_key = store.read()
    if stored_key is not None:
        return ResolvedBflCredential(stored_key, BflCredentialSource.STORED)
    return ResolvedBflCredential(None, BflCredentialSource.NONE)
