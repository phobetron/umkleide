import json
import os
from pathlib import Path

import pytest

import umkleide.credentials as credentials_module
from umkleide.credentials import (
    BFL_API_KEY_ENV,
    CREDENTIAL_KEY_FIELD,
    CREDENTIALS_FILENAME,
    BflCredentialSource,
    BflCredentialStore,
    CredentialStoreError,
    ResolvedBflCredential,
    resolve_bfl_credential,
)


def test_store_round_trips_a_normalized_key_in_a_private_file(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")

    store.write("  test-key  ")

    assert store.path == tmp_path / "data" / CREDENTIALS_FILENAME
    assert store.read() == "test-key"
    assert json.loads(store.path.read_text()) == {CREDENTIAL_KEY_FIELD: "test-key"}
    if os.name == "posix":
        assert store.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode semantics required")
def test_store_read_repairs_a_regular_credential_file_to_private_mode(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.path.parent.mkdir()
    store.path.write_text(json.dumps({CREDENTIAL_KEY_FIELD: "test-key"}))
    store.path.chmod(0o644)

    assert store.read() == "test-key"
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_store_returns_none_when_no_credential_file_exists(tmp_path: Path) -> None:
    assert BflCredentialStore(tmp_path / "data").read() is None


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        "{}",
        json.dumps({CREDENTIAL_KEY_FIELD: None}),
        json.dumps({CREDENTIAL_KEY_FIELD: "   "}),
        json.dumps({CREDENTIAL_KEY_FIELD: "key", "unexpected": True}),
    ],
)
def test_store_rejects_invalid_credential_files_without_disclosing_contents(
    tmp_path: Path, contents: str
) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.path.parent.mkdir()
    store.path.write_text(contents)

    with pytest.raises(CredentialStoreError) as exc_info:
        store.read()

    assert contents not in str(exc_info.value)


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics differ on Windows")
def test_store_rejects_symlinked_credentials_file(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")
    target = tmp_path / "other.json"
    target.write_text(json.dumps({CREDENTIAL_KEY_FIELD: "test-key"}))
    store.path.parent.mkdir()
    store.path.symlink_to(target)

    with pytest.raises(CredentialStoreError, match="credential"):
        store.read()


def test_store_rejects_a_nonregular_credentials_path(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.path.mkdir(parents=True)

    with pytest.raises(CredentialStoreError, match="credential"):
        store.read()


def test_store_atomic_write_failure_keeps_the_existing_file_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.write("old-key")
    old_contents = store.path.read_bytes()

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("disk interrupted")

    monkeypatch.setattr(credentials_module.os, "replace", fail_replace)

    with pytest.raises(CredentialStoreError, match="credential"):
        store.write("new-key")

    assert store.path.read_bytes() == old_contents
    assert list(store.path.parent.glob(".credentials.*")) == []


def test_clear_is_idempotent(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")

    assert store.clear() is None
    store.write("test-key")
    assert store.clear() is None
    assert store.read() is None


def test_resolve_uses_nonblank_environment_before_stored_credential(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.write("stored-key")

    resolved = resolve_bfl_credential(store, {BFL_API_KEY_ENV: "  environment-key  "})

    assert resolved.source is BflCredentialSource.ENVIRONMENT
    assert resolved.key == "environment-key"
    assert store.read() == "environment-key"


def test_resolve_uses_stored_credential_for_blank_environment(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.write("stored-key")

    resolved = resolve_bfl_credential(store, {BFL_API_KEY_ENV: " \t "})

    assert resolved.source is BflCredentialSource.STORED
    assert resolved.key == "stored-key"


def test_resolve_skips_a_malformed_file_when_environment_has_a_key(tmp_path: Path) -> None:
    store = BflCredentialStore(tmp_path / "data")
    store.path.parent.mkdir()
    store.path.write_text("not json")

    resolved = resolve_bfl_credential(store, {BFL_API_KEY_ENV: "environment-key"})

    assert resolved.source is BflCredentialSource.ENVIRONMENT
    assert resolved.key == "environment-key"


def test_resolve_without_any_credential_is_explicitly_none(tmp_path: Path) -> None:
    resolved = resolve_bfl_credential(BflCredentialStore(tmp_path / "data"), {})

    assert resolved.source is BflCredentialSource.NONE
    assert resolved.key is None


def test_resolved_credential_text_never_discloses_its_key() -> None:
    canary = "credential-canary-must-not-appear"
    resolved = ResolvedBflCredential(canary, BflCredentialSource.ENVIRONMENT)

    assert canary not in repr(resolved)
    assert canary not in str(resolved)
