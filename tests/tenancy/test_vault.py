"""Credential vault behaviour, including the cross-tenant binding."""

from __future__ import annotations

import base64
import os
import uuid

import pytest

from cortex.security import vault
from cortex.security.vault import (
    VaultError,
    decrypt_credential,
    encrypt_credential,
    rewrap_data_key,
)

SECRET = '{"type":"service_account","private_key":"-----BEGIN PRIVATE KEY-----"}'


def test_roundtrip() -> None:
    tenant = uuid.uuid4()
    wrapped, ct = encrypt_credential(tenant, "ga4", SECRET)
    assert decrypt_credential(tenant, "ga4", wrapped, ct) == SECRET


def test_ciphertext_does_not_contain_plaintext() -> None:
    tenant = uuid.uuid4()
    _, ct = encrypt_credential(tenant, "ga4", SECRET)
    assert b"private_key" not in ct
    assert b"service_account" not in ct


def test_each_encryption_uses_a_fresh_data_key() -> None:
    tenant = uuid.uuid4()
    w1, c1 = encrypt_credential(tenant, "ga4", SECRET)
    w2, c2 = encrypt_credential(tenant, "ga4", SECRET)
    assert w1 != w2, "data keys must not repeat"
    assert c1 != c2, "identical plaintext must not produce identical ciphertext"


def test_credential_cannot_be_decrypted_by_another_tenant() -> None:
    """Lifting a ciphertext into another tenant's row must fail, not leak."""
    owner, attacker = uuid.uuid4(), uuid.uuid4()
    wrapped, ct = encrypt_credential(owner, "hubspot", SECRET)
    with pytest.raises(VaultError):
        decrypt_credential(attacker, "hubspot", wrapped, ct)


def test_credential_cannot_be_reused_across_providers() -> None:
    tenant = uuid.uuid4()
    wrapped, ct = encrypt_credential(tenant, "hubspot", SECRET)
    with pytest.raises(VaultError):
        decrypt_credential(tenant, "github", wrapped, ct)


def test_tampering_is_detected() -> None:
    tenant = uuid.uuid4()
    wrapped, ct = encrypt_credential(tenant, "slack", SECRET)
    corrupted = bytearray(ct)
    corrupted[-1] ^= 0x01
    with pytest.raises(VaultError):
        decrypt_credential(tenant, "slack", wrapped, bytes(corrupted))


def test_master_key_rotation_preserves_ciphertext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotation re-wraps the data key only; credential ciphertexts stay as they are."""
    tenant = uuid.uuid4()
    old_master = vault._load_master_key()
    wrapped, ct = encrypt_credential(tenant, "ga4", SECRET)

    new_master = os.urandom(32)
    rewrapped = rewrap_data_key(tenant, "ga4", wrapped, old_master, new_master)
    assert rewrapped != wrapped

    monkeypatch.setattr(vault, "_load_master_key", lambda: new_master)
    # Same ciphertext, new wrapped key, still decrypts.
    assert decrypt_credential(tenant, "ga4", rewrapped, ct) == SECRET
    # And the old wrapped key no longer opens under the new master.
    with pytest.raises(VaultError):
        decrypt_credential(tenant, "ga4", wrapped, ct)


def test_unset_master_key_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank key must raise, never fall back to something weak."""
    monkeypatch.setattr(vault, "get_settings", lambda: _FakeSettings(""))
    with pytest.raises(VaultError, match="not set"):
        encrypt_credential(uuid.uuid4(), "ga4", SECRET)


def test_wrong_length_master_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    short = base64.urlsafe_b64encode(os.urandom(16)).decode()
    monkeypatch.setattr(vault, "get_settings", lambda: _FakeSettings(short))
    with pytest.raises(VaultError, match="32 bytes"):
        encrypt_credential(uuid.uuid4(), "ga4", SECRET)


def test_non_base64_master_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vault, "get_settings", lambda: _FakeSettings("not!valid!base64!"))
    with pytest.raises(VaultError, match="base64"):
        encrypt_credential(uuid.uuid4(), "ga4", SECRET)


class _FakeSettings:
    def __init__(self, key: str) -> None:
        self.vault_master_key = key


class TestLabelBinding:
    """F-03. A tenant may hold several credentials per provider, so
    (tenant_id, provider) does not identify a row. Without the label in the AAD,
    swapping two such rows' ciphertexts decrypted cleanly — Cortex would query one
    GA4 property using the other's key while pairing it with the wrong metadata."""

    def test_label_is_required_to_decrypt(self) -> None:
        tenant = uuid.uuid4()
        wrapped, ct = encrypt_credential(tenant, "ga4", SECRET, label="app")
        assert decrypt_credential(tenant, "ga4", wrapped, ct, label="app") == SECRET
        with pytest.raises(VaultError):
            decrypt_credential(tenant, "ga4", wrapped, ct, label="marketing-site")

    def test_rows_cannot_be_swapped_within_a_provider(self) -> None:
        tenant = uuid.uuid4()
        app_wrapped, app_ct = encrypt_credential(tenant, "ga4", "KEY-APP", label="app")
        mkt_wrapped, mkt_ct = encrypt_credential(
            tenant, "ga4", "KEY-MARKETING", label="marketing-site"
        )

        # Each opens under its own label.
        assert decrypt_credential(tenant, "ga4", app_wrapped, app_ct, label="app") == "KEY-APP"
        assert (
            decrypt_credential(tenant, "ga4", mkt_wrapped, mkt_ct, label="marketing-site")
            == "KEY-MARKETING"
        )
        # Neither opens under the other's label.
        with pytest.raises(VaultError):
            decrypt_credential(tenant, "ga4", app_wrapped, app_ct, label="marketing-site")
        with pytest.raises(VaultError):
            decrypt_credential(tenant, "ga4", mkt_wrapped, mkt_ct, label="app")

    def test_default_label_roundtrips(self) -> None:
        """Most credentials use the default label; that path must stay simple."""
        tenant = uuid.uuid4()
        wrapped, ct = encrypt_credential(tenant, "ga4", SECRET)
        assert decrypt_credential(tenant, "ga4", wrapped, ct) == SECRET

    def test_tenant_and_provider_are_still_bound(self) -> None:
        """The label is additional to the earlier bindings, not a replacement."""
        tenant, other = uuid.uuid4(), uuid.uuid4()
        wrapped, ct = encrypt_credential(tenant, "ga4", SECRET, label="app")
        with pytest.raises(VaultError):
            decrypt_credential(other, "ga4", wrapped, ct, label="app")
        with pytest.raises(VaultError):
            decrypt_credential(tenant, "hubspot", wrapped, ct, label="app")


class TestLegacyCompatibility:
    """Credentials sealed before F-03 must keep opening, without a migration."""

    @staticmethod
    def _seal_v1(tenant: uuid.UUID, provider: str, plaintext: str) -> tuple[bytes, bytes]:
        """Reproduce the pre-fix encryption, which bound no label."""
        master = vault._load_master_key()
        aad = vault._legacy_aad(tenant, provider)
        data_key = os.urandom(32)
        return vault._seal(master, data_key, aad), vault._seal(data_key, plaintext.encode(), aad)

    def test_v1_credential_still_decrypts(self) -> None:
        tenant = uuid.uuid4()
        wrapped, ct = self._seal_v1(tenant, "ga4", SECRET)
        assert decrypt_credential(tenant, "ga4", wrapped, ct) == SECRET

    def test_v1_credential_decrypts_under_any_label(self) -> None:
        """v1 carried no label, so it cannot be label-checked. Documented, not
        silently ignored: needs_rewrap() identifies these rows."""
        tenant = uuid.uuid4()
        wrapped, ct = self._seal_v1(tenant, "ga4", SECRET)
        assert decrypt_credential(tenant, "ga4", wrapped, ct, label="anything") == SECRET

    def test_needs_rewrap_flags_legacy_rows(self) -> None:
        tenant = uuid.uuid4()
        legacy_wrapped, _ = self._seal_v1(tenant, "ga4", SECRET)
        assert vault.needs_rewrap(tenant, "ga4", legacy_wrapped) is True

    def test_needs_rewrap_is_false_for_current_rows(self) -> None:
        tenant = uuid.uuid4()
        wrapped, _ = encrypt_credential(tenant, "ga4", SECRET, label="app")
        assert vault.needs_rewrap(tenant, "ga4", wrapped, label="app") is False

    def test_wrong_tenant_still_fails_on_both_paths(self) -> None:
        """The fallback must not become a way to bypass tenant binding."""
        tenant, other = uuid.uuid4(), uuid.uuid4()
        wrapped, ct = self._seal_v1(tenant, "ga4", SECRET)
        with pytest.raises(VaultError):
            decrypt_credential(other, "ga4", wrapped, ct)
