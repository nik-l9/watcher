"""Credential vault — envelope encryption for third-party credentials.

Design:

  master key (env, 32 bytes)
      wraps ──► per-credential data key (random 32 bytes, stored wrapped)
                    encrypts ──► credential plaintext

A fresh data key per credential row means compromising one ciphertext does not
help with any other, and rotating the master key only requires re-wrapping data
keys rather than re-encrypting every payload.

AES-256-GCM throughout, so tampering with a ciphertext fails decryption rather
than yielding garbage plaintext. The tenant id is bound in as additional
authenticated data: a ciphertext lifted into another tenant's row will not
decrypt.

Plaintext must never be logged, cached, or returned in an API response.
"""

from __future__ import annotations

import base64
import os
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from cortex.config.settings import get_settings

_KEY_BYTES = 32
_NONCE_BYTES = 12


class VaultError(Exception):
    """Raised for configuration or integrity failures. Never contains plaintext."""


def _load_master_key() -> bytes:
    raw = get_settings().vault_master_key
    if not raw:
        raise VaultError(
            "CORTEX_VAULT_MASTER_KEY is not set. Generate one with: "
            'python -c "import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"'
        )
    try:
        key = base64.urlsafe_b64decode(raw)
    except Exception as exc:  # noqa: BLE001 - surfaced as config error
        raise VaultError("CORTEX_VAULT_MASTER_KEY is not valid urlsafe base64") from exc
    if len(key) != _KEY_BYTES:
        raise VaultError(
            f"CORTEX_VAULT_MASTER_KEY must decode to {_KEY_BYTES} bytes, got {len(key)}"
        )
    return key


def _aad(tenant_id: uuid.UUID, provider: str, label: str) -> bytes:
    """Additional authenticated data binding a ciphertext to its exact row.

    The label is included because a tenant may hold several credentials per
    provider — two GA4 properties, for instance — so (tenant_id, provider) alone
    does not identify a row. Without the label, swapping two such rows' ciphertexts
    decrypted cleanly, and Cortex would query one property using the other's key
    while pairing it with the wrong metadata. See docs/security-findings.md F-03.
    """
    return f"cortex:v2:{tenant_id}:{provider}:{label}".encode()


def _legacy_aad(tenant_id: uuid.UUID, provider: str) -> bytes:
    """The v1 AAD, which omitted the label.

    Retained only so credentials encrypted before F-03 keep decrypting. Callers
    re-wrap on first use; nothing writes v1.
    """
    return f"cortex:v1:{tenant_id}:{provider}".encode()


def _seal(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(_NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def _open(key: bytes, blob: bytes, aad: bytes) -> bytes:
    if len(blob) <= _NONCE_BYTES:
        raise VaultError("ciphertext too short")
    nonce, body = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
    try:
        return AESGCM(key).decrypt(nonce, body, aad)
    except InvalidTag as exc:
        raise VaultError(
            "credential failed authentication — wrong master key, wrong tenant, or tampering"
        ) from exc


def encrypt_credential(
    tenant_id: uuid.UUID, provider: str, plaintext: str, *, label: str = "default"
) -> tuple[bytes, bytes]:
    """Encrypt a credential. Returns (wrapped_data_key, ciphertext)."""
    master = _load_master_key()
    aad = _aad(tenant_id, provider, label)

    data_key = os.urandom(_KEY_BYTES)
    try:
        ciphertext = _seal(data_key, plaintext.encode(), aad)
        wrapped_data_key = _seal(master, data_key, aad)
    finally:
        # Best effort: Python cannot guarantee erasure of immutable bytes, but
        # dropping the strong reference at least shortens the window.
        del data_key
    return wrapped_data_key, ciphertext


def decrypt_credential(
    tenant_id: uuid.UUID,
    provider: str,
    wrapped_data_key: bytes,
    ciphertext: bytes,
    *,
    label: str = "default",
) -> str:
    """Decrypt a credential. The result is a secret — do not log or persist it."""
    master = _load_master_key()
    aad = _aad(tenant_id, provider, label)
    try:
        data_key = _open(master, wrapped_data_key, aad)
    except VaultError:
        # Fall back to the pre-F-03 AAD once, so credentials stored before the
        # label was bound in still open. A genuinely wrong tenant, provider or
        # label fails both attempts and still raises.
        legacy = _legacy_aad(tenant_id, provider)
        data_key = _open(master, wrapped_data_key, legacy)
        try:
            return _open(data_key, ciphertext, legacy).decode()
        finally:
            del data_key
    try:
        return _open(data_key, ciphertext, aad).decode()
    finally:
        del data_key


def needs_rewrap(
    tenant_id: uuid.UUID,
    provider: str,
    wrapped_data_key: bytes,
    *,
    label: str = "default",
) -> bool:
    """True when a credential is still sealed under the pre-F-03 AAD.

    Lets an operator find and re-encrypt legacy rows so the compatibility path can
    eventually be removed.
    """
    master = _load_master_key()
    try:
        _open(master, wrapped_data_key, _aad(tenant_id, provider, label))
    except VaultError:
        return True
    return False


def rewrap_data_key(
    tenant_id: uuid.UUID,
    provider: str,
    wrapped_data_key: bytes,
    old_master: bytes,
    new_master: bytes,
    *,
    label: str = "default",
) -> bytes:
    """Rotate the master key without touching credential ciphertexts."""
    aad = _aad(tenant_id, provider, label)
    data_key = _open(old_master, wrapped_data_key, aad)
    try:
        return _seal(new_master, data_key, aad)
    finally:
        del data_key
