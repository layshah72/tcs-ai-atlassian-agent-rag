from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


class AuthenticationError(RuntimeError):
    """Raised when Atlassian authentication cannot be configured or completed."""


class DPAPIOAuthTokenStorage:
    """Persist OAuth data encrypted to the current Windows user with DPAPI."""

    def __init__(self, storage_key: str):
        if os.name != "nt":
            raise AuthenticationError("OAuth token persistence currently requires Windows DPAPI.")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise AuthenticationError("LOCALAPPDATA is unavailable; cannot select a private OAuth storage folder.")
        namespace = hashlib.sha256(storage_key.encode("utf-8")).hexdigest()[:16]
        self.directory = Path(local_app_data) / "TCS-Confluent-chatbot" / "oauth" / namespace

    def _path(self, item: str) -> Path:
        if item not in {"tokens", "client_info"}:
            raise ValueError("Unsupported OAuth storage item")
        return self.directory / f"{item}.dpapi"

    @staticmethod
    def _crypt(data: bytes, *, protect: bool, entropy: bytes) -> bytes:
        """Encrypt/decrypt bytes with current-user DPAPI, without UI prompts."""
        if os.name != "nt":
            raise AuthenticationError("Windows DPAPI is only available on Windows.")
        from ctypes import wintypes

        class DataBlob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

        def blob_for(value: bytes) -> tuple[DataBlob, Any]:
            buffer = ctypes.create_string_buffer(value)
            return DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer

        input_blob, input_buffer = blob_for(data)
        entropy_blob, entropy_buffer = blob_for(entropy)
        output_blob = DataBlob()
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        operation = getattr(crypt32, "CryptProtectData" if protect else "CryptUnprotectData")
        operation.restype = wintypes.BOOL
        local_free = kernel32.LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        common_args = [ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.POINTER(DataBlob),
                       ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob)]
        if protect:
            operation.argtypes = [ctypes.POINTER(DataBlob), wintypes.LPCWSTR, ctypes.POINTER(DataBlob),
                                  ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob)]
            success = operation(ctypes.byref(input_blob), "TCS Confluent Chatbot OAuth",
                                ctypes.byref(entropy_blob), None, None, 0x1, ctypes.byref(output_blob))
        else:
            operation.argtypes = common_args
            success = operation(ctypes.byref(input_blob), None, ctypes.byref(entropy_blob),
                                None, None, 0x1, ctypes.byref(output_blob))
        _ = input_buffer, entropy_buffer
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            local_free(output_blob.pbData)

    def _read_value(self, item: str) -> str | None:
        try:
            encrypted = self._path(item).read_bytes()
        except FileNotFoundError:
            return None
        entropy = f"TCS-Confluent-chatbot:{item}:v1".encode()
        try:
            return self._crypt(encrypted, protect=False, entropy=entropy).decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AuthenticationError(
                "Could not decrypt saved Atlassian OAuth data for this Windows account. "
                "Remove the app's OAuth files under %LOCALAPPDATA%\\TCS-Confluent-chatbot\\oauth and sign in again."
            ) from exc

    def _write_value(self, item: str, value: str) -> None:
        entropy = f"TCS-Confluent-chatbot:{item}:v1".encode()
        try:
            encrypted = self._crypt(value.encode(), protect=True, entropy=entropy)
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f"{item}-", suffix=".tmp", delete=False) as handle:
                handle.write(encrypted)
                temp_path = Path(handle.name)
            os.replace(temp_path, self._path(item))
        except OSError as exc:
            raise AuthenticationError(
                "Windows could not save encrypted Atlassian OAuth data in the current user's Local AppData folder. "
                "Check that the folder is writable and that Streamlit runs under the same Windows account."
            ) from exc

    async def get_tokens(self) -> OAuthToken | None:
        value = await asyncio.to_thread(self._read_value, "tokens")
        return OAuthToken.model_validate_json(value) if value else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await asyncio.to_thread(self._write_value, "tokens", tokens.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        value = await asyncio.to_thread(self._read_value, "client_info")
        return OAuthClientInformationFull.model_validate_json(value) if value else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(self._write_value, "client_info", client_info.model_dump_json())


# Historical name kept for integrations; the implementation uses DPAPI, not keyring.
KeyringOAuthTokenStorage = DPAPIOAuthTokenStorage


class FileOAuthTokenStorage:
    """Cross-platform OAuth token storage encrypted with Fernet (cryptography package).

    Set OAUTH_STORAGE_SECRET to a stable secret (32+ chars) in the environment.
    If unset, a process-local ephemeral key is used (tokens lost on restart).
    """

    def __init__(self, storage_key: str) -> None:
        from cryptography.fernet import Fernet

        app_data = os.getenv("LOCALAPPDATA") or os.getenv("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        namespace = hashlib.sha256(storage_key.encode()).hexdigest()[:16]
        self.directory = Path(app_data) / "TCS-Confluent-chatbot" / "oauth" / namespace
        secret = os.getenv("OAUTH_STORAGE_SECRET", "")
        if secret:
            key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
        else:
            key = Fernet.generate_key()
        self._fernet = Fernet(key)

    def _path(self, item: str) -> Path:
        if item not in {"tokens", "client_info"}:
            raise ValueError("Unsupported OAuth storage item")
        return self.directory / f"{item}.enc"

    def _read_value(self, item: str) -> str | None:
        try:
            encrypted = self._path(item).read_bytes()
        except FileNotFoundError:
            return None
        try:
            return self._fernet.decrypt(encrypted).decode("utf-8")
        except Exception as exc:
            raise AuthenticationError(
                "Could not decrypt saved Atlassian OAuth data. "
                "Delete the app's OAuth files and sign in again."
            ) from exc

    def _write_value(self, item: str, value: str) -> None:
        encrypted = self._fernet.encrypt(value.encode())
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f"{item}-", suffix=".tmp", delete=False) as fh:
            fh.write(encrypted)
            tmp = Path(fh.name)
        os.replace(tmp, self._path(item))

    async def get_tokens(self) -> OAuthToken | None:
        value = await asyncio.to_thread(self._read_value, "tokens")
        return OAuthToken.model_validate_json(value) if value else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await asyncio.to_thread(self._write_value, "tokens", tokens.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        value = await asyncio.to_thread(self._read_value, "client_info")
        return OAuthClientInformationFull.model_validate_json(value) if value else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(self._write_value, "client_info", client_info.model_dump_json())


def make_oauth_token_storage(storage_key: str) -> DPAPIOAuthTokenStorage | FileOAuthTokenStorage:
    """Return the platform-appropriate OAuth token storage backend."""
    if os.name == "nt":
        return DPAPIOAuthTokenStorage(storage_key)
    return FileOAuthTokenStorage(storage_key)
