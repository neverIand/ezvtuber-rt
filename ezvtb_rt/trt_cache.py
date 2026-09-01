"""Disk-cache helpers for TensorRT-RTX engines and runtime kernels.

This module intentionally has no CUDA or TensorRT imports so cache naming and
file handling can be tested without initializing a GPU runtime.
"""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterator, List, Optional, Tuple, Union


PathLike = Union[str, os.PathLike]
ENGINE_CACHE_SCHEMA = "2"
RUNTIME_CACHE_SCHEMA = "1"
ENGINE_CACHE_ENV = "EZVTB_TRT_CACHE_DIR"
DEFAULT_CACHE_APP_DIR = "EasyVtuber"
DEFAULT_CACHE_DIR_NAME = "trt-cache"
LEGACY_CACHE_DIR_NAME = "ezvtuber_rt_engines"
HASH_MANIFEST_SCHEMA = 1
HASH_MANIFEST_NAME = "hash-manifest-v1.json"

# Any change here can alter the serialized engine and therefore must produce a
# new cache key. It mirrors build_engine() in trt_utils.py.
BUILDER_CONFIG_FINGERPRINT = (
    "workspace=4GiB;tiling=full;compute=current;"
    "dynamic-batch=min1-opt4-max4;dynamic-opt-level=5"
)


@lru_cache(maxsize=128)
def _sha256_for_file_state(
    path: str,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    del size, mtime_ns, ctime_ns  # File state is part of the memoization key.
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_manifest(cache_dir: Path) -> dict:
    manifest_path = cache_dir / HASH_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema") != HASH_MANIFEST_SCHEMA:
        return {}
    files = payload.get("files")
    return files if isinstance(files, dict) else {}


def _valid_manifest_digest(entry, stat) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    digest = entry.get("sha256")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        return None
    expected_state = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    stored_state = (
        entry.get("size"),
        entry.get("mtime_ns"),
        entry.get("ctime_ns"),
    )
    return digest if stored_state == expected_state else None


def _store_hash_manifest_entry(
    cache_dir: Path,
    key: str,
    stat,
    digest: str,
) -> None:
    try:
        files = _load_hash_manifest(cache_dir)
        files[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "sha256": digest,
        }
        payload = json.dumps(
            {"schema": HASH_MANIFEST_SCHEMA, "files": files},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        atomic_write(cache_dir / HASH_MANIFEST_NAME, payload)
    except (OSError, TypeError, ValueError):
        # The manifest is a startup optimization only.  Hashing remains the
        # source of truth when it cannot be persisted.
        pass


def file_sha256(
    path: PathLike,
    manifest_cache_dir: Optional[PathLike] = None,
) -> str:
    """Return a content digest with optional cross-process memoization."""
    resolved = str(Path(path).resolve())
    stat = os.stat(resolved)
    manifest_dir = (
        None if manifest_cache_dir is None else Path(manifest_cache_dir)
    )
    manifest_key = os.path.normcase(resolved)
    if manifest_dir is not None:
        digest = _valid_manifest_digest(
            _load_hash_manifest(manifest_dir).get(manifest_key),
            stat,
        )
        if digest is not None:
            return digest

    digest = _sha256_for_file_state(
        resolved,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    current_stat = os.stat(resolved)
    original_state = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    current_state = (
        current_stat.st_size,
        current_stat.st_mtime_ns,
        current_stat.st_ctime_ns,
    )
    if current_state != original_state:
        return file_sha256(path, manifest_cache_dir)
    if manifest_dir is not None:
        _store_hash_manifest_entry(
            manifest_dir,
            manifest_key,
            current_stat,
            digest,
        )
    return digest


class CacheInUseError(RuntimeError):
    """Raised when a TensorRT engine build lock makes cleanup unsafe."""


def get_default_cache_dir() -> Path:
    """Return a persistent per-user cache directory for the current OS."""
    cache_root = os.environ.get("LOCALAPPDATA")
    if cache_root:
        return Path(cache_root) / DEFAULT_CACHE_APP_DIR / DEFAULT_CACHE_DIR_NAME

    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / DEFAULT_CACHE_APP_DIR / DEFAULT_CACHE_DIR_NAME
    return Path.home() / ".cache" / DEFAULT_CACHE_APP_DIR / DEFAULT_CACHE_DIR_NAME


def get_legacy_cache_dir() -> Path:
    """Return the temporary cache location used by older EasyVtuber builds."""
    return Path(tempfile.gettempdir()) / LEGACY_CACHE_DIR_NAME


def resolve_cache_dir(cache_dir: Optional[PathLike] = None) -> Path:
    """Return the configured cache directory without creating it."""
    if cache_dir is None:
        cache_dir = os.environ.get(ENGINE_CACHE_ENV)
    if not cache_dir:
        cache_dir = get_default_cache_dir()
    return Path(cache_dir)


def get_cache_dir(cache_dir: Optional[PathLike] = None) -> Path:
    uses_default_cache = cache_dir is None and not os.environ.get(ENGINE_CACHE_ENV)
    result = resolve_cache_dir(cache_dir)
    result.mkdir(parents=True, exist_ok=True)
    if uses_default_cache:
        migrate_legacy_cache(result)
    return result


def _is_managed_cache_file(path: Path) -> bool:
    """Return whether ``path`` is a file created by this cache module."""
    if not path.is_file():
        return False
    name = path.name
    if name.endswith(".trt") or name.endswith(".runtime.cache"):
        return True
    if name == HASH_MANIFEST_NAME:
        return True
    return (
        name.startswith(".")
        and name.endswith(".tmp")
        and (
            ".trt." in name
            or ".runtime.cache." in name
            or HASH_MANIFEST_NAME in name
        )
    )


def list_cache_files(cache_dir: Optional[PathLike] = None) -> List[Path]:
    """List managed cache files in the cache directory, without recursion."""
    directory = resolve_cache_dir(cache_dir)
    try:
        return sorted(
            (path for path in directory.iterdir() if _is_managed_cache_file(path)),
            key=lambda path: path.name,
        )
    except FileNotFoundError:
        return []


def list_cache_locks(cache_dir: Optional[PathLike] = None) -> List[Path]:
    """List TensorRT engine-build lock files without changing them."""
    directory = resolve_cache_dir(cache_dir)
    try:
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.name.endswith(".trt.lock")
            ),
            key=lambda path: path.name,
        )
    except FileNotFoundError:
        return []


def _read_lock_pid(lock_path: Path) -> Optional[int]:
    try:
        fields = lock_path.read_text(encoding="utf-8").split()
    except (OSError, UnicodeError):
        return None
    for field in fields:
        if not field.startswith("pid="):
            continue
        try:
            return int(field[4:])
        except ValueError:
            return None
    return None


def _is_process_alive(pid: int) -> bool:
    """Check a PID without terminating it or importing optional packages."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        wait_timeout = 0x00000102
        wait_failed = 0xFFFFFFFF
        access_denied = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        wait_for_single_object = kernel32.WaitForSingleObject
        wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        wait_for_single_object.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(synchronize, False, pid)
        if not handle:
            # Protected processes may deny access even though they are alive.
            return ctypes.get_last_error() == access_denied
        try:
            wait_result = wait_for_single_object(handle, 0)
            return wait_result in (wait_timeout, wait_failed)
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _remove_stale_lock(lock_path: Path, stale_after_seconds: float) -> bool:
    """Remove a dead-PID or old malformed lock, returning whether it vanished."""
    try:
        pid = _read_lock_pid(lock_path)
        if pid is not None:
            if _is_process_alive(pid):
                return False
            lock_path.unlink()
            return True

        lock_age = time.time() - lock_path.stat().st_mtime
        if lock_age > stale_after_seconds:
            lock_path.unlink()
            return True
    except FileNotFoundError:
        return True
    return False


def list_active_cache_locks(
    cache_dir: Optional[PathLike] = None,
    stale_after_seconds: float = 900.0,
) -> List[Path]:
    """Return live locks after safely pruning stale lock files."""
    return [
        lock_path
        for lock_path in list_cache_locks(cache_dir)
        if not _remove_stale_lock(lock_path, stale_after_seconds)
    ]


def _move_cache_file(source: Path, destination: Path) -> None:
    """Move one cache file atomically, with a cross-volume copy fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
        return
    except FileNotFoundError:
        raise
    except OSError:
        pass

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".migration.tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
        try:
            source.unlink()
        except OSError:
            # The persistent copy is already complete.  A locked legacy file
            # can safely remain for a later cleanup attempt.
            pass
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def migrate_legacy_cache(
    cache_dir: Optional[PathLike] = None,
    legacy_cache_dir: Optional[PathLike] = None,
) -> Tuple[int, int]:
    """Move valid legacy cache files into the persistent cache directory.

    Migration is incremental and restart-safe: every destination update is an
    atomic replace, incomplete ``.tmp`` files and unrelated files stay behind,
    and a legacy engine-build lock makes migration fail closed.
    """
    destination_dir = resolve_cache_dir(cache_dir)
    source_dir = (
        get_legacy_cache_dir()
        if legacy_cache_dir is None
        else Path(legacy_cache_dir)
    )
    if source_dir.resolve() == destination_dir.resolve():
        return 0, 0

    locks = list_active_cache_locks(source_dir)
    if locks:
        raise CacheInUseError(
            "Legacy TensorRT cache is in use by an engine build: "
            + ", ".join(path.name for path in locks)
        )

    candidates = [
        path
        for path in list_cache_files(source_dir)
        if not path.name.endswith(".tmp")
    ]
    if not candidates:
        return 0, 0

    destination_dir.mkdir(parents=True, exist_ok=True)
    migrated_count = 0
    migrated_bytes = 0
    for source in candidates:
        destination = destination_dir / source.name
        if destination.exists():
            continue
        try:
            size = source.stat().st_size
            _move_cache_file(source, destination)
        except FileNotFoundError:
            continue
        migrated_count += 1
        migrated_bytes += size

    try:
        source_dir.rmdir()
    except (FileNotFoundError, OSError):
        pass

    return migrated_count, migrated_bytes


def get_cache_usage(cache_dir: Optional[PathLike] = None) -> Tuple[int, int]:
    """Return the number and total bytes of managed persistent cache files."""
    count = 0
    total_bytes = 0
    for path in list_cache_files(cache_dir):
        try:
            total_bytes += path.stat().st_size
            count += 1
        except FileNotFoundError:
            pass
    return count, total_bytes


def clear_cache(cache_dir: Optional[PathLike] = None) -> Tuple[int, int]:
    """Delete only managed persistent cache files and return count/bytes.

    The directory is never recursively removed.  A live engine-build lock makes
    cleanup fail closed so the launcher cannot delete a cache being written.
    """
    directory = resolve_cache_dir(cache_dir)
    locks = list_active_cache_locks(directory)
    if locks:
        raise CacheInUseError(
            "TensorRT cache is in use by an engine build: "
            + ", ".join(path.name for path in locks)
        )

    deleted_count = 0
    deleted_bytes = 0
    for path in list_cache_files(directory):
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        deleted_count += 1
        deleted_bytes += size

    # Remove the dedicated directory only when it is genuinely empty.  Any
    # unrelated file (particularly with an overridden path) is preserved.
    try:
        directory.rmdir()
    except (FileNotFoundError, OSError):
        pass

    return deleted_count, deleted_bytes


def _cache_token(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def get_engine_cache_path(
    onnx_path: PathLike,
    trt_version: str,
    cache_dir: Optional[PathLike] = None,
) -> Path:
    """Return a collision-safe path for one ONNX model and builder setup."""
    source = Path(onnx_path)
    directory = get_cache_dir(cache_dir)
    token = _cache_token(
        ENGINE_CACHE_SCHEMA,
        trt_version,
        BUILDER_CONFIG_FINGERPRINT,
        file_sha256(source, directory),
    )
    return directory / f"{source.stem}-{token}.trt"


def get_runtime_cache_path(
    source_path: PathLike,
    trt_version: str,
    cache_dir: Optional[PathLike] = None,
) -> Path:
    """Return a GPU/runtime-validated JIT cache path for an engine source."""
    source = Path(source_path)
    directory = get_cache_dir(cache_dir)
    token = _cache_token(
        RUNTIME_CACHE_SCHEMA,
        trt_version,
        BUILDER_CONFIG_FINGERPRINT,
        file_sha256(source, directory),
    )
    return directory / f"{source.stem}-{token}.runtime.cache"


def atomic_write(path: PathLike, data) -> None:
    """Replace a cache file atomically so interrupted writes are never reused."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


@contextmanager
def engine_build_lock(
    engine_path: PathLike,
    timeout_seconds: float = 600.0,
    stale_after_seconds: float = 900.0,
) -> Iterator[None]:
    """Prevent two EasyVtuber processes from building the same engine at once."""
    lock_path = Path(f"{engine_path}.lock")
    started_at = time.monotonic()

    while True:
        try:
            descriptor = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if _remove_stale_lock(lock_path, stale_after_seconds):
                continue

            if time.monotonic() - started_at >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for engine cache lock: {lock_path}")
            time.sleep(0.2)
            continue

        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"pid={os.getpid()} created={time.time()}\n")
        break

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_runtime_cache(runtime_config, path: PathLike):
    """Create, optionally hydrate, and attach a TensorRT-RTX runtime cache."""
    runtime_cache = runtime_config.create_runtime_cache()
    cache_path = Path(path)
    loaded = False
    if cache_path.is_file():
        loaded = bool(runtime_cache.deserialize(cache_path.read_bytes()))
        if not loaded and hasattr(runtime_cache, "reset"):
            runtime_cache.reset()
    attached = runtime_config.set_runtime_cache(runtime_cache)
    if attached is False:
        raise RuntimeError("TensorRT-RTX rejected the runtime cache")
    return runtime_cache, loaded


def save_runtime_cache(runtime_cache, path: PathLike) -> bool:
    """Serialize a runtime cache, replacing the file only when it changed."""
    serialized = runtime_cache.serialize()
    if serialized is None:
        raise RuntimeError("TensorRT-RTX returned an empty runtime cache")
    if hasattr(serialized, "__enter__"):
        with serialized as buffer:
            data = bytes(buffer)
    else:
        data = bytes(serialized)

    cache_path = Path(path)
    try:
        if cache_path.stat().st_size == len(data) and cache_path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass

    atomic_write(cache_path, data)
    return True
