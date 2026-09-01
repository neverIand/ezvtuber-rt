"""Disk-cache helpers for TensorRT-RTX engines and runtime kernels.

This module intentionally has no CUDA or TensorRT imports so cache naming and
file handling can be tested without initializing a GPU runtime.
"""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator, List, Optional, Tuple, Union


PathLike = Union[str, os.PathLike]
ENGINE_CACHE_SCHEMA = "2"
RUNTIME_CACHE_SCHEMA = "1"
ENGINE_CACHE_ENV = "EZVTB_TRT_CACHE_DIR"

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


def file_sha256(path: PathLike) -> str:
    """Return a content digest, avoiding duplicate reads within one process."""
    resolved = str(Path(path).resolve())
    stat = os.stat(resolved)
    return _sha256_for_file_state(
        resolved,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


class CacheInUseError(RuntimeError):
    """Raised when a TensorRT engine build lock makes cleanup unsafe."""


def resolve_cache_dir(cache_dir: Optional[PathLike] = None) -> Path:
    """Return the configured cache directory without creating it."""
    if cache_dir is None:
        cache_dir = os.environ.get(ENGINE_CACHE_ENV)
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "ezvtuber_rt_engines"
    return Path(cache_dir)


def get_cache_dir(cache_dir: Optional[PathLike] = None) -> Path:
    result = resolve_cache_dir(cache_dir)
    result.mkdir(parents=True, exist_ok=True)
    return result


def _is_managed_cache_file(path: Path) -> bool:
    """Return whether ``path`` is a file created by this cache module."""
    if not path.is_file():
        return False
    name = path.name
    if name.endswith(".trt") or name.endswith(".runtime.cache"):
        return True
    return (
        name.startswith(".")
        and name.endswith(".tmp")
        and (".trt." in name or ".runtime.cache." in name)
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
    """List active TensorRT engine-build lock files without deleting them."""
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
    locks = list_cache_locks(directory)
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
    token = _cache_token(
        ENGINE_CACHE_SCHEMA,
        trt_version,
        BUILDER_CONFIG_FINGERPRINT,
        file_sha256(source),
    )
    return get_cache_dir(cache_dir) / f"{source.stem}-{token}.trt"


def get_runtime_cache_path(
    source_path: PathLike,
    trt_version: str,
    cache_dir: Optional[PathLike] = None,
) -> Path:
    """Return a GPU/runtime-validated JIT cache path for an engine source."""
    source = Path(source_path)
    token = _cache_token(
        RUNTIME_CACHE_SCHEMA,
        trt_version,
        BUILDER_CONFIG_FINGERPRINT,
        file_sha256(source),
    )
    return get_cache_dir(cache_dir) / f"{source.stem}-{token}.runtime.cache"


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
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
                if lock_age > stale_after_seconds:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
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
