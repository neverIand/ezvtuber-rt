import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "ezvtb_rt" / "trt_cache.py"
SPEC = importlib.util.spec_from_file_location("ezvtb_rt_trt_cache_test", MODULE_PATH)
trt_cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trt_cache)


class FakeRuntimeCache:
    def __init__(self, serialized=b"new-cache"):
        self.serialized = serialized
        self.deserialized = None

    def deserialize(self, data):
        self.deserialized = data
        return True

    def serialize(self):
        return self.serialized


class FakeRuntimeConfig:
    def __init__(self, runtime_cache):
        self.runtime_cache = runtime_cache
        self.attached = None

    def create_runtime_cache(self):
        return self.runtime_cache

    def set_runtime_cache(self, runtime_cache):
        self.attached = runtime_cache
        return True


class TensorRTCacheTests(unittest.TestCase):
    def test_engine_cache_key_distinguishes_content_and_trt_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "model.onnx"
            same_content = second_dir / "model.onnx"
            different_content = root / "model.onnx"
            first.write_bytes(b"model-a")
            same_content.write_bytes(b"model-a")
            different_content.write_bytes(b"model-b")

            first_path = trt_cache.get_engine_cache_path(first, "1.3", root / "cache")
            same_path = trt_cache.get_engine_cache_path(same_content, "1.3", root / "cache")
            different_path = trt_cache.get_engine_cache_path(
                different_content,
                "1.3",
                root / "cache",
            )
            new_version_path = trt_cache.get_engine_cache_path(
                first,
                "1.4",
                root / "cache",
            )

            self.assertEqual(first_path, same_path)
            self.assertNotEqual(first_path, different_path)
            self.assertNotEqual(first_path, new_version_path)

    def test_atomic_write_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "cache.bin"
            destination.write_bytes(b"old")
            trt_cache.atomic_write(destination, b"replacement")

            self.assertEqual(destination.read_bytes(), b"replacement")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_runtime_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "runtime.cache"
            cache_path.write_bytes(b"existing-cache")
            runtime_cache = FakeRuntimeCache()
            runtime_config = FakeRuntimeConfig(runtime_cache)

            loaded_cache, loaded = trt_cache.load_runtime_cache(
                runtime_config,
                cache_path,
            )
            self.assertTrue(loaded)
            self.assertIs(loaded_cache, runtime_cache)
            self.assertIs(runtime_config.attached, runtime_cache)
            self.assertEqual(runtime_cache.deserialized, b"existing-cache")

            self.assertTrue(trt_cache.save_runtime_cache(runtime_cache, cache_path))
            self.assertEqual(cache_path.read_bytes(), b"new-cache")

            written_at = cache_path.stat().st_mtime_ns
            self.assertFalse(trt_cache.save_runtime_cache(runtime_cache, cache_path))
            self.assertEqual(cache_path.stat().st_mtime_ns, written_at)

    def test_engine_build_lock_removes_lock_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_path = Path(temporary_directory) / "model.trt"
            lock_path = Path(f"{engine_path}.lock")
            with trt_cache.engine_build_lock(engine_path):
                self.assertTrue(lock_path.is_file())
            self.assertFalse(lock_path.exists())

    def test_cache_cleanup_deletes_only_managed_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "cache"
            cache_dir.mkdir()
            engine = cache_dir / "model-abc.trt"
            runtime = cache_dir / "model-def.runtime.cache"
            temporary = cache_dir / ".model-abc.trt.random.tmp"
            unrelated = cache_dir / "keep-me.txt"
            nested = cache_dir / "nested"
            nested.mkdir()
            nested_engine = nested / "keep-me.trt"

            engine.write_bytes(b"engine")
            runtime.write_bytes(b"runtime")
            temporary.write_bytes(b"temporary")
            unrelated.write_bytes(b"unrelated")
            nested_engine.write_bytes(b"nested")

            self.assertEqual(trt_cache.get_cache_usage(cache_dir), (3, 22))
            deleted_count, deleted_bytes = trt_cache.clear_cache(cache_dir)

            self.assertEqual((deleted_count, deleted_bytes), (3, 22))
            self.assertFalse(engine.exists())
            self.assertFalse(runtime.exists())
            self.assertFalse(temporary.exists())
            self.assertEqual(unrelated.read_bytes(), b"unrelated")
            self.assertEqual(nested_engine.read_bytes(), b"nested")
            self.assertTrue(cache_dir.is_dir())

    def test_cache_cleanup_refuses_engine_build_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            engine = cache_dir / "model.trt"
            lock = cache_dir / "model.trt.lock"
            engine.write_bytes(b"engine")
            lock.write_text(f"pid={os.getpid()}", encoding="utf-8")

            with self.assertRaises(trt_cache.CacheInUseError):
                trt_cache.clear_cache(cache_dir)

            self.assertTrue(engine.exists())
            self.assertTrue(lock.exists())

    def test_empty_cache_inspection_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory) / "missing"

            self.assertEqual(trt_cache.get_cache_usage(cache_dir), (0, 0))
            self.assertEqual(trt_cache.list_cache_locks(cache_dir), [])
            self.assertFalse(cache_dir.exists())

    def test_default_cache_dir_is_persistent_and_override_still_wins(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            local_app_data = root / "local"
            override = root / "override"
            with mock.patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": str(local_app_data),
                    trt_cache.ENGINE_CACHE_ENV: "",
                },
            ):
                self.assertEqual(
                    trt_cache.resolve_cache_dir(),
                    local_app_data / "EasyVtuber" / "trt-cache",
                )
                os.environ[trt_cache.ENGINE_CACHE_ENV] = str(override)
                self.assertEqual(trt_cache.resolve_cache_dir(), override)

    def test_default_cache_migrates_valid_legacy_files_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "legacy"
            local_app_data = root / "local"
            legacy.mkdir()
            engine = legacy / "model-abc.trt"
            runtime = legacy / "model-def.runtime.cache"
            incomplete = legacy / ".model-abc.trt.random.tmp"
            unrelated = legacy / "keep.txt"
            engine.write_bytes(b"engine")
            runtime.write_bytes(b"runtime")
            incomplete.write_bytes(b"partial")
            unrelated.write_bytes(b"keep")

            with mock.patch.dict(
                os.environ,
                {
                    "LOCALAPPDATA": str(local_app_data),
                    trt_cache.ENGINE_CACHE_ENV: "",
                },
            ), mock.patch.object(
                trt_cache,
                "get_legacy_cache_dir",
                return_value=legacy,
            ):
                destination = trt_cache.get_cache_dir()
                second_destination = trt_cache.get_cache_dir()

            self.assertEqual(destination, local_app_data / "EasyVtuber" / "trt-cache")
            self.assertEqual(second_destination, destination)
            self.assertEqual((destination / engine.name).read_bytes(), b"engine")
            self.assertEqual((destination / runtime.name).read_bytes(), b"runtime")
            self.assertFalse(engine.exists())
            self.assertFalse(runtime.exists())
            self.assertEqual(incomplete.read_bytes(), b"partial")
            self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_environment_override_does_not_migrate_legacy_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "legacy"
            override = root / "override"
            legacy.mkdir()
            engine = legacy / "model.trt"
            engine.write_bytes(b"engine")

            with mock.patch.dict(
                os.environ,
                {trt_cache.ENGINE_CACHE_ENV: str(override)},
            ), mock.patch.object(
                trt_cache,
                "get_legacy_cache_dir",
                return_value=legacy,
            ):
                self.assertEqual(trt_cache.get_cache_dir(), override)

            self.assertTrue(engine.exists())
            self.assertFalse((override / engine.name).exists())

    def test_legacy_migration_refuses_engine_build_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "legacy"
            destination = root / "persistent"
            legacy.mkdir()
            engine = legacy / "model.trt"
            lock = legacy / "model.trt.lock"
            engine.write_bytes(b"engine")
            lock.write_text(f"pid={os.getpid()}", encoding="utf-8")

            with self.assertRaises(trt_cache.CacheInUseError):
                trt_cache.migrate_legacy_cache(destination, legacy)

            self.assertTrue(engine.exists())
            self.assertTrue(lock.exists())
            self.assertFalse(destination.exists())

    def test_current_process_is_reported_alive(self):
        self.assertTrue(trt_cache._is_process_alive(os.getpid()))

    def test_dead_process_lock_is_removed_immediately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = Path(temporary_directory) / "model.trt.lock"
            lock.write_text("pid=123", encoding="utf-8")

            with mock.patch.object(
                trt_cache,
                "_is_process_alive",
                return_value=False,
            ):
                self.assertTrue(trt_cache._remove_stale_lock(lock, 900.0))

            self.assertFalse(lock.exists())

    def test_engine_build_lock_reclaims_dead_process_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = Path(temporary_directory) / "model.trt"
            lock = Path(f"{engine}.lock")
            lock.write_text("pid=123", encoding="utf-8")

            with mock.patch.object(
                trt_cache,
                "_is_process_alive",
                return_value=False,
            ):
                with trt_cache.engine_build_lock(engine, timeout_seconds=0.1):
                    self.assertEqual(
                        trt_cache._read_lock_pid(lock),
                        os.getpid(),
                    )

            self.assertFalse(lock.exists())

    def test_live_process_lock_is_not_removed_even_when_old(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock = Path(temporary_directory) / "model.trt.lock"
            lock.write_text(f"pid={os.getpid()}", encoding="utf-8")
            os.utime(lock, (1, 1))

            self.assertFalse(trt_cache._remove_stale_lock(lock, 0.0))
            self.assertTrue(lock.exists())

    def test_old_malformed_lock_uses_age_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            lock = cache_dir / "model.trt.lock"
            lock.write_text("incomplete", encoding="utf-8")
            os.utime(lock, (1, 1))

            self.assertEqual(
                trt_cache.list_active_cache_locks(
                    cache_dir,
                    stale_after_seconds=0.0,
                ),
                [],
            )
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
