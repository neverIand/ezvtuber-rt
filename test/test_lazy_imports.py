from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).parents[1]


class LazyImportTests(unittest.TestCase):
    def test_package_import_does_not_load_any_inference_backend(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r}); "
            "import ezvtb_rt; "
            "forbidden = ('tensorrt_rtx', 'pycuda.autoinit', 'pycuda.driver', 'ezvtb_rt.cuda_primary', 'onnxruntime'); "
            "assert not any(name in sys.modules for name in forbidden)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
