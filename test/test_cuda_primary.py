import importlib.util
from pathlib import Path
import types
import unittest
from unittest import mock


SOURCE = Path(__file__).parents[1] / 'ezvtb_rt/cuda_primary.py'


class PrimaryContextTests(unittest.TestCase):
    def load_module(self, *, push_error=None):
        context = mock.Mock()
        context.push.side_effect = push_error
        device = mock.Mock()
        device.retain_primary_context.return_value = context
        driver = types.ModuleType('pycuda.driver')
        driver.init = mock.Mock()
        driver.Device = mock.Mock(return_value=device)
        package = types.ModuleType('pycuda')
        package.driver = driver
        package.__path__ = []
        utilities = types.ModuleType('pycuda.tools')
        utilities.clear_context_caches = mock.Mock()
        replacements = {'pycuda': package, 'pycuda.driver': driver, 'pycuda.tools': utilities}
        spec = importlib.util.spec_from_file_location('test_primary_context_module', SOURCE)
        module = importlib.util.module_from_spec(spec)
        return module, spec, replacements, context, device, driver, utilities

    def test_selects_ezvtb_device_and_balances_context_lifetime(self):
        module, spec, modules, context, device, driver, utilities = self.load_module()
        with mock.patch.dict('sys.modules', modules), mock.patch.dict(
                'os.environ', {'EZVTB_DEVICE_ID': '2', 'CUDA_DEVICE': '1'}), mock.patch('atexit.register') as register:
            spec.loader.exec_module(module)
            driver.Device.assert_called_once_with(2)
            device.retain_primary_context.assert_called_once_with()
            device.make_context.assert_not_called()
            context.push.assert_called_once_with()
            register.assert_called_once_with(module._finish_up)
            module._finish_up()
            module._finish_up()
        context.pop.assert_called_once_with()
        context.detach.assert_called_once_with()
        utilities.clear_context_caches.assert_called_once_with()

    def test_failed_push_releases_retained_context_without_registering_cleanup(self):
        module, spec, modules, context, *_ = self.load_module(push_error=RuntimeError('push failed'))
        with mock.patch.dict('sys.modules', modules), mock.patch('atexit.register') as register:
            with self.assertRaisesRegex(RuntimeError, 'push failed'):
                spec.loader.exec_module(module)
        context.detach.assert_called_once_with()
        context.pop.assert_not_called()
        register.assert_not_called()
