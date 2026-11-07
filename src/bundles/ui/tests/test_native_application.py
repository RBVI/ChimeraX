"""Native application construction is gated before any application mutation."""
import struct
import ctypes
from types import SimpleNamespace

import pytest

from chimerax.ui import native_application as native


def image_header(uuid, *, cpu=0x100000c, length=24, count=1):
    command = struct.pack('<2I', 0x1b, length) + bytes.fromhex(uuid)
    return struct.pack('<8I', 0xfeedfacf, cpu, 0, 6, count, len(command), 0, 0) + command


def test_exact_native_image_identity():
    uuid = '0123456789abcdef0123456789abcdef'
    assert native._image_uuid(image_header(uuid)) == uuid


def test_unregistered_application_uses_its_deleting_destructor():
    deleted = []
    destructor = ctypes.PYFUNCTYPE(None, ctypes.c_void_p)(deleted.append)
    table = (ctypes.c_void_p * 5)(None, None, None, None,
        ctypes.cast(destructor, ctypes.c_void_p).value)
    instance = ctypes.c_void_p(ctypes.addressof(table))
    address = ctypes.addressof(instance)
    native._destroy_unregistered_application(address)
    assert deleted == [address]


@pytest.mark.parametrize('change', ['truncated', 'architecture', 'zero_command', 'long_command', 'count'])
def test_unsupported_or_malformed_image_is_rejected(change):
    uuid = '0123456789abcdef0123456789abcdef'
    data = image_header(uuid)
    if change == 'truncated':
        data = data[:-1]
    elif change == 'architecture':
        data = image_header(uuid, cpu=0x1000007)
    elif change == 'zero_command':
        data = image_header(uuid, length=0)
    elif change == 'long_command':
        data = image_header(uuid, length=32)
    else:
        data = image_header(uuid, count=2)
    with pytest.raises(ValueError):
        native._image_uuid(data)


@pytest.fixture
def application(monkeypatch):
    monkeypatch.setattr(native, 'sys', SimpleNamespace(platform='darwin',
        implementation=SimpleNamespace(name='cpython')))
    class QApplication:
        __module__ = 'PySide6.QtWidgets'
        @staticmethod
        def instance():
            return None
    class UI(QApplication):
        pass
    return UI(), QApplication


@pytest.mark.parametrize('error', [OSError('missing image'), AttributeError('missing symbol'),
    ValueError('different image')])
def test_unrecognized_binding_preserves_ordinary_construction(application, monkeypatch, error):
    ui, app = application
    def unsupported(*args):
        raise error
    monkeypatch.setattr(native, '_prepare_constructor', unsupported)
    assert native.prepare_application_constructor(ui, app, ['ChimeraX']) is None


def test_allocation_failure_is_not_hidden_by_binding_fallback(application, monkeypatch):
    ui, app = application
    def unavailable(*args):
        raise MemoryError('allocation')
    monkeypatch.setattr(native, '_prepare_constructor', unavailable)
    with pytest.raises(MemoryError):
        native.prepare_application_constructor(ui, app, ['ChimeraX'])


@pytest.mark.parametrize('case', ['other_platform', 'other_interpreter', 'worker',
    'existing_application', 'mixin', 'embedded_nul', 'non_string'])
def test_unsupported_call_does_not_prepare_native_entry_points(application, monkeypatch, case):
    ui, app = application
    arguments = ['ChimeraX']
    def unexpected(*args):
        pytest.fail('Prepared native application entry points')
    monkeypatch.setattr(native, '_prepare_constructor', unexpected)
    if case == 'other_platform':
        native.sys.platform = 'linux'
    elif case == 'other_interpreter':
        native.sys.implementation.name = 'pypy'
    elif case == 'worker':
        monkeypatch.setattr(native, 'current_thread', lambda: object())
    elif case == 'existing_application':
        monkeypatch.setattr(app, 'instance', staticmethod(lambda: object()))
    elif case == 'mixin':
        class Mixin:
            pass
        class Mixed(app, Mixin):
            pass
        ui = Mixed()
    elif case == 'embedded_nul':
        arguments = ['ChimeraX\0other']
    else:
        arguments = [17]
    assert native.prepare_application_constructor(ui, app, arguments) is None
