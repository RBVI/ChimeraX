"""Writing Tools interception is all-or-nothing and limited to the inspected app."""
from types import SimpleNamespace

import pytest


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


@pytest.fixture
def policy(monkeypatch):
    from chimerax.ui import writing_tools as policy

    base, native = 0x10000000, 0x20000000
    state = SimpleNamespace(existing=False, uuid=policy._APPKIT_UUID,
        missing=None, changed_encoding=None, changed_implementation=None,
        source_encoding=b'B16@0:8', source_body=policy._ZERO_RETURN,
        mutations=[], reads=[])
    methods = {(cls, selector): (encoding, base + offset)
        for cls, selector, encoding, offset in policy._METHODS}

    class Pointer:
        @staticmethod
        def in_dll(framework, symbol):
            assert symbol == 'NSApp'
            return SimpleNamespace(value=state.existing)

    def method(cls, selector):
        key = (cls, selector)
        return None if key == state.missing else key

    def encoding(key):
        if key == 'source':
            return state.source_encoding
        return b'unknown' if key == state.changed_encoding else methods[key][0]

    def implementation(key):
        if key == 'source':
            return native
        return 0xdeadbeef if key == state.changed_implementation else methods[key][1]

    def read(address, size):
        assert (address, size) == (native, 8)
        state.reads.append(address)
        return state.source_body

    objc = SimpleNamespace(**{name: Function(callback) for name, callback in {
        'objc_getClass': lambda name: name,
        'sel_registerName': lambda name: name,
        'class_getClassMethod': method,
        'class_getInstanceMethod': lambda cls, selector: 'source',
        'method_getTypeEncoding': encoding,
        'method_getImplementation': implementation,
        'method_setImplementation': lambda key, value: state.mutations.append((key, value)),
    }.items()})
    monkeypatch.setattr(policy, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(policy, 'ctypes', SimpleNamespace(c_void_p=Pointer,
        c_char_p=object(), string_at=read))
    monkeypatch.setattr(policy, '_image', lambda *args: (base, state.uuid))
    return SimpleNamespace(module=policy, state=state, objc=objc, framework=object(), native=native)


def apply(policy):
    return policy.module.disable_writing_tools(policy.framework, policy.objc)


def test_known_standalone_image_installs_only_after_validation(policy):
    assert apply(policy) is True
    expected = [((cls, selector), policy.native)
        for cls, selector, encoding, offset in policy.module._METHODS]
    assert policy.state.mutations == expected


@pytest.mark.parametrize('field', ['missing', 'changed_encoding', 'changed_implementation'])
@pytest.mark.parametrize('index', range(4))
def test_any_changed_method_keeps_every_original_implementation(policy, field, index):
    cls, selector, _, _ = policy.module._METHODS[index]
    setattr(policy.state, field, (cls, selector))
    assert apply(policy) is False
    assert policy.state.mutations == []


@pytest.mark.parametrize('field,value', [('source_encoding', b'@16@0:8'),
    ('source_body', b'\0' * 8), ('uuid', 'uninspected'), ('existing', True)])
def test_unsupported_native_state_has_no_mutation(policy, field, value):
    setattr(policy.state, field, value)
    assert apply(policy) is False
    assert policy.state.mutations == []


def test_unknown_image_does_not_read_native_instructions(policy):
    policy.state.uuid = 'uninspected'
    assert apply(policy) is False
    assert policy.state.reads == []


@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_other_platforms_do_not_inspect_appkit(policy, monkeypatch, platform):
    monkeypatch.setattr(policy.module, 'sys', SimpleNamespace(platform=platform))
    monkeypatch.setattr(policy.module, '_image', lambda *args: pytest.fail('Loaded AppKit'))
    assert apply(policy) is False
    assert policy.state.mutations == []


def test_worker_thread_does_not_intercept_appkit(policy, monkeypatch):
    monkeypatch.setattr(policy.module, 'current_thread', lambda: object())
    assert apply(policy) is False
    assert policy.state.mutations == []


def test_unavailable_image_uses_normal_startup(policy, monkeypatch):
    def unavailable(*args):
        raise OSError('native image unavailable')
    monkeypatch.setattr(policy.module, '_image', unavailable)
    assert apply(policy) is False
    assert policy.state.mutations == []
