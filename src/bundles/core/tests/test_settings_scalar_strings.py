"""Settings serialize NumPy scalars without importing NumPy for plain values."""
import sys
from types import ModuleType

import pytest

from chimerax.core.configfile import Value


@pytest.mark.parametrize('value', [3, 1.25, True, None, 'text', (1, 2)])
def test_plain_values_work_with_absent_or_partial_numpy(monkeypatch, value):
    for module in (None, ModuleType('numpy')):
        monkeypatch.setitem(sys.modules, 'numpy', module)
        assert Value(value).to_str(value) == repr(value)


def test_numpy_scalar_literals_match_previous_serializer():
    import numpy
    for value in (numpy.int64(7), numpy.float32(1.25), numpy.bool_(True)):
        assert Value(value).to_str(value) == str(value)
    value = numpy.arange(3)
    assert Value(value).to_str(value) == repr(value)


def test_explicit_serializer_is_unchanged():
    function = lambda value: 'custom'
    assert Value(None, to_str=function).to_str is function
