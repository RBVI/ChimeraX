import json

import pytest


def test_encode_numpy_int_scalar():
    # Regression for sbagent#40: arr.sum() yields a numpy.int64 scalar, which
    # has 'shape' but no '__len__'. Previously this raised
    # "TypeError: Can't JSON-encode '1'".
    import numpy
    from chimerax.core.commands.run import ArrayJSONEncoder

    val = numpy.array([True]).sum()  # numpy.int64(1)
    assert json.loads(ArrayJSONEncoder().encode({"n": val})) == {"n": 1}


def test_encode_numpy_float_scalar():
    # Regression for sbagent#33: numpy.float32 from an info attribute path.
    import numpy
    from chimerax.core.commands.run import ArrayJSONEncoder

    val = numpy.float32(0.24107859)
    decoded = json.loads(ArrayJSONEncoder().encode({"x": val}))["x"]
    assert isinstance(decoded, float)
    assert decoded == pytest.approx(float(val))


def test_encode_numpy_bool_scalar():
    import numpy
    from chimerax.core.commands.run import ArrayJSONEncoder

    assert json.loads(ArrayJSONEncoder().encode({"b": numpy.bool_(True)})) == {"b": True}


def test_encode_nested_numpy_arrays():
    # The existing array path must keep working (no regression).
    import numpy
    from chimerax.core.commands.run import ArrayJSONEncoder

    payload = {"a": numpy.array([1, 2, 3]), "b": [numpy.array([4.0, 5.0])]}
    assert json.loads(ArrayJSONEncoder().encode(payload)) == {
        "a": [1, 2, 3],
        "b": [[4.0, 5.0]],
    }


def test_unencodable_object_still_raises():
    from chimerax.core.commands.run import ArrayJSONEncoder

    class Opaque:
        pass

    with pytest.raises(TypeError):
        ArrayJSONEncoder().encode({"o": Opaque()})
