import numpy

from chimerax.atomic.pbgroup import _pseudobond_primitive_batch
from chimerax.atomic.structure import _halfbond_primitive_batch


def test_bonds_are_two_uniform_color_half_cylinders():
    starts = numpy.array(((0, 0, 0),), numpy.float32)
    ends = numpy.array(((0, 0, 4),), numpy.float32)
    radii = numpy.array((0.25,), numpy.float32)
    colors = numpy.array(((255, 0, 0, 255), (0, 0, 255, 255)), numpy.uint8)

    geometry = _halfbond_primitive_batch(starts, ends, radii, colors)

    assert geometry.primitive_count == 2
    assert numpy.allclose(geometry.ends[0], (0, 0, 2))
    assert numpy.allclose(geometry.starts[1], (0, 0, 2))
    assert geometry.picking_ids.tolist() == [0, 0]
    assert numpy.all(geometry.colors == colors)
    assert not geometry.caps.any()


def test_pseudobond_dashes_are_explicit_capped_primitives():
    starts = numpy.array(((0, 0, 0),), numpy.float32)
    ends = numpy.array(((0, 0, 6),), numpy.float32)
    radii = numpy.array((0.1,), numpy.float32)
    colors = numpy.array(((255, 0, 0, 255), (0, 0, 255, 255)), numpy.uint8)

    geometry = _pseudobond_primitive_batch(
        starts, ends, radii, colors, dashes=3, sides=10)

    # Three visible dash intervals; the center interval is split only to
    # preserve the two endpoint colors.
    assert geometry.primitive_count == 4
    assert geometry.picking_ids.tolist() == [0, 0, 0, 0]
    assert geometry.sides == 10
    assert geometry.caps.tolist() == [
        [True, True], [True, False], [False, True], [True, True]]
    assert numpy.all(geometry.colors[:2] == colors[0])
    assert numpy.all(geometry.colors[2:] == colors[1])


def test_even_pseudobond_dash_count_preserves_shader_phase_and_spacing():
    starts = numpy.array(((0, 0, 0),), numpy.float32)
    ends = numpy.array(((0, 0, 6),), numpy.float32)
    radii = numpy.array((0.1,), numpy.float32)
    colors = numpy.array(((255, 0, 0, 255), (0, 0, 255, 255)), numpy.uint8)

    geometry = _pseudobond_primitive_batch(
        starts, ends, radii, colors, dashes=6, sides=10)

    assert geometry.primitive_count == 6
    assert numpy.allclose(geometry.starts[:, 2], (.25, 1.25, 2.25, 3.25, 4.25, 5.25))
    assert numpy.allclose(geometry.ends[:, 2], (.75, 1.75, 2.75, 3.75, 4.75, 5.75))
    assert geometry.caps.all()
