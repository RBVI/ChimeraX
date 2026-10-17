from chimerax.graphics.view import _draw_until_rendering_complete


class _Renderer:
    _pathtrace_accumulation_complete = False


def test_progressive_capture_draws_until_complete():
    renderer = _Renderer()
    draws = []

    def draw():
        draws.append(None)
        renderer._pathtrace_accumulation_complete = len(draws) >= 3

    _draw_until_rendering_complete(draw, renderer, wait_for_rendering=True)

    assert len(draws) == 3


def test_regular_capture_draws_only_once():
    renderer = _Renderer()
    draws = []

    _draw_until_rendering_complete(
        lambda: draws.append(None), renderer, wait_for_rendering=False
    )

    assert len(draws) == 1


def test_waiting_is_a_no_op_for_non_progressive_renderer():
    renderer = object()
    draws = []

    _draw_until_rendering_complete(
        lambda: draws.append(None), renderer, wait_for_rendering=True
    )

    assert len(draws) == 1
