from chimerax.graphics.view import _draw_until_rendering_complete, image_from_rgba


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


def test_image_from_rgba_handles_reversed_vulkan_screen_readback():
    from numpy import array, uint8

    screen = array([[[100, 50, 25, 255]], [[25, 50, 100, 128]]], dtype=uint8)
    rgba = screen[::-1]

    image = image_from_rgba(rgba)

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (100, 50, 25)
    assert image.getpixel((0, 1)) == (25, 50, 100)
