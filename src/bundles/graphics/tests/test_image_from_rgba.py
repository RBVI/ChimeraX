def test_image_from_rgba():
    # An odd size, so rows are not a multiple of 4 or 8 bytes.
    import numpy
    from chimerax.graphics import image_from_rgba

    rgba = numpy.random.default_rng(0).integers(0, 256, (7, 5, 4), dtype=numpy.uint8)
    rgb = image_from_rgba(rgba)
    assert rgb.mode == "RGB" and rgb.size == (5, 7)
    assert (numpy.asarray(rgb) == rgba[::-1, :, :3]).all()


def test_image_from_rgba_transparent_background():
    import numpy
    from chimerax.graphics import image_from_rgba

    rgba = numpy.random.default_rng(0).integers(0, 256, (7, 5, 4), dtype=numpy.uint8)
    image = image_from_rgba(rgba, transparent_background=True)
    assert image.mode == "RGBA" and image.size == (5, 7)
    assert (numpy.asarray(image) == rgba[::-1]).all()
