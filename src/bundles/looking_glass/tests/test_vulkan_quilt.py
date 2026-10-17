from types import SimpleNamespace

from chimerax.looking_glass.lookingglass import LookingGlassCamera


def test_quilt_tile_preserves_previous_views_and_flips_row_origin():
    events = []
    camera = LookingGlassCamera.__new__(LookingGlassCamera)
    camera._quilt_size = (4096, 4096)
    camera._quilt_tile_size = (819, 455)
    camera._quilt_columns = 5
    camera._tile_framebuffer = SimpleNamespace(color_texture=SimpleNamespace(
        _handle=17, prepare_for_sampling=lambda: events.append('sample')))
    quilt = object()
    camera._quilt_framebuffer = lambda r: quilt
    renderer = SimpleNamespace(
        _ctx=1, end_frame=lambda: events.append('end'),
        pop_framebuffer=lambda: events.append('pop'),
        push_framebuffer=lambda fb: events.append(('push', fb)),
        begin_frame=lambda: events.append('begin'),
        draw_background=lambda: events.append('clear'),
        set_viewport=lambda *v: events.append(v),
        draw_fullscreen_quad=lambda *a, **kw: events.append('draw'),
        _vulkan=SimpleNamespace(begin_render_pass=lambda **kw: events.append(
            ('load', kw['load_color'], kw['load_depth']))))
    camera._finish_vulkan_tile(6, renderer)
    assert events == ['end', 'pop', 'sample', ('push', quilt), 'begin',
                      ('load', 1, 1), (819, 3186, 819, 455), 'draw', 'end', 'pop']
