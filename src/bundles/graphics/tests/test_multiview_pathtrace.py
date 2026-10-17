from types import SimpleNamespace

from chimerax.graphics.camera360 import CubeMapCamera
from chimerax.graphics.render_graph import PassContext, RenderGraph, RenderPass
from chimerax.graphics.render_passes import PathTracePass, RTAOPass


def _context(view_count):
    renderer = SimpleNamespace(
        supports_ray_tracing=lambda: True,
        lighting=SimpleNamespace(pathtrace=True),
    )
    camera = SimpleNamespace(number_of_views=lambda: view_count)
    return SimpleNamespace(renderer=renderer, camera=camera)


def test_pathtrace_runs_for_single_view_camera():
    assert PathTracePass().enabled(_context(1))


def test_pathtrace_falls_back_to_raster_for_multiview_camera():
    assert not PathTracePass().enabled(_context(6))


def test_rtao_falls_back_to_raster_for_multiview_camera():
    context = _context(6)
    context.renderer.lighting.rtao = 1
    assert not RTAOPass().enabled(context)


def test_vulkan_cube_faces_are_separate_submissions():
    events = []
    render = SimpleNamespace(
        end_frame=lambda: events.append("end"),
        pop_framebuffer=lambda: events.append("pop"),
        push_framebuffer=lambda framebuffer: events.append(("push", framebuffer)),
    )
    camera = CubeMapCamera.__new__(CubeMapCamera)
    camera._vulkan_cube_face_framebuffers = lambda renderer, count: range(count)

    camera._set_vulkan_face_target(0, render, 6)
    camera._set_vulkan_face_target(1, render, 6)

    assert events == [("push", 0), "end", "pop", ("push", 1)]


def test_vulkan_multiview_target_is_selected_before_pre_view_passes():
    events = []

    class PreViewPass(RenderPass):
        def execute(self, ctx):
            events.append("pre-view")

    camera = SimpleNamespace(
        number_of_views=lambda: 2,
        _set_vulkan_face_target=lambda *args: None,
        set_render_target=lambda view_num, render: events.append(
            ("target", view_num)),
        draw_background=lambda view_num, render: events.append(
            ("background", view_num)),
    )
    renderer = SimpleNamespace(
        _vulkan=object(),
        backend_name='vulkan',
        offscreen=SimpleNamespace(enabled=False),
        outline=SimpleNamespace(offscreen_outline_needed=False),
    )
    context = PassContext(
        renderer=renderer,
        camera=camera,
        view=SimpleNamespace(silhouette=None),
        opaque_drawings=[],
        transparent_drawings=[],
        highlight_drawings=[],
        on_top_drawings=[],
    )
    graph = RenderGraph()
    graph.add_pre_view(PreViewPass())

    graph.execute(context)

    assert events == [
        ("target", 0),
        "pre-view",
        ("background", 0),
        ("target", 1),
        ("background", 1),
    ]
