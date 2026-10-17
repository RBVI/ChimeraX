"""Camera target lifetime, precision, and submission-order regressions."""
from types import SimpleNamespace

import numpy as np

from chimerax.graphics.camera360 import CubeMapCamera
from chimerax.graphics.render_graph import PassContext, RenderGraph, RenderPass
from chimerax.graphics.vulkan import VulkanFramebuffer, VulkanRender


def test_precision_target_is_started_before_shadow_submission():
    events = []
    screen = object()
    offscreen = SimpleNamespace(enabled=True,
                                start=lambda r: events.append('offscreen'))
    renderer = SimpleNamespace(
        backend_name='vulkan', offscreen=offscreen,
        current_framebuffer=lambda: screen, default_framebuffer=lambda: screen)
    camera = SimpleNamespace(number_of_views=lambda: 1,
                             set_render_target=lambda *a: events.append('camera'))
    ctx = PassContext(renderer=renderer, camera=camera,
                      view=SimpleNamespace(silhouette=None),
                      opaque_drawings=[object()], transparent_drawings=[],
                      highlight_drawings=[], on_top_drawings=[])

    class ShadowPass(RenderPass):
        def execute(self, ctx):
            events.append('shadow')

    graph = RenderGraph()
    graph.add_pre_view(ShadowPass())
    graph.execute(ctx)
    assert events == ['camera', 'offscreen', 'shadow']
    assert ctx.offscreen_started


def test_cube_cache_recreated_after_backend_replacement():
    retired = []
    old_render = object()
    old = [SimpleNamespace(width=16, height=16, _render=old_render,
                           delete=lambda: retired.append(True)) for _ in range(6)]
    renderer = SimpleNamespace()

    def create(name, width, height, alpha):
        return SimpleNamespace(width=width, height=height, _render=renderer,
                               activate=lambda: True)

    renderer.create_framebuffer = create
    camera = CubeMapCamera.__new__(CubeMapCamera)
    camera._vulkan_framebuffers = old
    camera._cube_face_size = 16
    new = camera._vulkan_cube_face_framebuffers(renderer, 6)
    assert len(retired) == 6
    assert all(fb._render is renderer for fb in new)
    assert camera._vulkan_cube_face_framebuffers(renderer, 6) is new


def test_half_float_readback_preserves_raw_values_and_image_orientation():
    source = np.array([[[0, .5, 1, 1]], [[2, -.5, .25, 1]]], dtype=np.float16)
    events = []

    def read(**kw):
        assert kw['data'].dtype == np.float16
        kw['data'][:] = source
        events.append('read')

    fb = VulkanFramebuffer.__new__(VulkanFramebuffer)
    fb._handle, fb.width, fb.height, fb.color_bits = 17, 1, 2, 16
    fb._render = SimpleNamespace(
        end_frame=lambda: events.append('end'), _ctx=1,
        _vulkan=SimpleNamespace(read_framebuffer=read))
    np.testing.assert_array_equal(fb.read_image(raw=True), source[::-1])
    expected = np.array([[[255, 0, 64, 255]], [[0, 128, 255, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(fb.read_image(), expected)
    assert events == ['end', 'read', 'end', 'read']


def test_metal_export_invalidates_when_wrapper_changes_image():
    render = VulkanRender.__new__(VulkanRender)
    identity = [100, 200]
    exports, releases = [], []

    def export(**kw):
        exports.append(tuple(identity))
        return len(exports)

    render._vulkan = SimpleNamespace(texture_identity=lambda **kw: tuple(identity))
    render._metal_rt_module = SimpleNamespace(
        export_texture=export, release_texture=lambda **kw: releases.append(kw['texture']))
    render._metal_rt_ctx = 1
    render._exported_textures, render._exported_texture_images = {}, {}
    assert render._export_vk_texture(55) == 1
    assert render._export_vk_texture(55) == 1
    identity[:] = [300, 400]
    assert render._export_vk_texture(55) == 2
    assert exports == [(100, 200), (300, 400)]
    assert releases == [1]


def test_stereo_depth_prepass_preserves_eye_shift():
    from chimerax.graphics.renderer import SHADER_STEREO_360
    renderer = VulkanRender.__new__(VulkanRender)
    renderer._depth_only_rendering = False
    renderer.draw_depth_only(True)
    assert not renderer.disable_capabilities & SHADER_STEREO_360
    renderer._depth_only_rendering = True  # Shadow-map projection has no eye shift.
    renderer.draw_depth_only(True)
    assert renderer.disable_capabilities & SHADER_STEREO_360


def test_stereo_projection_samples_gpu_faces_and_covers_odd_size():
    events = []
    camera = CubeMapCamera.__new__(CubeMapCamera)
    camera._vulkan_framebuffers = [SimpleNamespace(color_texture=SimpleNamespace(
        _handle=i, prepare_for_sampling=lambda: events.append('sample'))) for i in range(12)]
    renderer = SimpleNamespace(
        draw_background=lambda: events.append('background'),
        render_size=lambda: (7, 5), _background_color=(0, 0, 0, 1),
        set_viewport=lambda *v: events.append(v),
        full_viewport=lambda: events.append('restore'),
        draw_fullscreen_quad_textures=lambda textures, *a, **kw: events.append(textures))
    camera._project_vulkan_targets(renderer, 'top-bottom')
    assert events[:13] == ['sample'] * 12 + ['background']
    assert events[13:] == [(0, 0, 7, 2), list(range(6)),
                           (0, 2, 7, 3), list(range(6, 12)), 'restore']


def test_scene_depth_copy_uses_native_depth_size_not_window_size():
    # MoltenVK picks the swapchain extent from the CAMetalLayer drawableSize,
    # which can differ from the Qt window size * devicePixelRatio.
    created, copied, destroyed = [], [], []
    native_size = [(801, 601)]

    def create_texture(**kw):
        created.append((kw['width'], kw['height']))
        return len(created)

    render = VulkanRender.__new__(VulkanRender)
    render._ctx = 1
    render._width, render._height = 800, 600
    render._framebuffer_stack = []
    render._scene_depth_copy = None
    render._scene_depth_copy_size = None
    render.destroy_texture = destroyed.append
    render._vulkan = SimpleNamespace(
        depth_texture_size=lambda **kw: native_size[0],
        create_texture=create_texture,
        copy_depth_texture=lambda **kw: copied.append(kw['texture']))

    assert render.get_scene_depth_texture() == 1
    assert render.get_scene_depth_texture() == 1
    assert created == [(801, 601)]
    native_size[0] = (640, 480)
    assert render.get_scene_depth_texture() == 2
    assert created == [(801, 601), (640, 480)]
    assert destroyed == [None, 1]
    assert copied == [1, 1, 2]

    native_size[0] = None
    assert render.get_scene_depth_texture() is None
