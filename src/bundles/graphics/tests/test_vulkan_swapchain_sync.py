"""Python renderer state follows swapchains the native side recreates."""
from types import SimpleNamespace

from chimerax.graphics.vulkan import VulkanRender


def _render(states):
    calls = []
    native = SimpleNamespace(
        swapchain_state=lambda context: states.pop(0),
        begin_frame=lambda context: True,
        end_frame=lambda context: calls.append("end"),
    )
    r = object.__new__(VulkanRender)
    r._vulkan = native
    r._ctx = 1
    r._width, r._height = 1000, 600
    r._swapchain_generation = None
    r._in_frame = False
    r._framebuffer_stack = []
    r._context = SimpleNamespace(window=None)
    r._release_exported_textures = lambda: calls.append("release")
    r.set_viewport = lambda x, y, w, h: calls.append(("viewport", w, h))
    return r, calls


def test_end_frame_adopts_natively_recreated_swapchain_size():
    # end_frame recreates the swapchain when MoltenVK reports the layer
    # size changed; Python's size and viewport must follow.
    r, calls = _render([(1, 1000, 600), (2, 1200, 700)])
    r._sync_swapchain_state()
    calls.clear()

    r._in_frame = True
    r.end_frame()

    assert (r._width, r._height) == (1200, 700)
    assert calls == ["end", "release", ("viewport", 1200, 700)]


def test_unchanged_generation_keeps_exports_and_size():
    r, calls = _render([(3, 1000, 600), (3, 1000, 600)])
    r._sync_swapchain_state()
    calls.clear()

    r._sync_swapchain_state()

    assert calls == []
    assert (r._width, r._height) == (1000, 600)


def test_recreation_at_same_size_still_releases_texture_exports():
    # Recovery after a failed frame rebuilds the swapchain at the same size;
    # Metal exports of the destroyed images must not be reused.
    r, calls = _render([(1, 1000, 600), (2, 1000, 600)])
    r._sync_swapchain_state()
    calls.clear()

    r._sync_swapchain_state()

    assert calls == ["release"]


def test_offscreen_target_keeps_its_viewport():
    r, calls = _render([(1, 1000, 600), (2, 1200, 700)])
    r._sync_swapchain_state()
    calls.clear()
    r._framebuffer_stack = [object()]

    r._sync_swapchain_state()

    assert (r._width, r._height) == (1200, 700)
    assert calls == ["release"]


def test_deferred_zero_size_swapchain_does_not_resize():
    r, calls = _render([(1, 1000, 600), (2, 0, 0)])
    r._sync_swapchain_state()
    calls.clear()

    r._sync_swapchain_state()

    assert (r._width, r._height) == (1000, 600)
    assert calls == ["release"]
