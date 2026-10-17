"""Headset ABI and submission ordering without a connected headset."""
import ctypes as C
import sys
from types import SimpleNamespace

import pytest

from chimerax.vive.vulkan import GraphicsBindingVulkan, SwapchainImageVulkan
from chimerax.vive.openxr import XR


def test_openxr_vulkan_handles_are_not_truncated():
    if C.sizeof(C.c_void_p) != 8:
        pytest.skip('64-bit handle regression')
    handle = 0x123456789ABC
    binding = GraphicsBindingVulkan(dict(instance=handle, physical_device=handle + 1,
        device=handle + 2, queue_family_index=3, queue_index=0))
    assert binding.instance == handle
    assert binding.physical_device == handle + 1
    assert binding.device == handle + 2
    assert binding.type == 1000025000
    image = SwapchainImageVulkan()
    image.image = handle
    assert image.image == handle
    assert image.type == 1000025001


@pytest.mark.parametrize('copy_fails', [False, True])
def test_openxr_releases_acquired_image_after_submission(monkeypatch, copy_fails):
    events = []
    monkeypatch.setitem(sys.modules, 'xr', SimpleNamespace(
        SwapchainImageReleaseInfo=lambda: None,
        release_swapchain_image=lambda *a: events.append('release')))

    def copy(**kw):
        events.append(('copy', kw['image']))
        if copy_fails:
            raise RuntimeError('copy failed')

    renderer = SimpleNamespace(
        end_frame=lambda: events.append('submit'),
        pop_framebuffer=lambda: events.append('pop'), _ctx=1,
        _vulkan=SimpleNamespace(copy_framebuffer_to_image=copy))
    manager = XR()
    manager._frame_started = True
    manager._vulkan_render = renderer
    manager._swapchains = [17, 18]
    manager._framebuffers = [SimpleNamespace(_handle=20)]
    manager._swapchain_image_buffer = [[SimpleNamespace(image=30)]]
    manager._swapchain_format = 43
    manager._acquired_images = {0: 0}
    if copy_fails:
        with pytest.raises(RuntimeError, match='copy failed'):
            manager.release_opengl_render_target(renderer, 'left')
    else:
        manager.release_opengl_render_target(renderer, 'left')
    assert events == ['submit', ('copy', 30), 'pop', 'release']
    assert manager._acquired_images == {}
