from types import SimpleNamespace


def test_macos_context_uses_qt_pixel_size_for_initial_surface(monkeypatch):
    import chimerax.graphics as graphics
    from chimerax.graphics import vulkan

    calls = {}

    def create_context(**kwargs):
        calls["context"] = kwargs
        return 33

    def create_surface(**kwargs):
        calls["surface"] = kwargs
        return 22

    native = SimpleNamespace(
        create_instance=lambda: 11,
        create_context=create_context,
        device_name=lambda **kwargs: "test device",
        max_3d_texture_size=lambda **kwargs: 2048,
    )
    metal = SimpleNamespace(create_surface=create_surface)
    monkeypatch.setattr(vulkan.sys, "platform", "darwin")
    monkeypatch.setattr(graphics, "_vulkan", native, raising=False)
    monkeypatch.setattr(graphics, "_metal_rt", metal, raising=False)

    window = SimpleNamespace(
        winId=lambda: 7,
        width=lambda: 500,
        height=lambda: 300,
        devicePixelRatio=lambda: 2.0,
    )
    vulkan.VulkanContext(window, object())

    assert calls["surface"] == {
        "instance": 11,
        "window_handle": 7,
        "width": 1000,
        "height": 600,
    }
    assert calls["context"]["width"] == 1000
    assert calls["context"]["height"] == 600
