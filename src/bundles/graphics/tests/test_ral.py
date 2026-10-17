from gc import collect
from types import SimpleNamespace

import numpy as np

from chimerax.graphics.ral import (
    BackendCapabilities,
    BackendRegistry,
    DeviceResourceCache,
    GraphicsBackend,
    GraphicsDevice,
    GraphicsFeature,
    Texture,
    backend_registry,
    graphics_device_for_renderer,
    using_graphics_device,
)


def _backend(name="test", draw_on_surface=None):
    return GraphicsBackend(
        name=name,
        capabilities=BackendCapabilities(frozenset({GraphicsFeature.MESH})),
        create_context=lambda window, ui, stereo: None,
        create_renderer=lambda context, window: SimpleNamespace(delete=lambda: None),
        configure_surface=lambda window, context: None,
        draw_on_surface=(
            draw_on_surface
            if draw_on_surface is not None
            else lambda renderer, surface, draw: draw()
        ),
        release_surface=lambda renderer, surface: None,
    )


def test_standalone_metal_is_not_registered():
    assert "metal" not in backend_registry.names()


def test_backend_registry_loads_backends_lazily_once():
    registry = BackendRegistry()
    calls = []
    registry.register_lazy("test", lambda: calls.append(True) or _backend())

    assert registry.get("TEST") is registry.get("test")
    assert calls == [True]
    assert registry.names() == ("test",)


def test_device_resource_cache_is_per_owner_and_revision():
    class Owner:
        pass

    owner = Owner()
    destroyed = []
    cache = DeviceResourceCache()
    cache.put(owner, "vertices", "v1", revision=1, destroy=destroyed.append)

    assert cache.get(owner, "vertices", revision=1) == "v1"
    assert cache.get(owner, "vertices", revision=2) is None

    cache.put(owner, "vertices", "v2", revision=2, destroy=destroyed.append)
    assert destroyed == ["v1"]
    del owner
    collect()
    assert destroyed == ["v1", "v2"]


def test_endpoint_draw_uses_backend_surface_strategy():
    calls = []

    def draw_on_surface(renderer, surface, draw):
        calls.append((renderer, surface))
        return draw()

    backend = _backend(draw_on_surface=draw_on_surface)
    context = SimpleNamespace(pixel_scale=lambda: 1)
    device = GraphicsDevice(backend, context)
    context.graphics_device = device
    view = SimpleNamespace(
        _render=None,
        _lighting=object(),
        _material=object(),
        _silhouette=SimpleNamespace(thickness=0),
        highlighted_thickness=0,
        redraw_needed=False,
    )

    def attach(endpoint):
        view._render = endpoint.renderer
        view._render_endpoint = endpoint

    view.attach_render_endpoint = attach
    endpoint = device.create_endpoint(view, "surface")

    assert endpoint.draw(lambda: "drawn") == "drawn"
    assert calls == [(endpoint.renderer, "surface")]


def test_primary_endpoint_close_does_not_release_active_surface():
    released = []
    deleted = []
    backend = _backend()
    backend = GraphicsBackend(
        **{
            **backend.__dict__,
            "release_surface": lambda renderer, surface: released.append(surface),
            "create_renderer": lambda context, window: SimpleNamespace(
                delete=lambda: deleted.append(window)
            ),
        }
    )
    context = SimpleNamespace(pixel_scale=lambda: 1)
    device = GraphicsDevice(backend, context)
    view = SimpleNamespace(
        _render=None,
        _lighting=object(),
        _material=object(),
        _silhouette=SimpleNamespace(thickness=0),
        redraw_needed=False,
    )
    view.attach_render_endpoint = lambda endpoint: (
        setattr(view, "_render", endpoint.renderer),
        setattr(view, "_render_endpoint", endpoint),
    )
    endpoint = device.create_endpoint(view, "primary")

    endpoint.close()

    assert released == []
    assert deleted == ["primary"]
    assert view._render is None


def test_logical_texture_has_independent_native_resource_per_device():
    created = []

    class NativeTexture:
        def __init__(self, backend_name, logical):
            self.backend_name = backend_name
            self.data = logical.data
            self.bind_count = 0
            created.append(self)

        def bind_texture(self, *args, **kwargs):
            self.bind_count += 1

        def delete_texture(self):
            pass

    def texture_backend(name):
        backend = _backend(name)
        return GraphicsBackend(
            **{
                **backend.__dict__,
                "create_texture": (
                    lambda device, logical: NativeTexture(name, logical)
                ),
            }
        )

    device_a = GraphicsDevice(texture_backend("a"), SimpleNamespace())
    device_b = GraphicsDevice(texture_backend("b"), SimpleNamespace())
    texture = Texture(np.zeros((2, 3, 4), dtype=np.uint8))

    with using_graphics_device(device_a):
        texture.bind_texture()
        texture.bind_texture()
    with using_graphics_device(device_b):
        texture.bind_texture()

    assert [native.backend_name for native in created] == ["a", "b"]
    assert [native.bind_count for native in created] == [2, 1]
    assert texture.size == (3, 2)


def test_direct_legacy_renderer_is_adopted_by_a_device():
    backend = _backend("legacy-test")
    backend_registry.register(backend)
    context = SimpleNamespace(backend_name="legacy-test")
    renderer = SimpleNamespace(
        backend_name="legacy-test", opengl_context=context
    )

    device = graphics_device_for_renderer(renderer)

    assert renderer.graphics_device is device
    assert context.graphics_device is device
    assert renderer.capabilities is backend.capabilities
    assert graphics_device_for_renderer(renderer) is device
