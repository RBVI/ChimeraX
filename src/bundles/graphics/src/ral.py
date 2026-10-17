"""Renderer abstraction layer shared by ChimeraX graphics backends.

The RAL separates four pieces that were historically collapsed into the
OpenGL ``Render`` object: backend discovery, a device, presentation surfaces,
and per-view render endpoints.  A session may own devices for more than one
backend at once, and the same CPU-side Drawing can therefore be presented by
OpenGL, Vulkan, WGPU, or Metal without changing global module state.

Backends are imported lazily.  Importing this module never loads Vulkan,
wgpu-native, Metal, or PyOpenGL.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, FrozenSet
from weakref import WeakKeyDictionary, WeakSet, finalize


_current_device = ContextVar("chimerax_graphics_device", default=None)


def current_graphics_device(required=True):
    """Return the device whose renderer is currently executing.

    This is execution-local state, not a selected process-wide backend.  It
    therefore remains correct when OpenGL and Vulkan endpoints are rendered
    alternately, and when rendering later moves to worker threads.
    """

    device = _current_device.get()
    if device is None and required:
        raise RuntimeError("no graphics device is active")
    return device


@contextmanager
def using_graphics_device(device):
    token = _current_device.set(device)
    try:
        yield device
    finally:
        _current_device.reset(token)


def graphics_device_for_renderer(renderer):
    """Return, or lazily attach, the device that owns ``renderer``.

    A few small legacy surfaces (notably the status bar) construct an OpenGL
    ``Render`` directly instead of creating a View endpoint.  Adopting that
    renderer here lets backend-neutral resources work there without making
    every overlay owner understand the RAL.
    """

    device = getattr(renderer, "graphics_device", None)
    if device is not None:
        return device
    context = getattr(renderer, "opengl_context", None)
    if context is None:
        raise RuntimeError("renderer has no graphics context")
    device = getattr(context, "graphics_device", None)
    if device is None:
        backend_name = getattr(
            renderer, "backend_name", getattr(context, "backend_name", "opengl")
        )
        device = GraphicsDevice(backend_registry.get(backend_name), context)
        context.graphics_device = device
        context.backend_name = backend_name
    renderer.graphics_device = device
    renderer.backend_name = device.backend.name
    renderer.capabilities = device.capabilities
    if device.primary_renderer is None:
        device.primary_renderer = renderer
    return device


class GraphicsFeature(str, Enum):
    """Features whose availability affects frame-graph construction."""

    MESH = "mesh"
    TEXTURE_2D = "texture-2d"
    TEXTURE_3D = "texture-3d"
    VOLUME_RAYCAST = "volume-raycast"
    SHADOW = "shadow"
    MULTISHADOW = "multishadow"
    SILHOUETTE = "silhouette"
    SELECTION_OUTLINE = "selection-outline"
    OFFSCREEN = "offscreen"
    READBACK = "readback"
    MULTI_SURFACE = "multi-surface"
    EXPLICIT_PRESENT = "explicit-present"
    DIRECT_DRAWING = "direct-drawing"
    FRONT_BUFFER_XOR = "front-buffer-xor"
    GTAO = "gtao"
    STEREO = "stereo"
    SCENE_RENDERER = "scene-renderer"


@dataclass(frozen=True)
class BackendCapabilities:
    """Declarative backend capabilities; never infer these with ``hasattr``."""

    features: FrozenSet[GraphicsFeature] = frozenset()

    def supports(self, feature: GraphicsFeature) -> bool:
        return feature in self.features

    def require(self, *features: GraphicsFeature) -> None:
        missing = [feature.value for feature in features if feature not in self.features]
        if missing:
            raise RuntimeError(
                "graphics backend is missing required features: %s"
                % ", ".join(missing)
            )


@dataclass(frozen=True)
class GraphicsBackend:
    """Lazy factories and metadata for one graphics implementation."""

    name: str
    capabilities: BackendCapabilities
    create_context: Callable
    create_renderer: Callable
    configure_surface: Callable
    draw_on_surface: Callable
    release_surface: Callable
    create_texture: Callable | None = None


class BackendRegistry:
    """Process-wide registry of backend *types*, not active backend state."""

    def __init__(self):
        self._backends = {}
        self._loaders = {}

    def register(self, backend: GraphicsBackend) -> None:
        self._backends[backend.name] = backend

    def register_lazy(self, name: str, loader: Callable[[], GraphicsBackend]) -> None:
        self._loaders[name] = loader

    def get(self, name: str) -> GraphicsBackend:
        name = name.lower()
        backend = self._backends.get(name)
        if backend is None:
            loader = self._loaders.get(name)
            if loader is None:
                raise ValueError("unknown graphics backend %r" % name)
            backend = loader()
            if backend.name != name:
                raise ValueError(
                    "graphics backend loader for %r returned %r"
                    % (name, backend.name)
                )
            self.register(backend)
        return backend

    def names(self):
        return tuple(sorted(set(self._backends) | set(self._loaders)))


class DeviceResourceCache:
    """GPU objects belonging to one device, keyed by CPU owner and revision.

    The values are deliberately opaque to the RAL.  A backend decides how to
    create and destroy them.  Weak owner keys ensure closing a model cannot
    leave an ``id(drawing)`` cache entry that is later reused for another
    model.
    """

    def __init__(self):
        self._owners = WeakKeyDictionary()
        self._finalizers = WeakKeyDictionary()
        self._unowned = {}

    def _entries(self, owner):
        if owner is None:
            return self._unowned
        entries = self._owners.get(owner)
        if entries is None:
            entries = {}
            self._owners[owner] = entries
            self._finalizers[owner] = finalize(
                owner, self._destroy_entries, entries
            )
        return entries

    def get(self, owner, key, revision=None):
        entry = self._entries(owner).get(key)
        if entry is None:
            return None
        stored_revision, resource, _destroy = entry
        return resource if stored_revision == revision else None

    def put(self, owner, key, resource, revision=None, destroy=None):
        entries = self._entries(owner)
        old = entries.pop(key, None)
        if old is not None:
            self._destroy_entry(old)
        entries[key] = (revision, resource, destroy)
        return resource

    def get_or_create(self, owner, key, revision, create, destroy=None):
        resource = self.get(owner, key, revision)
        if resource is None:
            resource = self.put(
                owner, key, create(), revision=revision, destroy=destroy
            )
        return resource

    def release(self, owner, key=None):
        entries = self._unowned if owner is None else self._owners.get(owner)
        if not entries:
            return
        if key is None:
            doomed = tuple(entries.values())
            entries.clear()
            if owner is not None:
                self._owners.pop(owner, None)
                owner_finalizer = self._finalizers.pop(owner, None)
                if owner_finalizer is not None:
                    owner_finalizer.detach()
        else:
            entry = entries.pop(key, None)
            doomed = () if entry is None else (entry,)
        for entry in doomed:
            self._destroy_entry(entry)

    def clear(self):
        for owner in tuple(self._owners):
            self.release(owner)
        self.release(None)

    @staticmethod
    def _destroy_entry(entry):
        _revision, resource, destroy = entry
        if destroy is not None:
            destroy(resource)

    @classmethod
    def _destroy_entries(cls, entries):
        for entry in tuple(entries.values()):
            cls._destroy_entry(entry)
        entries.clear()


@dataclass(eq=False)
class GraphicsDevice:
    """One backend device/context and all GPU resources allocated from it."""

    backend: GraphicsBackend
    context: object
    resources: DeviceResourceCache = field(default_factory=DeviceResourceCache)
    _closed: bool = False
    primary_renderer: object = field(default=None, init=False)
    _endpoints: WeakSet = field(default_factory=WeakSet, init=False)

    @classmethod
    def create(cls, backend_name, window, ui, *, stereo=False):
        backend = backend_registry.get(backend_name)
        backend.configure_surface(window, None)
        context = backend.create_context(window, ui, stereo)
        device = cls(backend, context)
        # Compatibility bridge for code receiving only the old context.
        context.graphics_device = device
        context.backend_name = backend.name
        return device

    @property
    def capabilities(self):
        return self.backend.capabilities

    def configure_surface(self, window):
        self.backend.configure_surface(window, self.context)

    def create_endpoint(self, view, window):
        if self._closed:
            raise RuntimeError("graphics device is closed")
        self.configure_surface(window)
        renderer = self.backend.create_renderer(self.context, window)
        renderer.graphics_device = self
        renderer.backend_name = self.backend.name
        renderer.capabilities = self.capabilities
        if self.primary_renderer is None:
            self.primary_renderer = renderer
        endpoint = RenderEndpoint(self, window, view, renderer)
        self._endpoints.add(endpoint)
        view.attach_render_endpoint(endpoint)
        return endpoint

    def _endpoint_closed(self, endpoint):
        self._endpoints.discard(endpoint)
        if self.primary_renderer is endpoint.renderer:
            self.primary_renderer = next(
                (e.renderer for e in self._endpoints if not e._closed), None
            )

    def close(self):
        if self._closed:
            return
        # Endpoints normally close themselves before their device.  Sweep any
        # stragglers here so no renderer can retain a native context pointer
        # after context.delete() destroys the device.
        for endpoint in tuple(self._endpoints):
            try:
                endpoint.close()
            except Exception:
                pass
        make_current = getattr(self.context, "make_current", None)
        if make_current is not None:
            make_current()
        with using_graphics_device(self):
            self.resources.clear()
        delete = getattr(self.context, "delete", None)
        if delete is not None:
            delete()
        self._closed = True


@dataclass(eq=False)
class RenderEndpoint:
    """Binding of one View to one presentation surface on one device."""

    device: GraphicsDevice
    surface: object
    view: object
    renderer: object
    _closed: bool = False

    @property
    def backend_name(self):
        return self.device.backend.name

    @property
    def capabilities(self):
        return self.device.capabilities

    def retarget(self):
        """Target this endpoint's surface and return the previous surface."""

        target = getattr(self.renderer, "set_target_window", None)
        if target is not None:
            return target(self.surface)
        shared = getattr(self.renderer, "use_shared_context", None)
        if shared is not None:
            return shared(self.surface)
        return None

    def restore(self, previous_surface):
        if previous_surface is None:
            return
        target = getattr(self.renderer, "set_target_window", None)
        if target is not None:
            target(previous_surface)
            return
        shared = getattr(self.renderer, "use_shared_context", None)
        if shared is not None:
            shared(previous_surface)

    def draw(self, draw):
        with using_graphics_device(self.device):
            return self.device.backend.draw_on_surface(
                self.renderer, self.surface, draw
            )

    def close(self):
        if self._closed:
            return
        # release_surface is for an auxiliary presentation window.  Releasing
        # the device's active primary surface is invalid (Vulkan explicitly
        # rejects it) and used to abort close before renderer.delete(), leaving
        # a renderer that later called into an already-destroyed VkDevice.
        is_primary = self.device.primary_renderer is self.renderer
        try:
            if not is_primary:
                self.device.backend.release_surface(self.renderer, self.surface)
        finally:
            try:
                delete = getattr(self.renderer, "delete", None)
                if delete is not None:
                    delete()
            finally:
                if getattr(self.view, "_render_endpoint", None) is self:
                    self.view._render_endpoint = None
                    self.view._render = None
                self._closed = True
                self.device._endpoint_closed(self)


class Texture:
    """Backend-neutral texture data with one native object per device.

    The public API intentionally matches the historical OpenGL ``Texture``.
    Data and sampler intent live here; OpenGL/Vulkan/Metal objects are cached
    by ``GraphicsDevice`` and rebuilt independently when the revision changes.
    """

    def __init__(
        self,
        data=None,
        dimension=2,
        cube_map=False,
        linear_interpolation=True,
        clamp_to_edge=False,
    ):
        self.data = self._contiguous(data)
        self.dimension = dimension
        self.is_cubemap = cube_map
        self._linear_interpolation = linear_interpolation
        self.clamp_to_edge = clamp_to_edge
        self.size = self._size_from_data(self.data, dimension)
        self._initialization = None
        self._revision = 0
        self._devices = WeakSet()
        self._deleted = False

    @staticmethod
    def _contiguous(data):
        if data is None:
            return None
        flags = getattr(data, "flags", None)
        return data if flags is None or flags["C_CONTIGUOUS"] else data.copy()

    @staticmethod
    def _size_from_data(data, dimension):
        shape = getattr(data, "shape", None)
        if shape is None or len(shape) < dimension:
            return None
        return tuple(int(s) for s in shape[dimension - 1 :: -1])

    @property
    def linear_interpolation(self):
        return self._linear_interpolation

    @linear_interpolation.setter
    def linear_interpolation(self, linear):
        linear = bool(linear)
        if getattr(self, "_linear_interpolation", linear) != linear:
            self._linear_interpolation = linear
            self._revision += 1
        else:
            self._linear_interpolation = linear

    def set_linear_interpolation(self, linear):
        self.linear_interpolation = linear

    def _native(self, device=None):
        if self._deleted:
            raise RuntimeError("texture has been deleted")
        if device is None:
            device = current_graphics_device()
        factory = device.backend.create_texture
        if factory is None:
            raise RuntimeError(
                "%s backend cannot create textures" % device.backend.name
            )

        revision = self._revision

        def create():
            native = factory(device, self)
            init = self._initialization
            if init is not None and self.data is None:
                method, args, kwargs = init
                getattr(native, method)(*args, **kwargs)
            return native

        def destroy(native):
            delete = getattr(native, "delete_texture", None)
            if delete is not None:
                delete()

        native = device.resources.get_or_create(
            self, "texture", revision, create, destroy=destroy
        )
        self._devices.add(device)
        return native

    def native_texture(self, device=None):
        """Return this texture materialized for ``device``."""

        return self._native(device)

    def initialize_rgba(self, size):
        self._initialize("initialize_rgba", size)

    def initialize_8_bit(self, size):
        self._initialize("initialize_8_bit", size)

    def initialize_depth(self, size, depth_compare_mode=True):
        self._initialize(
            "initialize_depth", size, depth_compare_mode=depth_compare_mode
        )

    def _initialize(self, method, size, **kwargs):
        self.size = tuple(size)
        self.data = None
        self._initialization = (method, (tuple(size),), kwargs)
        self._revision += 1
        device = current_graphics_device(required=False)
        if device is not None:
            self._native(device)

    def reload_texture(self, data, now=False):
        self.data = self._contiguous(data)
        self.size = self._size_from_data(self.data, self.dimension)
        self._initialization = None
        self._revision += 1
        device = current_graphics_device(required=False)
        if now and device is not None:
            native = self._native(device)
            native.reload_texture(self.data, now=True)

    def bind_texture(self, tex_unit=None, texture_unit=None):
        native = self._native()
        try:
            return native.bind_texture(
                tex_unit=tex_unit, texture_unit=texture_unit
            )
        except TypeError:
            unit = tex_unit if tex_unit is not None else texture_unit
            return native.bind_texture(unit)

    def unbind_texture(self, tex_unit=None, texture_unit=None):
        native = self._native()
        try:
            return native.unbind_texture(
                tex_unit=tex_unit, texture_unit=texture_unit
            )
        except TypeError:
            unit = tex_unit if tex_unit is not None else texture_unit
            return native.unbind_texture(unit)

    def normalization(self):
        return self._native().normalization()

    def read_texture_data(self, numpy_data_type=None):
        return self._native().read_texture_data(numpy_data_type)

    def prepare_for_sampling(self):
        prepare = getattr(self._native(), "prepare_for_sampling", None)
        if prepare is not None:
            return prepare()

    def delete_texture(self):
        if self._deleted:
            return
        current = current_graphics_device(required=False)
        if current is not None:
            current.resources.release(self, "texture")
        # Other devices retain their native object until their cache is
        # cleared. Destroying an OpenGL object from the wrong current
        # context is invalid, so eager cross-device deletion is unsafe.
        self.data = None
        self._deleted = True

    def __getattr__(self, name):
        # Transitional bridge for renderer internals that consume native
        # handles. Resolution still uses the execution-local device.
        if name in {"id", "_handle", "gl_target", "_format_code"}:
            device = current_graphics_device(required=False)
            if device is None:
                return None
            return getattr(self._native(device), name)
        raise AttributeError(name)


_COMMON_FEATURES = frozenset(
    {
        GraphicsFeature.MESH,
        GraphicsFeature.TEXTURE_2D,
        GraphicsFeature.TEXTURE_3D,
        GraphicsFeature.VOLUME_RAYCAST,
        GraphicsFeature.SHADOW,
        GraphicsFeature.MULTISHADOW,
        GraphicsFeature.SILHOUETTE,
        GraphicsFeature.SELECTION_OUTLINE,
        GraphicsFeature.OFFSCREEN,
        GraphicsFeature.READBACK,
        GraphicsFeature.MULTI_SURFACE,
    }
)


def _surface_type(name):
    def configure(window, context):
        import sys
        from Qt.QtGui import QSurface

        if name == "opengl":
            window.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        elif name == "vulkan":
            surface_type = (
                QSurface.SurfaceType.MetalSurface
                if sys.platform == "darwin"
                else QSurface.SurfaceType.VulkanSurface
            )
            window.setSurfaceType(surface_type)
        elif name == "wgpu":
            surface_type = (
                QSurface.SurfaceType.MetalSurface
                if sys.platform == "darwin"
                else QSurface.SurfaceType.RasterSurface
            )
            window.setSurfaceType(surface_type)
        else:
            window.setSurfaceType(QSurface.SurfaceType.MetalSurface)
        if context is not None:
            prepare = getattr(context, "prepare_window", None)
            if prepare is not None:
                prepare(window)

    return configure


def _draw_shared_surface(renderer, surface, draw, *, explicit=False):
    previous = renderer.use_shared_context(surface)
    try:
        return draw()
    finally:
        if explicit:
            try:
                renderer.done_current()
            finally:
                renderer.use_shared_context(previous)
        else:
            try:
                renderer.use_shared_context(previous)
            finally:
                renderer.done_current()


def _release_shared_surface(renderer, surface):
    release = getattr(renderer, "release_window", None)
    if release is not None:
        release(surface)


def _opengl_backend():
    def create_context(window, ui, stereo):
        from .opengl import OpenGLContext

        return OpenGLContext(window, ui.primaryScreen(), use_stereo=stereo)

    def create_renderer(context, window):
        from .opengl import Render

        return Render(context)

    def create_texture(device, texture):
        from .opengl import Texture as OpenGLTexture

        return OpenGLTexture(
            texture.data,
            dimension=texture.dimension,
            cube_map=texture.is_cubemap,
            linear_interpolation=texture.linear_interpolation,
            clamp_to_edge=texture.clamp_to_edge,
        )

    return GraphicsBackend(
        "opengl",
        BackendCapabilities(
            _COMMON_FEATURES
            | {GraphicsFeature.FRONT_BUFFER_XOR, GraphicsFeature.STEREO}
        ),
        create_context,
        create_renderer,
        _surface_type("opengl"),
        lambda renderer, surface, draw: _draw_shared_surface(
            renderer, surface, draw
        ),
        _release_shared_surface,
        create_texture,
    )


def _vulkan_backend():
    def create_context(window, ui, stereo):
        if stereo:
            raise RuntimeError("Vulkan does not support sequential stereo")
        from .vulkan import VulkanContext

        return VulkanContext(window, ui.primaryScreen())

    def create_renderer(context, window):
        from .vulkan import VulkanRender

        return VulkanRender(context)

    def create_texture(device, texture):
        from .vulkan import VulkanTexture

        return VulkanTexture(
            texture.data,
            dimension=texture.dimension,
            cube_map=texture.is_cubemap,
            linear_interpolation=texture.linear_interpolation,
            clamp_to_edge=texture.clamp_to_edge,
            render=device.primary_renderer,
        )

    return GraphicsBackend(
        "vulkan",
        BackendCapabilities(
            _COMMON_FEATURES
            | {
                GraphicsFeature.DIRECT_DRAWING,
                GraphicsFeature.EXPLICIT_PRESENT,
            }
        ),
        create_context,
        create_renderer,
        _surface_type("vulkan"),
        lambda renderer, surface, draw: _draw_shared_surface(
            renderer, surface, draw, explicit=True
        ),
        _release_shared_surface,
        create_texture,
    )


def _metal_backend():
    def create_context(window, ui, stereo):
        if stereo:
            raise RuntimeError("Metal does not support sequential stereo")
        from .metal import MetalContext

        return MetalContext(window, ui.primaryScreen())

    def create_renderer(context, window):
        from .metal import MetalRender

        return MetalRender(context)

    def create_texture(device, texture):
        from .metal import MetalTexture

        return MetalTexture(
            texture.data,
            dimension=texture.dimension,
            cube_map=texture.is_cubemap,
            linear_interpolation=texture.linear_interpolation,
            clamp_to_edge=texture.clamp_to_edge,
            render=device.primary_renderer,
        )

    return GraphicsBackend(
        "metal",
        BackendCapabilities(
            (_COMMON_FEATURES - {GraphicsFeature.SELECTION_OUTLINE})
            | {GraphicsFeature.DIRECT_DRAWING, GraphicsFeature.GTAO}
        ),
        create_context,
        create_renderer,
        _surface_type("metal"),
        lambda renderer, surface, draw: _draw_shared_surface(
            renderer, surface, draw
        ),
        _release_shared_surface,
        create_texture,
    )


def _wgpu_backend():
    def create_context(window, ui, stereo):
        from .wgpu import WgpuContext

        return WgpuContext(
            window, ui.primaryScreen(), use_stereo=stereo, session=ui.session
        )

    def create_renderer(context, window):
        from .wgpu import WgpuRender

        return WgpuRender(context, window)

    def create_texture(device, texture):
        from .wgpu import WgpuTexture

        return WgpuTexture(texture)

    return GraphicsBackend(
        "wgpu",
        BackendCapabilities(
            (_COMMON_FEATURES - {
                GraphicsFeature.OFFSCREEN,
                GraphicsFeature.READBACK,
            })
            | {
                GraphicsFeature.EXPLICIT_PRESENT,
                GraphicsFeature.SCENE_RENDERER,
            }
        ),
        create_context,
        create_renderer,
        _surface_type("wgpu"),
        lambda renderer, surface, draw: draw(),
        lambda renderer, surface: renderer.release_window(surface),
        create_texture,
    )


backend_registry = BackendRegistry()
backend_registry.register_lazy("opengl", _opengl_backend)
backend_registry.register_lazy("vulkan", _vulkan_backend)
backend_registry.register_lazy("metal", _metal_backend)
backend_registry.register_lazy("wgpu", _wgpu_backend)
