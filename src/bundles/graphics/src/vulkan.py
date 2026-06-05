"""
Vulkan rendering backend for ChimeraX.

Implements the Renderer protocol using the Vulkan API via the _vulkan
C extension module. All Vulkan API calls go through _vulkan -- this module
handles the Python-level state management, pipeline caching, and
translation between the Drawing/View interface and Vulkan commands.
"""

import sys
import numpy as np

# Vulkan targets all platforms: native on Linux/Windows, MoltenVK on macOS.


class VulkanContext:
    """
    Wraps the Vulkan instance, device, command queue, and swapchain.

    Created from a Qt QWindow's VkSurfaceKHR handle.
    Equivalent to OpenGLContext in the OpenGL backend.
    """

    def __init__(self, graphics_window, screen, use_stereo=False):
        from chimerax.graphics import _vulkan

        self.window = graphics_window
        self._screen = screen
        self._vulkan = _vulkan

        # Get native window handle from Qt (HWND on Windows, NSView* on macOS).
        window_handle = int(graphics_window.winId())

        if sys.platform == "darwin":
            from chimerax.graphics import _metal_rt
            instance = _vulkan.create_instance()
            surface = _metal_rt.create_surface(
                instance=instance, window_handle=window_handle
            )
            self._ctx = _vulkan.create_context(
                window_handle=window_handle, instance=instance, surface=surface
            )
        else:
            self._ctx = _vulkan.create_context(window_handle=window_handle)
        self._device_name = _vulkan.device_name(context=self._ctx)
        self._max_3d_texture_size = _vulkan.max_3d_texture_size(context=self._ctx)

        # Shader/pipeline programs cache (keyed by capabilities bitmask)
        self.shader_programs = {}
        self.current_shader_program = None
        self.current_viewport = None

        # Compatibility: OpenGL Bindings class registers with these
        from weakref import WeakSet

        self._bindings = WeakSet()
        self._framebuffers = WeakSet()
        self._framebuffer_color_bits = 8

    def make_current(self, window=None):
        """Vulkan doesn't need make_current -- context is always available."""
        return True

    def done_current(self):
        pass

    def swap_buffers(self, window=None):
        """Present the current frame."""
        self._vulkan.end_frame(context=self._ctx)

    def pixel_scale(self):
        return self.window.devicePixelRatio()

    def delete(self):
        if self._ctx is not None:
            self._vulkan.destroy_context(context=self._ctx)
            self._ctx = None


class VulkanShader:
    """
    A Vulkan pipeline state object, equivalent to an OpenGL shader program.

    Created for a specific combination of capability flags. Cached and
    reused for subsequent draws with the same capabilities.
    """

    def __init__(self, capabilities, pipeline_handle):
        self.capabilities = capabilities
        self._pipeline = pipeline_handle
        self._pipeline_depth_only = None
        self._pipeline_depth_prepass = None
        self._uniforms = {}  # name -> value, flushed before draw
        self.check_uniforms = False

    def set_integer(self, name, value):
        self._uniforms[name] = ("int", value)

    def set_float(self, name, value):
        self._uniforms[name] = ("float", value)

    def set_vector2(self, name, vector):
        self._uniforms[name] = ("float2", np.asarray(vector, dtype=np.float32))

    def set_vector3(self, name, vector):
        self._uniforms[name] = ("float3", np.asarray(vector, dtype=np.float32))

    def set_rgba(self, name, color):
        self._uniforms[name] = ("float4", np.asarray(color, dtype=np.float32))

    def set_float4(self, name, v4, count=1):
        self._uniforms[name] = ("float4", np.asarray(v4, dtype=np.float32))

    def set_float3_array(self, name, vectors):
        self._uniforms[name] = ("float3_array", np.asarray(vectors, dtype=np.float32))

    def set_matrix(self, name, matrix):
        self._uniforms[name] = ("matrix", np.asarray(matrix, dtype=np.float32))

    def validate_program(self):
        pass


class VulkanBuffer:
    """Wraps a Vulkan GPU buffer."""

    def __init__(self, vulkan_module, ctx, data=None):
        self._vulkan = vulkan_module
        self._ctx = ctx
        self._handle = None
        if data is not None:
            self.update_buffer_data(data)

    def update_buffer_data(self, data):
        arr = np.ascontiguousarray(data)
        if self._handle is None:
            self._handle = self._vulkan.create_buffer(context=self._ctx, data=arr)
        else:
            try:
                self._vulkan.update_buffer(buffer=self._handle, data=arr)
            except RuntimeError:
                # Buffer too small -- recreate
                self._vulkan.destroy_buffer(buffer=self._handle)
                self._handle = self._vulkan.create_buffer(context=self._ctx, data=arr)

    def delete_buffer(self):
        if self._handle is not None:
            self._vulkan.destroy_buffer(buffer=self._handle)
            self._handle = None


class VulkanFramebuffer:
    """Offscreen Vulkan render target used for image capture.

    Mirrors the shape of the OpenGL Framebuffer class (name, width,
    height, alpha, activate(), delete()) so view.image_rgba can use either
    interchangeably via Render.create_framebuffer(...).

    Construction defers the underlying VkImage / VkRenderPass creation
    until activate() is first called, so failed sizes can be detected
    cheaply (matches OpenGL's Framebuffer.activate -> bool semantics).
    """

    def __init__(self, name, render, width, height, alpha=False):
        self.name = name
        self.width = width
        self.height = height
        self.alpha = alpha
        self.viewport = (0, 0, width, height)
        self._render = render
        self._handle = None      # OffscreenFramebuffer* (C side)
        self._deleted = False

    def activate(self):
        if self._handle is not None:
            return True
        if self._deleted:
            return False
        m = self._render._vulkan
        ctx = self._render._ctx
        try:
            self._handle = m.create_offscreen_framebuffer(
                context=ctx,
                width=int(self.width),
                height=int(self.height),
                alpha=1 if self.alpha else 0,
            )
        except Exception:
            self._handle = None
            return False
        return self._handle is not None

    def delete(self):
        if self._handle is not None:
            try:
                self._render._vulkan.destroy_offscreen_framebuffer(
                    context=self._render._ctx, framebuffer=self._handle,
                )
            except Exception:
                pass
            self._handle = None
        self._deleted = True


# Module-level reference to the active VulkanRender. Set by VulkanRender's
# constructor; consulted by VulkanTexture so plain `Texture(...)` calls
# from backend-agnostic Python code (e.g. chimerax.map.image3d) can find a
# VkDevice without an explicit handle.
_active_vulkan_render = None

import os as _os
_VK_DRAW_DEBUG = bool(_os.environ.get("CHIMERAX_VK_DRAW_DEBUG"))
del _os


def _set_active_vulkan_render(render):
    global _active_vulkan_render
    _active_vulkan_render = render


# Format codes shared with the C side (see vulkan_texture.cpp).
_FORMAT_RGBA8       = 0
_FORMAT_R8          = 1
_FORMAT_DEPTH32F    = 2
_FORMAT_DEPTH32F_S8 = 3
_FORMAT_R32F        = 4
_FORMAT_R16_UNORM   = 5
_FORMAT_R16_SFLOAT  = 6
_FORMAT_R16_SNORM   = 7
_FORMAT_R8_SNORM    = 8


def _expand_to_rgba8(arr):
    """If `arr` has a trailing channel axis with 2 (LA) or 3 (RGB)
    components and uint8 dtype, return an equivalent RGBA8 array. Mirrors
    the OpenGL behaviour where GL_LUMINANCE_ALPHA / GL_RGB textures expand
    to vec4 when sampled."""
    import numpy as np
    if arr.dtype != np.uint8 or arr.ndim < 2:
        return arr
    last = arr.shape[-1]
    if last == 4:
        return arr
    if last == 3:
        # Add fully-opaque alpha.
        out = np.empty(arr.shape[:-1] + (4,), dtype=np.uint8)
        out[..., :3] = arr
        out[..., 3] = 255
        return out
    if last == 2:
        # Luminance + alpha -> RGB=L, A=A.
        out = np.empty(arr.shape[:-1] + (4,), dtype=np.uint8)
        out[..., 0] = arr[..., 0]
        out[..., 1] = arr[..., 0]
        out[..., 2] = arr[..., 0]
        out[..., 3] = arr[..., 1]
        return out
    return arr


def _infer_format_from_data(data):
    """Pick a (format_code, components) for a numpy data array.

    Mirrors the OpenGL Texture's shape/dtype inference so volume code
    (image3d.py) gets the same semantics on Vulkan."""
    import numpy as np
    arr = np.asarray(data)
    last = arr.shape[-1] if arr.ndim > 1 else 1
    # If the trailing axis is not a typical channel count, treat the data
    # as single-channel.
    components = last if last in (1, 2, 3, 4) else 1
    if components == 4 and arr.dtype == np.uint8:
        return _FORMAT_RGBA8, 4
    if components == 1:
        if arr.dtype == np.uint8:
            return _FORMAT_R8, 1
        if arr.dtype == np.int8:
            return _FORMAT_R8_SNORM, 1
        if arr.dtype == np.uint16:
            return _FORMAT_R16_UNORM, 1
        if arr.dtype == np.int16:
            return _FORMAT_R16_SNORM, 1
        if arr.dtype == np.float16:
            return _FORMAT_R16_SFLOAT, 1
        if arr.dtype == np.float32:
            return _FORMAT_R32F, 1
    # Fallback: convert to RGBA8 (the caller is expected to have run
    # _expand_to_rgba8 first to upgrade 2-/3-channel uint8 arrays).
    return _FORMAT_RGBA8, 4


class VulkanTexture:
    """
    Vulkan-backed texture implementing the chimerax.graphics.Texture
    protocol.  Construction is lazy: the underlying VkImage is allocated on
    the first `initialize_*`, `reload_texture`, or `bind_texture` call so
    callers that build a Texture from a numpy array still work.
    """

    def __init__(
        self,
        data=None,
        dimension=2,
        cube_map=False,
        linear_interpolation=True,
        clamp_to_edge=False,
    ):
        self.data = data
        self.dimension = dimension
        self.is_cubemap = cube_map
        self.linear_interpolation = linear_interpolation
        self.clamp_to_edge = clamp_to_edge
        self.size = None
        self._handle = None         # opaque pointer to TextureWithDevice
        self._format_code = None    # one of _FORMAT_* above
        self._components = None
        self._render = _active_vulkan_render
        # Compatibility with code that probes Texture.id:
        self.id = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_render(self):
        if self._render is None:
            self._render = _active_vulkan_render
        return self._render

    def _size_xyz(self, size):
        # Texture protocol uses (w,) / (w,h) / (w,h,d). Pad to 3.
        if self.dimension == 1:
            return (int(size[0]), 1, 1)
        if self.dimension == 2:
            return (int(size[0]), int(size[1]), 1)
        return (int(size[0]), int(size[1]), int(size[2]))

    def _allocate(self, size, format_code, components):
        if self._handle is not None:
            return
        r = self._ensure_render()
        if r is None:
            return
        w, h, d = self._size_xyz(size)
        m = r._vulkan
        ctx = r._ctx
        is_depth = format_code in (_FORMAT_DEPTH32F, _FORMAT_DEPTH32F_S8)
        self._handle = m.create_texture(
            context=ctx,
            width=w,
            height=h,
            format=format_code,
            linear_filter=1 if self.linear_interpolation else 0,
            depth_compare=1 if is_depth else 0,
            dimension=self.dimension,
            depth=d,
        )
        self.size = tuple(size)
        self._format_code = format_code
        self._components = components
        self.id = self._handle

    def _size_from_shape(self, shape, components):
        # OpenGL Texture stores shape as (Z, Y, X[, C]) for 3D, (Y, X[, C])
        # for 2D, (X[, C]) for 1D. Drop the channel axis (if any) and
        # reverse so size is (W,) / (W, H) / (W, H, D) — what the C-side
        # create_texture expects. The trailing channel axis is present
        # whenever shape has one more dimension than the texture dimension,
        # regardless of channel count (e.g. single-channel L8 plane data
        # arrives as (H, W, 1), not (H, W)).
        if len(shape) > self.dimension:
            shape = shape[:-1]
        return tuple(shape[::-1])

    def _allocate_from_data(self, data):
        import numpy as np
        arr = _expand_to_rgba8(np.asarray(data))
        format_code, components = _infer_format_from_data(arr)
        size = self._size_from_shape(arr.shape, components)
        self._allocate(size, format_code, components)

    def _upload(self, data):
        if self._handle is None:
            return
        r = self._ensure_render()
        if r is None:
            return
        import numpy as np
        # Expansion (LA/RGB -> RGBA) and contiguous repack so the staging
        # buffer copies the same number of bytes the texture format expects.
        arr = np.ascontiguousarray(_expand_to_rgba8(np.asarray(data)))
        r._vulkan.update_texture(texture=self._handle, data=arr)

    # ------------------------------------------------------------------
    # Texture protocol
    # ------------------------------------------------------------------
    def initialize_rgba(self, size):
        self._allocate(size, _FORMAT_RGBA8, 4)

    def initialize_8_bit(self, size):
        self._allocate(size, _FORMAT_R8, 1)

    def initialize_depth(self, size, depth_compare_mode=True):
        self._allocate(size, _FORMAT_DEPTH32F, 1)

    def reload_texture(self, data, now=False):
        # Re-derive the intended format/size from the actual data so we
        # don't try to copy (e.g.) 2-byte-per-texel uint16 data into an
        # RGBA8 texture that initialize_rgba allocated with a wrong
        # assumption. If the existing texture doesn't match, destroy and
        # re-allocate. (Mirrors OpenGL Texture.fill_opengl_texture.)
        import numpy as np
        arr = _expand_to_rgba8(np.ascontiguousarray(data))
        new_format, new_components = _infer_format_from_data(arr)
        new_size = self._size_from_shape(arr.shape, new_components)
        if self._handle is not None:
            if (self._format_code != new_format
                    or tuple(self.size or ()) != new_size):
                self.delete_texture()
        if self._handle is None:
            self._allocate(new_size, new_format, new_components)
        self._upload(arr)

    def bind_texture(self, tex_unit=None, texture_unit=None):
        # Some callers use `tex_unit`, others `texture_unit`.
        unit = tex_unit if tex_unit is not None else texture_unit
        # Lazy-init from constructor data on first bind.
        if self._handle is None and self.data is not None:
            d = self.data
            self.data = None
            self._allocate_from_data(d)
            self._upload(d)
        if self._handle is None:
            return
        # Default binding based on dimension matches the mesh shader's set 1
        # layout (see mesh.frag): tex2d=0, tex3d=1, texcube=2,
        # shadow=3, multishadow=4, ambient_tex3d=5, colormap_tex=6.
        if unit is None:
            if self.is_cubemap:
                unit = 2
            elif self.dimension == 1:
                unit = 6
            elif self.dimension == 3:
                unit = 1
            else:
                unit = 0
        r = self._ensure_render()
        if r is None:
            return
        r._vulkan.set_fragment_texture(
            context=r._ctx, texture=self._handle, index=int(unit)
        )

    def unbind_texture(self, tex_unit=None, texture_unit=None):
        # Vulkan descriptor sets are rebound per-draw via flush_descriptors,
        # so explicit unbind is a no-op here.
        pass

    def normalization(self):
        """Scale factor mapping a source data value to the [-1,1] / [0,1]
        range a sampler returns for this texture's format.  Volume rendering
        relies on this for `colormap_range` so the shader sees pre-scaled
        offsets.  Mirrors opengl.Texture.normalization()."""
        norms = {
            _FORMAT_RGBA8:      1.0 / 255.0,
            _FORMAT_R8:         1.0 / 255.0,
            _FORMAT_R8_SNORM:   1.0 / 127.0,
            _FORMAT_R16_UNORM:  1.0 / 65535.0,
            _FORMAT_R16_SNORM:  1.0 / 32767.0,
            _FORMAT_R16_SFLOAT: 1.0,
            _FORMAT_R32F:       1.0,
            _FORMAT_DEPTH32F:   1.0,
            _FORMAT_DEPTH32F_S8: 1.0,
        }
        return norms.get(self._format_code, 1.0)

    def delete_texture(self):
        if self._handle is None:
            return
        r = self._ensure_render()
        if r is None:
            self._handle = None
            self.id = None
            return
        r._vulkan.destroy_texture(texture=self._handle)
        self._handle = None
        self.id = None


class VulkanRender:
    """
    Vulkan rendering backend. Implements the Renderer protocol.

    This is the Vulkan equivalent of the Render class in opengl.py.
    Drawing and View call methods on this object to render the scene.
    """

    def __init__(self, vulkan_context):
        self._context = vulkan_context
        self._vulkan = vulkan_context._vulkan
        self._ctx = vulkan_context._ctx
        # Compatibility: Drawing._draw_geometry accesses renderer.opengl_context
        self.opengl_context = vulkan_context

        # Register as the active Vulkan render so plain `Texture(...)` calls
        # from backend-agnostic code find a VkDevice.  Also point
        # `chimerax.graphics.Texture` at VulkanTexture so volume code that
        # does `from chimerax.graphics import Texture` picks it up.
        _set_active_vulkan_render(self)
        try:
            import chimerax.graphics as _g
            _g.set_texture_class(VulkanTexture)
        except (ImportError, AttributeError):
            # set_texture_class may not exist if running against an older
            # graphics package; that's fine — set_active_vulkan_render is
            # still the primary mechanism.
            pass
        self._width = 800
        self._height = 600
        self._in_frame = False
        self._drawing_buffers = {}  # id(drawing) -> {name -> buffer_handle}
        self._pipeline_cache = {}  # (shader_name, constants_tuple, depth_only) -> pipeline_handle
        self._depth_only_rendering = False
        self._depth_prepass_rendering = False
        self._sphere_ao_active = False
        self._oit_rendering = False
        # OpenGL's initial state is GL_CULL_FACE disabled, and ChimeraX
        # only turns it on for two specific things (volume backing cube
        # and 3D ray-cast cube).  Vulkan bakes cull state into the
        # pipeline, so default off and cache per cull flag.
        self._face_culling_enabled = False
        # Pipeline-baked blend / depth-write state for volume rendering.
        # `_blend_op_max` -> BLEND_OP_MAX (maximum intensity projection).
        # `_blend_premult` -> srcFactor=ONE (transparent voxels not dimmed).
        # `_depth_write_enabled` -> write depth (transparent volume planes
        # disable this so they don't occlude later geometry).
        self._blend_op_max = False
        self._blend_premult = False
        self._depth_write_enabled = True

        # Capability system (same constants as OpenGL backend)
        from .renderer import (
            SHADER_LIGHTING,
            SHADER_LIGHTING_NORMALS,
            SHADER_DEPTH_CUE,
            SHADER_SHADOW,
            SHADER_MULTISHADOW,
            SHADER_SHIFT_AND_SCALE,
            SHADER_INSTANCING,
            SHADER_VERTEX_COLORS,
            SHADER_IMPOSTOR_SPHERE,
            SHADER_IMPOSTOR_CYLINDER,
        )

        self.enable_capabilities = 0
        self.disable_capabilities = 0

        # Matrix state
        self.current_projection_matrix = None
        self.current_view_matrix = None
        self.current_model_view_matrix = None
        self.current_model_matrix = None

        # Shader cache
        self.current_shader_program = None
        self._dummy_float_buffer = None
        self._dummy_color_buffer = None
        self._dummy_shift_scale_buffer = None
        self._dummy_matrix_buffer = None
        self._depth_compare_func = 1  # less

        # Framebuffer stack
        self._framebuffer_stack = []

        # Lighting (reuse existing Lighting class for parameters)
        from .opengl import Lighting, Material

        self.lighting = Lighting()
        self.material = Material()

        # Sub-systems (stub implementations initially)
        self.shadow = VulkanShadow(self)
        self.multishadow = VulkanMultishadow(self)
        self.silhouette = _StubSilhouette()
        self.outline = _StubOutline()
        self.offscreen = _StubOffscreen()
        self.blend = _StubBlend()

        self.model_color = (1, 1, 1, 1)
        self._background_color = (0, 0, 0, 1)
        self._near_far_clip = (0, 1)

        # Set capability constants as class attributes for Drawing access
        # (Drawing._shader_options references Render.SHADER_LIGHTING etc.)
        from . import renderer as _r

        for name in dir(_r):
            if name.startswith("SHADER_"):
                setattr(VulkanRender, name, getattr(_r, name))

    def delete(self):
        """Clean up renderer resources."""
        if self._in_frame:
            self.end_frame()
        # Restore module-level state we hijacked in __init__ so the next
        # backend (e.g. switching back to OpenGL at runtime) doesn't keep
        # constructing VulkanTextures bound to this dead render.
        global _active_vulkan_render
        if _active_vulkan_render is self:
            _active_vulkan_render = None
        try:
            import chimerax.graphics as _g
            from chimerax.graphics import opengl as _opengl
            if getattr(_g, "Texture", None) is VulkanTexture:
                _g.set_texture_class(_opengl.Texture)
        except (ImportError, AttributeError):
            pass

    # -------------------------------------------------------------------
    # Context management
    # -------------------------------------------------------------------

    def make_current(self):
        # Clean up any dangling frame from a previous failed draw
        if self._in_frame:
            self.end_frame()
        return self._context.make_current()

    def done_current(self):
        self.end_frame()  # Safety: end any dangling frame
        self._context.done_current()

    def swap_buffers(self):
        self.end_frame()

    def pixel_scale(self):
        return self._context.pixel_scale()

    def check_for_opengl_errors(self):
        return None  # No OpenGL errors in Vulkan

    def check_opengl_version(self, *args, **kw):
        pass  # Not applicable to Vulkan

    def set_default_framebuffer_size(self, width, height):
        s = self._context.pixel_scale()
        self._width = int(s * width)
        self._height = int(s * height)
        # Invalidate Metal RT texture/buffer export caches — the Vulkan
        # textures are about to be destroyed and recreated.
        self._exported_textures = None
        # On macOS, sync the CAMetalLayer's bounds before swapchain
        # recreation so MoltenVK sees the correct drawableSize.
        if sys.platform == "darwin":
            try:
                from chimerax.graphics import _metal_rt
                _metal_rt.resize_surface()
            except ImportError:
                pass
        self._context._vulkan.resize_drawable(
            context=self._ctx, width=self._width, height=self._height
        )
        self.set_viewport(0, 0, self._width, self._height)

    def opengl_version(self):
        return (0, 0)  # Not OpenGL

    def opengl_version_number(self):
        return 0

    def opengl_vendor(self):
        return "Vulkan"

    def opengl_renderer(self):
        return self._context._device_name

    @property
    def recording_opengl(self):
        return False

    @property
    def front_buffer_valid(self):
        return True

    # -- Render pass management -----------------------------------------------

    def begin_pass(self, descriptor):
        """Begin a render pass described by *descriptor* (a PassDescriptor)."""
        m = self._vulkan
        ctx = self._ctx

        self.begin_frame()

        # End any active encoder cleanly.
        if ctx:
            m.end_render_pass(context=ctx)

        if descriptor.depth_only and descriptor.depth_texture is not None:
            tex = descriptor.depth_texture
            # Vulkan depth textures are raw handles (long), not Texture objects.
            # Width/height come from the descriptor viewport or texture size.
            if descriptor.viewport:
                _, _, w, h = descriptor.viewport
            else:
                # Infer from the texture -- the multishadow/shadow code knows
                # the size and passes it via viewport.
                w = h = getattr(tex, "width", 1024)
            m.begin_depth_pass(context=ctx, depth_texture=tex, width=w, height=h)
            self._depth_only_rendering = True
        else:
            cc = descriptor.clear_color or self._background_color
            r, g, b, a = cc
            cd = 1 if descriptor.clear_depth is not None else 0
            m.begin_render_pass(
                context=ctx, clear_r=r, clear_g=g, clear_b=b, clear_a=a, clear_depth=cd
            )

        if descriptor.viewport is not None:
            self.set_viewport(*descriptor.viewport)

    def end_pass(self):
        """End the current render pass started by begin_pass()."""
        m = self._vulkan
        ctx = self._ctx
        m.end_render_pass(context=ctx)
        self._depth_only_rendering = False
        self.full_viewport()

    # -- Framebuffer stack (stub) ---------------------------------------------

    def default_framebuffer(self):
        return None  # TODO: Vulkan framebuffer

    def current_framebuffer(self):
        return _StubFramebuffer(self)

    def create_framebuffer(self, name, width, height, alpha=False):
        """Factory for an offscreen render target. Mirrors
        opengl.Framebuffer's constructor signature so view.image_rgba can
        ask the active renderer for an image-capture target without
        depending on backend specifics."""
        return VulkanFramebuffer(name, self, width, height, alpha=alpha)

    def push_framebuffer(self, fb):
        self._framebuffer_stack.append(fb)
        # If this is a VulkanFramebuffer, hand its handle to the C side so
        # the next begin_frame / begin_render_pass / end_frame use it
        # instead of the swapchain. Other framebuffer types (e.g. legacy
        # opengl ones some shadow / blend code may push) are tracked but
        # don't redirect the swapchain path.
        if isinstance(fb, VulkanFramebuffer) and fb._handle is not None:
            self._vulkan.set_active_offscreen(
                context=self._ctx, framebuffer=fb._handle)
            self.set_viewport(0, 0, fb.width, fb.height)

    def pop_framebuffer(self):
        fb = None
        if self._framebuffer_stack:
            fb = self._framebuffer_stack.pop()
        # Reactivate whatever offscreen target sits at the top of the
        # stack (if any), otherwise clear the offscreen state so the
        # swapchain path resumes. Reset the viewport to match the new
        # active target so subsequent draws use the right pixel space.
        new_handle = 0
        new_size = (self._width, self._height)
        for f in reversed(self._framebuffer_stack):
            if isinstance(f, VulkanFramebuffer) and f._handle is not None:
                new_handle = f._handle
                new_size = (f.width, f.height)
                break
        self._vulkan.set_active_offscreen(context=self._ctx, framebuffer=new_handle)
        self.set_viewport(0, 0, new_size[0], new_size[1])
        return fb

    def rendering_to_screen(self):
        return len(self._framebuffer_stack) == 0

    def frame_buffer_image(self, w, h, rgba=None, front_buffer=False):
        """Read back the active render target as a (h, w, 4) uint8 RGBA
        numpy array. Origin is top-left from Vulkan's perspective; the
        OpenGL contract is bottom-left, so we flip vertically before
        returning to match what view.image_rgba expects.

        Only works while a VulkanFramebuffer of size (w, h) is active --
        i.e. the caller did push_framebuffer(fb) and view.draw rendered
        into it. Reading the swapchain isn't supported here."""
        if rgba is None:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
        else:
            rgba.fill(0)

        active = None
        for f in reversed(self._framebuffer_stack):
            if isinstance(f, VulkanFramebuffer) and f._handle is not None:
                active = f
                break
        if active is None or active.width != w or active.height != h:
            return rgba

        try:
            # C side writes (h, w, 4) row-major top-down.
            self._vulkan.read_framebuffer(
                context=self._ctx, framebuffer=active._handle, data=rgba,
            )
        except Exception as e:
            import sys
            print(f"Vulkan frame_buffer_image error: {e}",
                  file=sys.stderr, flush=True)
            return rgba

        # OpenGL convention is row 0 = bottom; view.image() flips again
        # for PIL. Match that so callers don't see an upside-down image.
        return rgba[::-1, :, :]

    def set_offscreen_color_bits(self, bits):
        pass

    # -- Depth texture factory (used by render passes) -----------------------

    def create_depth_texture(self, width, height):
        """Create a depth texture for shadow mapping."""
        return self._vulkan.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=2,
            linear_filter=0,
            depth_compare=1,
        )

    def destroy_depth_texture(self, texture):
        """Destroy a depth texture created by create_depth_texture."""
        if texture is not None:
            self._vulkan.destroy_texture(texture=texture)

    def enable_shader_shadows(self, enable):
        self.enable_shader_capability(self.SHADER_SHADOW, enable)

    def enable_shader_multishadows(self, enable):
        self.enable_shader_capability(self.SHADER_MULTISHADOW, enable)

    def set_lighting_shader_capabilities(self):
        lp = self.lighting
        # Depth cue is not yet fully supported -- uniform values for
        # depth_cue_color and depth_cue_range are placeholders, so
        # enabling it would darken/blacken the scene.
        # self.enable_shader_capability(self.SHADER_DEPTH_CUE, lp.depth_cue)
        self.enable_shader_shadows(lp.shadows)
        self.enable_shader_multishadows(lp.multishadow > 0)

    def initialize_opengl(self, width, height):
        """Vulkan equivalent of OpenGL initialization."""
        # set_default_framebuffer_size applies pixel_scale internally.
        self.set_default_framebuffer_size(width, height)
        self.set_viewport(0, 0, self._width, self._height)

    def enable_depth_test(self, enable):
        pass

    def enable_backface_culling(self, enable):
        self._face_culling_enabled = bool(enable)

    def enable_blending(self, enable):
        pass

    def blend_alpha(self, alpha_factor=True):
        # alpha_factor=False switches src factor from SRC_ALPHA to ONE
        # (used by volume planes when dim_transparent_voxels is off).
        self._blend_premult = not bool(alpha_factor)

    def blend_add(self, f):
        pass

    def blend_max(self, enable):
        # Maximum intensity projection: BLEND_OP_MAX instead of ADD.
        self._blend_op_max = bool(enable)

    def enable_xor(self, enable):
        pass

    def write_depth(self, write):
        self._depth_write_enabled = bool(write)

    def depth_invert(self, invert):
        pass

    def draw_transparent(self, draw_depth, draw):
        """Simple single-layer transparency: draw depth first, then color."""
        draw_depth()
        draw()

    # ------------------------------------------------------------------
    # OIT (Weighted Blended Order-Independent Transparency)
    # ------------------------------------------------------------------

    def begin_oit_pass(self):
        """Begin OIT accumulation. Ends the current render pass, creates
        OIT resources if needed, and starts the OIT render pass."""
        w, h = self.render_size()
        if w <= 0 or h <= 0:
            return
        self._vulkan.create_oit_resources(context=self._ctx, width=w, height=h)
        self._vulkan.begin_oit_pass(context=self._ctx)
        self._oit_rendering = True

    def end_oit_pass(self):
        """End OIT accumulation and composite the result over the opaque
        scene. Resumes the main render pass with loaded color+depth."""
        self._oit_rendering = False
        self._vulkan.end_oit_pass(context=self._ctx)

        # Resume the main render pass so the composite quad (and any
        # subsequent passes) can draw to the swapchain.
        self._vulkan.begin_render_pass(
            context=self._ctx,
            clear_r=0, clear_g=0, clear_b=0, clear_a=1,
            clear_depth=0, load_color=1, load_depth=1)

        # Composite the OIT result over the opaque scene.
        accum_tex, reveal_tex = self._vulkan.get_oit_textures(context=self._ctx)
        self.draw_fullscreen_quad_two_textures(
            accum_tex, reveal_tex,
            "oit_composite", "main", "main",
        )

    def draw_xor_rectangle(self, *args, **kw):
        pass

    # ------------------------------------------------------------------
    # Volume raycasting. Mirrors the OpenGL backend's draw_volume_raycast:
    # flip the SHADER_VOLUME_RAYCASTING capability bit, render the
    # bounding-box geometry (the fragment shader reconstructs eye-space
    # rays from gl_FragCoord and walks the 3D texture), then restore
    # state. set_volume_parameters / set_bounding_box_planes are still
    # invoked by VolumeRaycastDrawing.set_shader_options, so we cache
    # them here and read them when building the raycast UBO in
    # _draw_single.
    # ------------------------------------------------------------------
    def draw_volume_raycast(self, drawing, draw_pass):
        from .drawing import Drawing
        from .renderer import SHADER_VOLUME_RAYCASTING, SHADER_DEPTH_CUE
        self.enable_backface_culling(True)
        prev_caps = self.enable_capabilities
        self.enable_capabilities = (prev_caps | SHADER_VOLUME_RAYCASTING) & ~SHADER_DEPTH_CUE
        try:
            Drawing.draw(drawing, self, draw_pass)
        finally:
            self.enable_capabilities = prev_caps
            self.enable_backface_culling(False)

    def _set_raycasting_background(self):
        # The Vulkan path reads _background_color when building the
        # raycast UBO at draw time, so no explicit push is needed.
        pass

    def _set_window_params(self):
        # The Vulkan path bakes render_size() into the raycast UBO at
        # draw time, so no explicit push is needed.
        pass

    def set_volume_parameters(self, step, full_region_min, full_region_max):
        self._raycast_step_size = step
        self._raycast_full_region_min = full_region_min
        self._raycast_full_region_max = full_region_max

    def set_bounding_box_planes(self, max_corner, min_corner):
        self._raycast_bbox_max = max_corner
        self._raycast_bbox_min = min_corner

    @property
    def current_viewport(self):
        return (0, 0, self._width, self._height)

    @current_viewport.setter
    def current_viewport(self, v):
        pass

    def set_stereo_buffer(self, eye_num):
        pass

    def record_opengl_calls(self, record=True):
        pass

    def max_framebuffer_size(self):
        return 16384

    def framebuffer_rgba_bits(self):
        return (8, 8, 8, 8)

    def framebuffer_depth_bits(self):
        return 32

    def wait_for_vsync(self, wait):
        pass

    def use_shared_context(self, window):
        """Switch Vulkan rendering to a different QWindow.

        Creates a VkSurfaceKHR on the target window if needed, swaps the
        context's surface/depth-texture, and updates the viewport.  Returns
        the previous window so the caller can restore it.
        """
        prev_window = self._context.window
        if window is prev_window:
            return prev_window

        # End any in-flight frame before switching surfaces.
        self.end_frame()

        surface_handle = int(window.winId())
        self._vulkan.set_surface_for_window(
            context=self._ctx, surface_handle=surface_handle)

        self._context.window = window

        # Update Python-side size from the pixel-scaled surface size.
        scale = window.devicePixelRatio()
        self._width = max(1, int(window.width() * scale))
        self._height = max(1, int(window.height() * scale))

        self.make_current()
        self.full_viewport()

        return prev_window

    def swap_interval(self):
        return 1

    def set_near_far_clip(self, near_far):
        self._near_far_clip = near_far

    # -------------------------------------------------------------------
    # Pipeline / shader selection
    # -------------------------------------------------------------------

    def shader(self, capabilities):
        capabilities |= self.enable_capabilities
        capabilities &= ~self.disable_capabilities

        # Per-drawing "off" flags override global renderer effects, matching
        # the OpenGL shader selection path.
        cap_pairs = (
            (self.SHADER_NO_SHADOW, self.SHADER_SHADOW),
            (self.SHADER_NO_MULTISHADOW, self.SHADER_MULTISHADOW),
            (self.SHADER_NO_DEPTH_CUE, self.SHADER_DEPTH_CUE),
            (self.SHADER_NO_CLIP_PLANES, self.SHADER_CLIP_PLANES),
        )
        for no_cap, cap in cap_pairs:
            if capabilities & no_cap:
                capabilities &= ~(cap | no_cap)

        programs = self._context.shader_programs
        if capabilities in programs:
            p = programs[capabilities]
        else:
            # Create new pipeline for this capability set
            p = VulkanShader(capabilities, None)  # TODO: actual pipeline creation
            programs[capabilities] = p

        self.current_shader_program = p
        self._context.current_shader_program = p
        return p

    def disable_shader_capabilities(self, mask):
        self.disable_capabilities = mask

    def enable_shader_capability(self, cap, enable):
        if enable:
            self.enable_capabilities |= cap
        else:
            self.enable_capabilities &= ~cap

    def draw_depth_only(self, depth_only=True):
        if depth_only:
            from .renderer import (
                SHADER_INSTANCING,
                SHADER_SHIFT_AND_SCALE,
                SHADER_TRANSPARENT_ONLY,
                SHADER_OPAQUE_ONLY,
                SHADER_CLIP_PLANES,
                SHADER_IMPOSTOR_SPHERE,
                SHADER_IMPOSTOR_CYLINDER,
            )

            d = ~(
                SHADER_INSTANCING
                | SHADER_SHIFT_AND_SCALE
                | SHADER_TRANSPARENT_ONLY
                | SHADER_OPAQUE_ONLY
                | SHADER_CLIP_PLANES
                | SHADER_IMPOSTOR_SPHERE
                | SHADER_IMPOSTOR_CYLINDER
            )
        else:
            d = 0
        self.disable_capabilities = d
        self._depth_prepass_rendering = bool(
            depth_only and not self._depth_only_rendering
        )

    # -------------------------------------------------------------------
    # Matrices
    # -------------------------------------------------------------------

    def set_projection_matrix(self, pm=None):
        if pm is not None:
            # Convert OpenGL projection (NDC z [-1,1]) to Vulkan (NDC z [0,1])
            # and flip Y axis (Vulkan Y points down, OpenGL Y points up).
            # vk_clip_z = 0.5 * gl_clip_z + 0.5 * gl_clip_w
            # vk_clip_y = -gl_clip_y
            # In column-major storage: math row 2 = pm[:, 2], row 3 = pm[:, 3]
            gl_pm = np.array(pm, dtype=np.float32)
            vk_pm = gl_pm.copy()
            vk_pm[:, 2] = 0.5 * gl_pm[:, 2] + 0.5 * gl_pm[:, 3]
            vk_pm[:, 1] = -gl_pm[:, 1]
            self.current_projection_matrix = vk_pm

    def set_view_matrix(self, vm):
        self.current_view_matrix = vm
        self.current_model_matrix = None
        self.current_model_view_matrix = None

    def set_model_matrix(self, model_matrix=None):
        if model_matrix is None:
            self.current_model_view_matrix = self.current_view_matrix
        else:
            # model_matrix is a Place object
            vm = self.current_view_matrix
            if vm is not None:
                mv = vm * model_matrix  # Place multiplication
                self.current_model_view_matrix = mv
            self.current_model_matrix = model_matrix
        if not self.lighting.move_lights_with_camera:
            self.update_lighting_parameters()

    # -------------------------------------------------------------------
    # Render state
    # -------------------------------------------------------------------

    def set_background_color(self, rgba):
        self._background_color = rgba

    def begin_frame(self):
        """Start a new frame -- get drawable and command buffer."""
        if not self._in_frame:
            self._vulkan.begin_frame(context=self._ctx)
            self._in_frame = True
            # Buffer handles may be recycled across frames (deferred
            # deletion + malloc reuse), so clear the export cache.
            self._exported_buffers = None

    def end_frame(self):
        """End the frame -- present and commit."""
        if self._in_frame:
            try:
                self._vulkan.end_frame(context=self._ctx)
            except Exception as e:
                import sys

                print(f"Vulkan end_frame error: {e}", file=sys.stderr)
            self._in_frame = False

    def draw_background(self, depth=True):
        try:
            self.begin_frame()
            r, g, b, a = self._background_color
            self._vulkan.begin_render_pass(
                context=self._ctx,
                clear_r=r,
                clear_g=g,
                clear_b=b,
                clear_a=a,
                clear_depth=1 if depth else 0,
            )
        except Exception as e:
            import sys

            print(f"Vulkan draw_background error: {e}", file=sys.stderr)
            self._in_frame = False

    def _draw_test_triangle(self):
        """Render a hardcoded triangle to verify the Vulkan pipeline."""
        if not hasattr(self, "_test_pipeline"):
            self._create_test_pipeline()
        if self._test_pipeline is None:
            return

        m = self._vulkan
        ctx = self._ctx
        m.set_pipeline(context=ctx, pipeline=self._test_pipeline)
        m.set_vertex_buffer(context=ctx, buffer=self._test_vbuf, index=0, offset=0)
        m.draw_indexed(
            context=ctx,
            index_buffer=self._test_ibuf,
            primitive_type=0,
            index_count=3,
            index_offset=0,
            instance_count=1,
        )

    def _create_test_pipeline(self):
        """Create a minimal shader pipeline for test rendering."""
        vert_source = """\
#version 450

layout(location = 0) in vec3 position;
layout(location = 0) out vec4 v_color;

void main() {
    gl_Position = vec4(position, 1.0);
    v_color = vec4(position.x * 0.5 + 0.5, position.y * 0.5 + 0.5, 0.5, 1.0);
}
"""
        frag_source = """\
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = v_color;
}
"""
        try:
            m = self._vulkan
            ctx = self._ctx
            self._test_pipeline = m.create_pipeline(
                context=ctx,
                vert_source=vert_source,
                frag_source=frag_source,
                has_depth=1,
                constants=[],
            )

            # Triangle vertices in NDC
            verts = np.array(
                [
                    [0.0, 0.5, 0.0],
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                ],
                dtype=np.float32,
            )
            indices = np.array([0, 1, 2], dtype=np.uint32)

            self._test_vbuf = m.create_buffer(context=ctx, data=verts)
            self._test_ibuf = m.create_buffer(context=ctx, data=indices)
        except Exception as e:
            import sys

            print(f"Vulkan test pipeline error: {e}", file=sys.stderr)
            self._test_pipeline = None

    def allow_equal_depth(self, equal):
        # Compare function values match vulkan_set_depth_stencil_state:
        # 1 = less, 3 = lessEqual.
        self._depth_compare_func = 3 if equal else 1
        self._set_current_depth_state()

    def _set_current_depth_state(self):
        self._vulkan.set_depth_stencil_state(
            context=self._ctx,
            depth_write=1,
            depth_test=1,
            compare_func=self._depth_compare_func,
        )

    def set_depth_range(self, near, far):
        pass  # TODO: viewport depth range

    def set_model_color(self, color=None):
        if color is not None:
            self.model_color = color

    def set_frame_number(self, frame=None):
        pass

    # -------------------------------------------------------------------
    # Viewport and framebuffer
    # -------------------------------------------------------------------

    def render_size(self):
        # Active offscreen framebuffer overrides window size so projection
        # matrix / viewport setup match the capture target.
        for f in reversed(self._framebuffer_stack):
            if isinstance(f, VulkanFramebuffer):
                return (f.width, f.height)
        return (self._width, self._height)

    def set_viewport(self, x, y, w, h):
        self._vulkan.set_viewport(context=self._ctx, x=x, y=y, width=w, height=h)

    def full_viewport(self):
        w, h = self.render_size()
        self.set_viewport(0, 0, w, h)

    def update_viewport(self):
        self.full_viewport()

    # -------------------------------------------------------------------
    # Lighting
    # -------------------------------------------------------------------

    def activate_lighting(self):
        """Upload lighting parameters to buffer index 1."""
        lp = self.lighting
        mp = self.material
        from chimerax.geometry import normalize_vector

        # Compute derived lighting values (same as OpenGL backend)
        move = None if lp.move_lights_with_camera else self.current_view_matrix

        kld = normalize_vector(lp.key_light_direction)
        if move is not None:
            kld = move.transform_vector(kld)
        ds = mp.diffuse_reflectivity * lp.key_light_intensity
        kdc = tuple(ds * c for c in lp.key_light_color)
        ss = mp.specular_reflectivity * lp.key_light_intensity
        ksc = tuple(ss * c for c in lp.key_light_color)
        fld = normalize_vector(lp.fill_light_direction)
        if move is not None:
            fld = move.transform_vector(fld)
        fds = mp.diffuse_reflectivity * lp.fill_light_intensity
        fdc = tuple(fds * c for c in lp.fill_light_color)
        ams = mp.ambient_reflectivity * lp.ambient_light_intensity
        ac = tuple(ams * c for c in lp.ambient_light_color)

        # Pack LightingParams struct matching common.glsl layout exactly:
        #   vec3 key_light_direction; float _pad0;       (offset 0)
        #   vec3 key_light_diffuse_color; float _pad1;   (offset 4)
        #   vec3 key_light_specular_color; float _pad2;  (offset 8)
        #   float specular_exponent; float _pad3,4,5;    (offset 12)
        #   vec3 fill_light_direction; float _pad6;      (offset 16)
        #   vec3 fill_light_diffuse_color; float _pad7;  (offset 20)
        #   vec3 ambient_color; float _pad8;             (offset 24)
        parts = []
        parts.append(np.array([*kld, 0], dtype=np.float32))
        parts.append(np.array([*kdc, 0], dtype=np.float32))
        parts.append(np.array([*ksc, 0], dtype=np.float32))
        parts.append(np.array([mp.specular_exponent, 0, 0, 0], dtype=np.float32))
        parts.append(np.array([*fld, 0], dtype=np.float32))
        parts.append(np.array([*fdc, 0], dtype=np.float32))
        parts.append(np.array([*ac, 0], dtype=np.float32))
        ao_strength = getattr(lp, "sphere_ao_strength", 2.0)
        parts.append(np.array([ao_strength, 0, 0, 0], dtype=np.float32))
        data = np.concatenate(parts).astype(np.float32)
        data = np.ascontiguousarray(data)

        # Always create a fresh buffer rather than updating in-place.
        # The previous frame's command buffer may still be reading the
        # old buffer on the GPU; in-place updates would race.
        # Vulkan retains the old buffer until its command buffer completes.
        self._lighting_buffer = self._vulkan.create_buffer(
            context=self._ctx, data=data
        )

    def update_lighting_parameters(self):
        self.activate_lighting()

    def set_depth_cue_parameters(self):
        pass

    def set_clip_parameters(self):
        pass

    # -------------------------------------------------------------------
    # Textures
    # -------------------------------------------------------------------

    def set_colormap(self, colormap, colormap_range, data_texture):
        # Bind the 1D colormap texture at set 1 binding 6 and upload the
        # (offset, scale) pair to the ColormapRangeBlock UBO at set 0
        # binding 5. Matches mesh.frag/impostor_*.frag's expected layout.
        if colormap is not None:
            colormap.bind_texture(tex_unit=6)
        # Match OpenGL Render.set_colormap: pre-scale by the data texture's
        # sampler normalization so the shader's
        #   cmap_u = (sample - x) * y
        # recovers (data_value - v0) / (v1 - v0).
        v0, v1 = colormap_range
        s = data_texture.normalization() if data_texture is not None else 1.0
        denom = s * (v1 - v0)
        x = s * float(v0)
        y = (1.0 / denom) if denom != 0.0 else 0.0
        data = np.array([float(x), float(y), 0.0, 0.0], dtype=np.float32)
        m = self._vulkan
        ctx = self._ctx
        if getattr(self, "_colormap_range_buf", None) is None:
            self._colormap_range_buf = m.create_buffer(context=ctx, data=data)
        else:
            try:
                m.update_buffer(buffer=self._colormap_range_buf, data=data)
            except RuntimeError:
                m.destroy_buffer(buffer=self._colormap_range_buf)
                self._colormap_range_buf = m.create_buffer(context=ctx, data=data)
        m.set_descriptor_buffer(
            context=ctx, buffer=self._colormap_range_buf, index=5, offset=0,
        )

    def set_ambient_texture_transform(self, transform):
        pass

    # -------------------------------------------------------------------
    # Draw commands
    # -------------------------------------------------------------------

    def draw_elements(self, mode, count, offset=0, instance_count=1):
        pass  # TODO: encode draw command

    def draw_drawing(self, drawing, capabilities, highlighted_only=False):
        """
        Vulkan draw path: takes a Drawing object's raw data and renders it.
        Called from Drawing._draw_geometry when the renderer is Vulkan.
        """
        if not self._in_frame:
            import sys

            return
        try:
            self._draw_drawing_impl(drawing, capabilities, highlighted_only)
        except Exception as e:
            import sys, traceback

            print(f"Vulkan draw_drawing error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _draw_drawing_impl(self, drawing, capabilities, highlighted_only=False):
        vertices = drawing.vertices
        triangles = drawing.triangles
        if vertices is None or triangles is None:
            return

        # Set model matrix from drawing's scene positions
        pos = drawing.positions
        use_instancing = (
            len(pos) > 1
            or pos.shift_and_scale_array() is not None
            or getattr(drawing, "_force_instancing", lambda: False)()
        )
        if use_instancing:
            spos = drawing.parent.get_scene_positions(displayed_only=True)
        else:
            spos = drawing.get_scene_positions(displayed_only=True)
        for p in spos:
            self.set_model_matrix(p)
            self._draw_single(drawing, capabilities)

    def _draw_single(self, drawing, capabilities):
        """Issue Vulkan draw call for one scene position."""
        m = self._vulkan
        ctx = self._ctx

        # Apply renderer-level capability mask up-front so downstream
        # branches (raycast detection, pipeline selection, UBO upload)
        # see the same bits self.shader(capabilities) ends up using.
        # Without this, draw_volume_raycast set the SHADER_VOLUME_RAYCASTING
        # bit on enable_capabilities but _draw_single read the unmodified
        # `capabilities` argument and fell through to the mesh shader.
        capabilities = (capabilities | self.enable_capabilities) & ~self.disable_capabilities

        vertices = drawing.vertices
        triangles = drawing.triangles

        # Get/init the per-drawing buffer cache up-front so the topology /
        # element-array computation can memoize derived data here. Without
        # this, Mesh display style recomputes edges via the masked_edges()
        # C call every single frame, which dominates frame time on large
        # iso-surfaces.
        draw_id = id(drawing)
        bufs = self._drawing_buffers.get(draw_id)
        if bufs is None:
            bufs = {}
            self._drawing_buffers[draw_id] = bufs

        # Pick primitive topology and the matching index array from the
        # drawing's display style. Mesh derives unique edges from the
        # triangle index buffer; Dot uses the unique vertex set. Impostor
        # shaders always render their own triangle geometry, so they keep
        # triangle topology regardless of display_style. Volume raycast
        # also stays on triangle topology (it draws the volume's bounding
        # box; the fragment shader does the per-pixel ray traversal).
        from .renderer import (SHADER_IMPOSTOR_SPHERE, SHADER_IMPOSTOR_CYLINDER,
                               SHADER_VOLUME_RAYCASTING)
        from .drawing import Drawing as _D
        style = getattr(drawing, "display_style", _D.Solid)
        impostor = bool(capabilities & (SHADER_IMPOSTOR_SPHERE | SHADER_IMPOSTOR_CYLINDER))
        raycast = bool(capabilities & SHADER_VOLUME_RAYCASTING)
        if impostor or raycast or style == _D.Solid:
            elements = triangles
            topology = 0
        elif style == _D.Mesh:
            tm = getattr(drawing, "triangle_mask", None)
            em = getattr(drawing, "edge_mask", None)
            edges_key = (id(triangles), id(tm), id(em))
            elements = bufs.get("_mesh_edges")
            if elements is None or bufs.get("_mesh_edges_key") != edges_key:
                elements = drawing.masked_edges
                bufs["_mesh_edges"] = elements
                bufs["_mesh_edges_key"] = edges_key
            topology = 1
        elif style == _D.Dot:
            dot_key = id(triangles)
            elements = bufs.get("_dot_indices")
            if elements is None or bufs.get("_dot_indices_key") != dot_key:
                elements = np.unique(triangles)
                bufs["_dot_indices"] = elements
                bufs["_dot_indices_key"] = dot_key
            topology = 2
        else:
            elements = triangles
            topology = 0
        if elements is None or len(elements) == 0:
            return

        self._sphere_ao_active = (
            not self._depth_only_rendering
            and (capabilities & SHADER_IMPOSTOR_SPHERE)
            and hasattr(drawing, "_ao_offset_buffer")
            and drawing._ao_offset_buffer is not None
        )

        # Get or create Vulkan pipeline for these capabilities
        shader = self.shader(capabilities)
        if (
            shader._pipeline is None
            or shader._pipeline_depth_only != self._depth_only_rendering
            or shader._pipeline_depth_prepass != self._depth_prepass_rendering
            or getattr(shader, "_pipeline_sphere_ao", False) != self._sphere_ao_active
            or getattr(shader, "_pipeline_face_culling", True) != self._face_culling_enabled
            or getattr(shader, "_pipeline_blend_op_max", False) != self._blend_op_max
            or getattr(shader, "_pipeline_blend_premult", False) != self._blend_premult
            or getattr(shader, "_pipeline_depth_write", True) != self._depth_write_enabled
            or getattr(shader, "_pipeline_topology", 0) != topology
        ):
            if capabilities & SHADER_VOLUME_RAYCASTING:
                self._ensure_shader_pipeline(
                    shader, "raycast", "raycast_vertex", "raycast_fragment",
                    topology=topology,
                )
            elif capabilities & SHADER_IMPOSTOR_SPHERE:
                self._ensure_shader_pipeline(
                    shader, "impostor_sphere", "sphere_vertex", "sphere_fragment",
                    topology=topology,
                )
            elif capabilities & SHADER_IMPOSTOR_CYLINDER:
                self._ensure_shader_pipeline(
                    shader, "impostor_cylinder", "cylinder_vertex", "cylinder_fragment",
                    topology=topology,
                )
            else:
                self._ensure_shader_pipeline(
                    shader, "mesh", "mesh_vertex", "mesh_fragment",
                    topology=topology,
                )
        if shader._pipeline is None:
            return  # Cannot render without a pipeline

        m = self._vulkan
        ctx = self._ctx

        # Bind any texture/colormap the drawing exposes. Mirrors the work
        # Drawing._draw_geometry does in the OpenGL path; with Vulkan we
        # never reach that code because Drawing routes to draw_drawing.
        # 2D plane volumes store one texture per slice in `multitexture`;
        # `texture` is None in that case and we draw each slice separately
        # (handled at the draw_indexed call below).
        tex = getattr(drawing, "texture", None)
        mtex = getattr(drawing, "multitexture", None)
        if tex is not None:
            tex.bind_texture()
        cmap = getattr(drawing, "colormap", None)
        if cmap is not None:
            if tex is not None:
                self.set_colormap(cmap, drawing.colormap_range, tex)
            elif mtex:
                self.set_colormap(cmap, drawing.colormap_range, mtex[0])
        atex = getattr(drawing, "ambient_texture", None)
        if atex is not None:
            # Ambient 3D texture goes to set 1 binding 5 in mesh.frag.
            atex.bind_texture(tex_unit=5)
            tform = getattr(drawing, "ambient_texture_transform", None)
            if tform is not None:
                self.set_ambient_texture_transform(tform)

        # Mirror the OpenGL Drawing._draw_geometry path: give the drawing
        # a chance to push per-draw state into the renderer. For volume
        # raycast this populates _raycast_step_size / _raycast_bbox_min /
        # _raycast_full_region_min etc., which the raycast UBO build below
        # consumes.
        try:
            drawing.set_shader_options(self)
        except Exception:
            pass

        # Build and bind the volume-raycast UBO at set 0 binding 6 if the
        # raycast pipeline is active. Done per draw because the inverse
        # view/projection matrices change every frame as the camera moves.
        if raycast:
            self._upload_raycast_uniforms(bufs)

        m.set_pipeline(context=ctx, pipeline=shader._pipeline)
        self._set_current_depth_state()
        self._bind_dummy_vertex_buffers()

        # Per-vertex / per-instance buffer upload helper. Each call uploads
        # the source array's bytes to a GPU buffer; skip both the upload
        # AND the make_data array conversion when the source array's
        # identity hasn't changed since the last frame. ChimeraX rebuilds
        # arrays on real changes (vertex_colors = ...), so id() catches
        # the common case where geometry is static. Without this
        # short-circuit, a static iso-surface in Mesh display style
        # re-uploads every buffer every frame.
        def _upload(name, count_key, source, make_data, vk_index):
            srcid = id(source)
            srclen = len(source)
            if (name in bufs
                    and bufs.get(name + "_srcid") == srcid
                    and bufs.get(count_key) == srclen):
                m.set_vertex_buffer(context=ctx, buffer=bufs[name],
                                    index=vk_index, offset=0)
                return
            data = make_data()
            if name in bufs and bufs.get(count_key) == srclen:
                m.update_buffer(buffer=bufs[name], data=data)
            else:
                if name in bufs:
                    m.destroy_buffer(buffer=bufs[name])
                bufs[name] = m.create_buffer(context=ctx, data=data)
                bufs[count_key] = srclen
            bufs[name + "_srcid"] = srcid
            m.set_vertex_buffer(context=ctx, buffer=bufs[name],
                                index=vk_index, offset=0)

        # Zero-copy GPU path: if the drawing has pre-allocated VMA buffers
        # (e.g. from flying edges writing directly into GPU memory), bind
        # them directly and skip the upload.
        vk_bufs = getattr(drawing, '_vk_buffers', None)
        if vk_bufs and topology == 0:
            m.set_vertex_buffer(context=ctx, buffer=vk_bufs['vertices'],
                                index=10, offset=0)
            normals = drawing.normals
            if normals is not None and 'normals' in vk_bufs:
                m.set_vertex_buffer(context=ctx, buffer=vk_bufs['normals'],
                                    index=12, offset=0)
            bufs["indices"] = vk_bufs['indices']
            bufs["icount"] = len(drawing.triangles) * 3
            bufs["itopology"] = 0
            bufs["_indices_srcid"] = id(drawing.triangles)
            _used_gpu_bufs = True
        else:
            _used_gpu_bufs = False
            _upload("vertices", "vcount", vertices,
                    lambda: np.ascontiguousarray(vertices, dtype=np.float32), 10)

            # Index buffer. `elements` is the topology-specific index array:
            # (N,3) triangles for Solid, (M,2) edges for Mesh, (P,) points for
            # Dot. The element-array identity is stable across frames (the Mesh
            # edges are memoized above), so we can short-circuit the upload.
            elements_id = id(elements)
            if not (bufs.get("indices") is not None
                    and bufs.get("_indices_srcid") == elements_id
                    and bufs.get("itopology") == topology):
                idata = np.ascontiguousarray(np.asarray(elements).reshape(-1),
                                             dtype=np.uint32)
                idata_len = len(idata)
                if ("indices" not in bufs
                        or bufs.get("icount") != idata_len
                        or bufs.get("itopology") != topology):
                    if "indices" in bufs:
                        m.destroy_buffer(buffer=bufs["indices"])
                    bufs["indices"] = m.create_buffer(context=ctx, data=idata)
                    bufs["icount"] = idata_len
                    bufs["itopology"] = topology
                else:
                    m.update_buffer(buffer=bufs["indices"], data=idata)
                bufs["_indices_srcid"] = elements_id

        # Normals (buffer index 12)
        normals = drawing.normals
        if normals is not None and not _used_gpu_bufs:
            _upload("normals", "ncount", normals,
                    lambda: np.ascontiguousarray(normals, dtype=np.float32), 12)

        # Texture coordinates (vertex binding 11). Pipeline binding 11 is
        # vec3 (12-byte stride): .xy is sampled for 2D textures, .xyz for 3D
        # / cubemap. Drawings expose either (N,2) tex coords (planes,
        # outline images) or (N,3) (volume ray-cast box); pad the 2D case
        # to 3 floats so the stride matches the binding.
        tcoords = getattr(drawing, "texture_coordinates", None)
        if tcoords is not None:
            def _make_tdata():
                tarr = np.asarray(tcoords, dtype=np.float32)
                if tarr.ndim == 2 and tarr.shape[1] == 2:
                    pad = np.zeros((tarr.shape[0], 1), dtype=np.float32)
                    tarr = np.concatenate((tarr, pad), axis=1)
                return np.ascontiguousarray(tarr)
            _upload("tcoords", "tccount", tcoords, _make_tdata, 11)

        # Color buffer at vertex binding 13. Drawings expose two color
        # arrays: per-vertex (ribbons / meshes via `vertex_colors`) and
        # per-instance (atoms / cylinders via `colors`). Prefer
        # vertex_colors when present so meshes get a per-vertex stride
        # buffer matching the per-vertex pipeline binding rate set by the
        # mesh shader path.
        vcolors = getattr(drawing, "vertex_colors", None)
        colors = vcolors if vcolors is not None else drawing.colors
        if colors is not None and len(colors) > 0:
            _upload("colors", "ccount", colors,
                    lambda: np.ascontiguousarray(colors, dtype=np.uint8), 13)

        # Upload Uniforms struct to buffer 0 (vertex + fragment)
        udata = self._build_uniforms(drawing)
        if udata is not None:
            # During the shadow depth pass, always create a fresh buffer to
            # avoid a race condition: the main pass would overwrite the
            # shared buffer before the GPU finishes reading it for the
            # shadow draw calls.
            if self._depth_only_rendering:
                ubuf = m.create_buffer(context=ctx, data=udata)
                shadow_bufs = bufs.setdefault("_shadow_uniforms", [])
                shadow_bufs.append(ubuf)  # prevent GC until next frame
            else:
                # Clear any shadow-pass buffers from last frame
                if "_shadow_uniforms" in bufs:
                    for sb in bufs["_shadow_uniforms"]:
                        m.destroy_buffer(buffer=sb)
                    bufs["_shadow_uniforms"] = []
                if "uniforms" not in bufs:
                    bufs["uniforms"] = m.create_buffer(context=ctx, data=udata)
                else:
                    try:
                        m.update_buffer(buffer=bufs["uniforms"], data=udata)
                    except RuntimeError:
                        m.destroy_buffer(buffer=bufs["uniforms"])
                        bufs["uniforms"] = m.create_buffer(context=ctx, data=udata)
                ubuf = bufs["uniforms"]
            m.set_descriptor_buffer(context=ctx, buffer=ubuf, index=0, offset=0)

        # Lighting params at buffer 1 (vertex + fragment)
        if hasattr(self, "_lighting_buffer"):
            m.set_descriptor_buffer(
                context=ctx, buffer=self._lighting_buffer, index=1, offset=0
            )

        # Multishadow transform block at buffer 2.
        if (
            not self._depth_only_rendering
            and isinstance(self.multishadow, VulkanMultishadow)
            and self.multishadow._matrix_buffer is not None
        ):
            m.set_descriptor_buffer(
                context=ctx, buffer=self.multishadow._matrix_buffer, index=2, offset=0
            )

        # Shadow map texture at index 3 (skip during depth-only pass to avoid
        # read-while-write on the shadow texture that is the render target).
        if not self._depth_only_rendering and isinstance(self.shadow, VulkanShadow):
            self.shadow.bind_shadow_texture(self)
        if not self._depth_only_rendering and isinstance(
            self.multishadow, VulkanMultishadow
        ):
            self.multishadow.bind_multishadow_texture(self)

        # Instance data
        pos = drawing.positions
        use_instancing = len(pos) > 1 or pos.shift_and_scale_array() is not None
        instance_count = len(pos) if use_instancing else 1

        if _VK_DRAW_DEBUG:
            from .renderer import SHADER_IMPOSTOR_CYLINDER as _SIC
            if capabilities & _SIC:
                vcolors = getattr(drawing, "vertex_colors", None)
                dcolors = getattr(drawing, "colors", None)
                sas_arr = pos.shift_and_scale_array()
                mats = pos.opengl_matrices() if sas_arr is None and len(pos) > 1 else None
                import sys
                name = getattr(drawing, "name", repr(drawing))
                parent = getattr(drawing, "parent", None)
                pname = getattr(parent, "name", None) if parent is not None else None
                print(
                    f"[vk-cyl] drawing={name!r} parent={pname!r} "
                    f"verts={len(vertices)} tris={len(triangles)} "
                    f"pos={len(pos)} instance_count={instance_count} "
                    f"sas={None if sas_arr is None else sas_arr.shape} "
                    f"matrices={None if mats is None else mats.shape} "
                    f"vcolors={None if vcolors is None else vcolors.shape} "
                    f"colors={None if dcolors is None else dcolors.shape} "
                    f"caps=0x{capabilities:x}",
                    file=sys.stderr,
                )

        # Shift-and-scale instances (buffer index 4)
        sas = pos.shift_and_scale_array()
        if sas is not None:
            sasdata = np.ascontiguousarray(sas, dtype=np.float32)
            if "sas" not in bufs or bufs.get("sas_count") != len(sas):
                if "sas" in bufs:
                    m.destroy_buffer(buffer=bufs["sas"])
                bufs["sas"] = m.create_buffer(context=ctx, data=sasdata)
                bufs["sas_count"] = len(sas)
            else:
                m.update_buffer(buffer=bufs["sas"], data=sasdata)
            m.set_vertex_buffer(context=ctx, buffer=bufs["sas"], index=14, offset=0)

        # Instance matrices (buffer index 5). Always upload when there's no
        # shift-and-scale array, even for a single instance — the impostor
        # cylinder vertex shader reads inst_col0..3 unconditionally, and
        # skipping the bind would leave slot 15 pointing at a previous
        # drawing's matrix data.
        if sas is None:
            matrices = pos.opengl_matrices()
            if matrices is not None:
                mdata = np.ascontiguousarray(matrices, dtype=np.float32)
                if "matrices" not in bufs or bufs.get("mcount") != len(matrices):
                    if "matrices" in bufs:
                        m.destroy_buffer(buffer=bufs["matrices"])
                    bufs["matrices"] = m.create_buffer(context=ctx, data=mdata)
                    bufs["mcount"] = len(matrices)
                else:
                    m.update_buffer(buffer=bufs["matrices"], data=mdata)
                m.set_vertex_buffer(
                    context=ctx, buffer=bufs["matrices"], index=15, offset=0
                )

        # Sphere-sphere AO buffers at descriptor buffer 3 and 4.
        if self._sphere_ao_active:
            m.set_descriptor_buffer(
                context=ctx, buffer=drawing._ao_offset_buffer, index=3, offset=0
            )
            m.set_descriptor_buffer(
                context=ctx, buffer=drawing._ao_neighbor_buffer, index=4, offset=0
            )

        # Draw. For 2D plane volumes with `multitexture`, bind each slice's
        # texture and draw its 6-index quad in turn. Index buffer holds the
        # quad triangles for all slices concatenated; each slice gets
        # total_indices / num_textures indices. Multitexture is only used
        # for Solid-style volume planes, so we use the triangle count.
        # `icount` is whatever the index buffer was last (re)allocated with.
        index_count = bufs["icount"]
        if mtex:
            nt = len(mtex)
            ne = index_count // nt
            reverse = bool(getattr(drawing, "multitexture_reverse_order", False))
            order = range(nt - 1, -1, -1) if reverse else range(nt)
            for ti in order:
                mtex[ti].bind_texture()
                m.draw_indexed(
                    context=ctx,
                    index_buffer=bufs["indices"],
                    primitive_type=0,
                    index_count=ne,
                    index_offset=ti * ne,
                    instance_count=instance_count,
                )
        else:
            m.draw_indexed(
                context=ctx,
                index_buffer=bufs["indices"],
                primitive_type=0,
                index_count=index_count,
                index_offset=0,
                instance_count=instance_count,
            )

    def _upload_raycast_uniforms(self, bufs):
        """Pack the volume-raycast UBO and bind it at set 0 binding 6.

        The raycast fragment shader needs the inverse view + projection
        matrices to reconstruct an eye-space ray from gl_FragCoord, plus
        the volume's bounding box (world coords), the data-extent region
        used to map world coords back into [0,1] texture space, the ray
        step size, the window size (also for gl_FragCoord -> NDC), and
        the scene background color so the composite has somewhere to
        blend the leftover alpha onto. set_volume_parameters /
        set_bounding_box_planes (called via drawing.set_shader_options
        just above) feed the bbox + region; everything else lives on the
        renderer.
        """
        m = self._vulkan
        ctx = self._ctx

        # Inverse view (camera -> scene). Place objects expose .inverse().
        vm = self.current_view_matrix
        if vm is None:
            inv_view = np.eye(4, dtype=np.float32)
        else:
            try:
                inv_view = np.asarray(vm.inverse().opengl_matrix(), dtype=np.float32)
            except Exception:
                inv_view = np.eye(4, dtype=np.float32)

        # Inverse projection (clip -> view) for the (already Vulkan-flipped)
        # current projection matrix.
        pm = self.current_projection_matrix
        if pm is None:
            inv_proj = np.eye(4, dtype=np.float32)
        else:
            try:
                inv_proj = np.linalg.inv(np.asarray(pm, dtype=np.float32))
            except np.linalg.LinAlgError:
                inv_proj = np.eye(4, dtype=np.float32)

        bg = self._background_color
        if bg is None:
            bg = (0.0, 0.0, 0.0, 0.0)
        bbox_min = getattr(self, "_raycast_bbox_min", (-1.0, -1.0, -1.0))
        bbox_max = getattr(self, "_raycast_bbox_max", (1.0, 1.0, 1.0))
        full_min = getattr(self, "_raycast_full_region_min", bbox_min)
        full_max = getattr(self, "_raycast_full_region_max", bbox_max)
        step = float(getattr(self, "_raycast_step_size", 0.01))
        win_w, win_h = self.render_size()

        # std140 layout: each row below is 16 bytes (one float4-equivalent).
        parts = [
            np.asarray(inv_view, dtype=np.float32).reshape(-1),
            np.asarray(inv_proj, dtype=np.float32).reshape(-1),
            np.asarray(bg, dtype=np.float32).reshape(-1),
            np.array([bbox_min[0], bbox_min[1], bbox_min[2], step], dtype=np.float32),
            np.array([bbox_max[0], bbox_max[1], bbox_max[2], 0.0], dtype=np.float32),
            np.array([full_min[0], full_min[1], full_min[2], 0.0], dtype=np.float32),
            np.array([full_max[0], full_max[1], full_max[2], 0.0], dtype=np.float32),
            np.array([float(win_w), float(win_h), 0.0, 0.0], dtype=np.float32),
        ]
        data = np.ascontiguousarray(np.concatenate(parts))

        if "raycast_ubo" not in bufs:
            bufs["raycast_ubo"] = m.create_buffer(context=ctx, data=data)
        else:
            try:
                m.update_buffer(buffer=bufs["raycast_ubo"], data=data)
            except RuntimeError:
                m.destroy_buffer(buffer=bufs["raycast_ubo"])
                bufs["raycast_ubo"] = m.create_buffer(context=ctx, data=data)
        m.set_descriptor_buffer(
            context=ctx, buffer=bufs["raycast_ubo"], index=6, offset=0,
        )

    def _bind_dummy_vertex_buffers(self):
        """
        Bind defaults for optional vertex attributes.  The Vulkan pipeline
        descriptor includes texture coordinates, normals, colors, sphere
        shift/scale, and cylinder placement matrices for all shader variants;
        drawings only provide the subset their shader reads.
        """
        m = self._vulkan
        ctx = self._ctx
        if self._dummy_float_buffer is None:
            # Stride 12 (vec3) on bindings 11/12; the shader may read up to
            # one entry per vertex on impostor primitives that emit a fan
            # of vertices.  Size generously so we never read past the end.
            fdata = np.zeros((1024,), dtype=np.float32)
            self._dummy_float_buffer = m.create_buffer(context=ctx, data=fdata)
        if self._dummy_color_buffer is None:
            cdata = np.array([255, 255, 255, 255], dtype=np.uint8)
            self._dummy_color_buffer = m.create_buffer(context=ctx, data=cdata)
        if self._dummy_shift_scale_buffer is None:
            sdata = np.array([0, 0, 0, 1], dtype=np.float32)
            self._dummy_shift_scale_buffer = m.create_buffer(context=ctx, data=sdata)
        if self._dummy_matrix_buffer is None:
            mdata = np.eye(4, dtype=np.float32)
            self._dummy_matrix_buffer = m.create_buffer(context=ctx, data=mdata)

        for index in (11, 12):
            m.set_vertex_buffer(
                context=ctx, buffer=self._dummy_float_buffer, index=index, offset=0
            )
        m.set_vertex_buffer(
            context=ctx, buffer=self._dummy_color_buffer, index=13, offset=0
        )
        m.set_vertex_buffer(
            context=ctx, buffer=self._dummy_shift_scale_buffer, index=14, offset=0
        )
        m.set_vertex_buffer(
            context=ctx, buffer=self._dummy_matrix_buffer, index=15, offset=0
        )

    def _build_uniforms(self, drawing):
        """
        Build the Uniforms struct matching common.glsl layout.
        Returns a contiguous numpy byte array.
        """
        pm = self.current_projection_matrix
        mvm = self.current_model_view_matrix
        if pm is None or mvm is None:
            return None

        def mat4(m):
            if m is None:
                return np.eye(4, dtype=np.float32)
            if hasattr(m, "opengl_matrix"):
                return np.asarray(m.opengl_matrix(), dtype=np.float32)
            return np.asarray(m, dtype=np.float32)

        # Pack struct fields in order matching common.glsl Uniforms
        parts = []
        parts.append(mat4(mvm))  # model_view_matrix
        parts.append(mat4(pm))  # projection_matrix
        parts.append(mat4(self.current_model_matrix))  # model_matrix
        parts.append(mat4(self.current_view_matrix))  # view_matrix
        # Shadow transform (scene-to-shadow-texture coords)
        if (
            isinstance(self.shadow, VulkanShadow)
            and self.shadow._shadow_view_transform is not None
        ):
            parts.append(mat4(self.shadow._shadow_view_transform))
        else:
            parts.append(np.eye(4, dtype=np.float32))
        parts.append(np.eye(4, dtype=np.float32))  # ambient_tex3d_transform

        # color (float4) -- use drawing's single color if available
        colors = drawing.colors
        if colors is not None and len(colors) == 1:
            c = [v / 255.0 for v in colors[0]]
        else:
            c = self.model_color
        parts.append(np.array(c, dtype=np.float32))

        # depth_cue_color (float3 + pad)
        parts.append(np.array([0, 0, 0, 0], dtype=np.float32))

        # depth_cue_range (float2) + near/far clip (float2)
        near, far = self._near_far_clip
        parts.append(np.array([0, 1000, near, far], dtype=np.float32))

        # clip_planes: 8 x float4
        parts.append(np.zeros((8, 4), dtype=np.float32))

        # num_clip_planes (int), impostor_dashes (int), shadow_depth (float), shadow_count (int)
        tail = np.zeros(4, dtype=np.float32)
        tail_int = tail.view(np.int32)
        dashes = getattr(drawing, "impostor_dashes", 0)
        tail_int[1] = int(dashes)
        if isinstance(self.multishadow, VulkanMultishadow):
            tail[2] = np.float32(self.multishadow._multishadow_depth or 0)
            tail_int[3] = int(
                min(
                    self.multishadow.max_multishadows(),
                    len(self.multishadow._view_matrices),
                )
            )
        parts.append(tail)

        # Concatenate all parts as raw bytes
        data = np.concatenate([p.reshape(-1) for p in parts]).astype(np.float32)
        return np.ascontiguousarray(data)

    def _ensure_shader_pipeline(self, shader, shader_name, vertex_func, fragment_func,
                                topology=0):
        """Compile a shader pipeline from the shaders/ directory.

        topology: 0 = triangle list, 1 = line list, 2 = point list. Vulkan
        bakes topology into the pipeline, so each display style needs its
        own pipeline variant.
        """
        caps = shader.capabilities
        # Build function constants from capabilities
        from .renderer import (
            SHADER_LIGHTING,
            SHADER_VERTEX_COLORS,
            SHADER_SHADOW,
            SHADER_MULTISHADOW,
            SHADER_DEPTH_CUE,
            SHADER_CLIP_PLANES,
            SHADER_TEXTURE_2D,
            SHADER_TEXTURE_3D,
            SHADER_COLORMAP,
            SHADER_TEXTURE_3D_AMBIENT,
            SHADER_TRANSPARENT_ONLY,
            SHADER_OPAQUE_ONLY,
            SHADER_LIGHTING_NORMALS,
            SHADER_INSTANCING,
            SHADER_SHIFT_AND_SCALE,
            SHADER_TEXTURE_CUBEMAP,
            SHADER_ALL_WHITE,
        )

        constants = [0] * 20
        constants[0] = 1 if (caps & SHADER_LIGHTING) else 0
        constants[1] = 1 if (caps & SHADER_VERTEX_COLORS) else 0
        constants[2] = 1 if (caps & SHADER_SHADOW) else 0
        constants[3] = 1 if (caps & SHADER_MULTISHADOW) else 0
        constants[4] = 1 if (caps & SHADER_DEPTH_CUE) else 0
        constants[5] = 1 if (caps & SHADER_CLIP_PLANES) else 0
        constants[6] = 1 if (caps & SHADER_TEXTURE_2D) else 0
        constants[7] = 1 if (caps & SHADER_TEXTURE_3D) else 0
        constants[8] = 1 if (caps & SHADER_TEXTURE_CUBEMAP) else 0
        constants[9] = 1 if (caps & SHADER_LIGHTING_NORMALS) else 0
        constants[10] = 1 if (caps & SHADER_INSTANCING) else 0
        constants[11] = 1 if (caps & SHADER_SHIFT_AND_SCALE) else 0
        constants[12] = 1 if (caps & SHADER_COLORMAP) else 0
        constants[13] = 1 if (caps & SHADER_ALL_WHITE) else 0
        constants[14] = 1 if (caps & SHADER_TRANSPARENT_ONLY) else 0
        constants[15] = 1 if (caps & SHADER_OPAQUE_ONLY) else 0
        constants[16] = 1 if (caps & SHADER_TEXTURE_3D_AMBIENT) else 0
        constants[17] = 1 if self._depth_prepass_rendering else 0
        constants[18] = 1 if self._sphere_ao_active else 0
        constants[19] = 1 if self._oit_rendering else 0

        # Cache key includes shader name, constants, depth-only/prepass
        # flags, and the cull/blend/depth-write state baked into the pipeline.
        const_key = tuple(constants)
        depth_only = self._depth_only_rendering
        depth_prepass = self._depth_prepass_rendering
        face_culling = self._face_culling_enabled
        blend_op_max = self._blend_op_max
        blend_premult = self._blend_premult
        oit_mode = self._oit_rendering
        depth_write = False if oit_mode else self._depth_write_enabled
        cache_key = (shader_name, const_key, depth_only, depth_prepass,
                     face_culling, blend_op_max, blend_premult, depth_write,
                     topology, oit_mode)
        if cache_key in self._pipeline_cache:
            shader._pipeline = self._pipeline_cache[cache_key]
            shader._pipeline_depth_only = depth_only
            shader._pipeline_depth_prepass = depth_prepass
            shader._pipeline_face_culling = face_culling
            shader._pipeline_blend_op_max = blend_op_max
            shader._pipeline_blend_premult = blend_premult
            shader._pipeline_depth_write = depth_write
            shader._pipeline_topology = topology
            return
        try:
            import os, sys

            shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
            common_path = os.path.join(shader_dir, "common.glsl")
            vert_path = os.path.join(shader_dir, f"{shader_name}.vert")
            frag_path = os.path.join(shader_dir, f"{shader_name}.frag")

            with open(common_path, encoding='utf-8') as f:
                common_src = f.read()
            with open(vert_path, encoding='utf-8') as f:
                vert_src = f.read()
            with open(frag_path, encoding='utf-8') as f:
                frag_src = f.read()

            # Inline common.glsl into both shader stages
            vert_src = vert_src.replace('#include "common.glsl"', common_src)
            frag_src = frag_src.replace('#include "common.glsl"', common_src)

            # Mesh shader uses per-vertex colors (e.g. ribbons); the
            # impostor shaders use per-instance colors (one per atom/bond).
            instance_colors = 0 if shader_name == "mesh" else 1

            create_pipeline_kw = dict(
                context=self._ctx,
                vert_source=vert_src,
                frag_source=frag_src,
                has_depth=1,
                constants=constants,
                depth_only=1 if depth_only else 0,
                instance_colors=instance_colors,
                face_culling=1 if face_culling else 0,
                blend_op_max=1 if blend_op_max else 0,
                blend_premult=1 if blend_premult else 0,
                depth_write=1 if depth_write else 0,
                topology=int(topology),
                oit_mode=1 if oit_mode else 0,
            )
            if depth_prepass:
                try:
                    handle = self._vulkan.create_pipeline(
                        **create_pipeline_kw, color_write=0
                    )
                except TypeError as e:
                    if "keyword" not in str(e):
                        raise
                    handle = self._vulkan.create_pipeline(**create_pipeline_kw)
            else:
                handle = self._vulkan.create_pipeline(**create_pipeline_kw)
            self._pipeline_cache[cache_key] = handle
            shader._pipeline = handle
            shader._pipeline_depth_only = depth_only
            shader._pipeline_depth_prepass = depth_prepass
            shader._pipeline_sphere_ao = self._sphere_ao_active
            shader._pipeline_face_culling = face_culling
            shader._pipeline_blend_op_max = blend_op_max
            shader._pipeline_blend_premult = blend_premult
            shader._pipeline_depth_write = depth_write
            shader._pipeline_topology = topology
        except Exception as e:
            import sys

            print(
                f"Vulkan {shader_name} pipeline error: {e}", file=sys.stderr, flush=True
            )
            self._pipeline_cache[cache_key] = None
            shader._pipeline = None
            shader._pipeline_depth_only = depth_only
            shader._pipeline_depth_prepass = depth_prepass
            shader._pipeline_sphere_ao = self._sphere_ao_active
            shader._pipeline_face_culling = face_culling
            shader._pipeline_blend_op_max = blend_op_max
            shader._pipeline_blend_premult = blend_premult
            shader._pipeline_depth_write = depth_write
            shader._pipeline_topology = topology

    # -------------------------------------------------------------------
    # Stereo
    # -------------------------------------------------------------------

    def set_stereo_buffer(self, eye_num):
        pass

    def set_stereo_360_params(self, origin, vertical, shift):
        pass

    # -------------------------------------------------------------------
    # Post-processing (fullscreen quad)
    # -------------------------------------------------------------------

    _quad_vbuf = None
    _quad_tcbuf = None
    _quad_ibuf = None
    _postprocess_pipelines = {}

    def _ensure_quad_buffers(self):
        """Create vertex/texcoord/index buffers for a fullscreen quad."""
        if self._quad_vbuf is not None:
            return
        m = self._vulkan
        ctx = self._ctx
        verts = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32
        )
        # Vulkan NDC has Y pointing DOWN, and texture V=0 is at the top.
        # Map vertex (-1,-1) (top-left of screen) -> (0,0) (top-left of
        # texture), so a postprocess sampler reads the texel that the
        # compute kernel wrote at the same window pixel. Padding to vec3
        # to match the binding-11 stride declared in vulkan_pipeline.cpp.
        tcoords = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._quad_vbuf = m.create_buffer(context=ctx, data=verts)
        self._quad_tcbuf = m.create_buffer(context=ctx, data=tcoords)
        self._quad_ibuf = m.create_buffer(context=ctx, data=indices)

    def _get_postprocess_pipeline(
        self, shader_file, vertex_func, fragment_func, has_depth=False
    ):
        """Get or compile a post-processing pipeline."""
        key = (shader_file, vertex_func, fragment_func, has_depth)
        if key in self._postprocess_pipelines:
            return self._postprocess_pipelines[key]

        import os

        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

        # Post-processing shaders use separate .vert/.frag files
        vert_path = os.path.join(shader_dir, f"{shader_file}.vert")
        frag_path = os.path.join(shader_dir, f"{shader_file}.frag")
        with open(vert_path, encoding='utf-8') as f:
            vert_source = f.read()
        with open(frag_path, encoding='utf-8') as f:
            frag_source = f.read()

        try:
            # Fullscreen quads are wound clockwise in Vulkan NDC (Y is
            # down) so back-face culling with the default CCW front-face
            # would cull every triangle. Disable culling for postprocess
            # passes — they always cover the full viewport.
            handle = self._vulkan.create_pipeline(
                context=self._ctx,
                vert_source=vert_source,
                frag_source=frag_source,
                has_depth=1 if has_depth else 0,
                constants=[0] * 20,
                depth_only=0,
                face_culling=0,
            )
        except Exception as e:
            import sys

            # Loud failure: previously this silently returned None and
            # the caller silently skipped the draw, hiding shader bugs.
            # Echo to stderr AND vulkan_trace.log so post-mortems don't
            # have to guess why a post-process pass produced no output.
            msg = (f"Vulkan postprocess pipeline FAILED for "
                   f"'{shader_file}' ({vertex_func}/{fragment_func}): {e}")
            print(msg, file=sys.stderr)
            try:
                with open("vulkan_trace.log", "a") as tf:
                    tf.write(msg + "\n")
            except Exception:
                pass
            handle = None
        self._postprocess_pipelines[key] = handle
        return handle

    def draw_fullscreen_quad(
        self,
        texture,
        shader_file,
        vertex_func,
        fragment_func,
        uniforms_data=None,
        has_depth=False,
    ):
        """Render a fullscreen quad with a post-processing shader."""
        m = self._vulkan
        ctx = self._ctx

        self._ensure_quad_buffers()
        pipeline = self._get_postprocess_pipeline(
            shader_file, vertex_func, fragment_func, has_depth=has_depth
        )
        if pipeline is None:
            return

        m.set_pipeline(context=ctx, pipeline=pipeline)
        m.set_depth_stencil_state(
            context=ctx, depth_write=0, depth_test=0, compare_func=7
        )  # always

        m.set_vertex_buffer(context=ctx, buffer=self._quad_vbuf, index=10, offset=0)
        m.set_vertex_buffer(context=ctx, buffer=self._quad_tcbuf, index=11, offset=0)
        m.set_fragment_texture(context=ctx, texture=texture, index=0)

        if uniforms_data is not None:
            ubuf = m.create_buffer(context=ctx, data=uniforms_data)
            m.set_descriptor_buffer(context=ctx, buffer=ubuf, index=0, offset=0)

        m.draw_indexed(
            context=ctx,
            index_buffer=self._quad_ibuf,
            primitive_type=0,
            index_count=6,
            index_offset=0,
            instance_count=1,
        )

        self._set_current_depth_state()

    def draw_fullscreen_quad_two_textures(
        self, texture0, texture1, shader_file, vertex_func, fragment_func,
        uniforms_data=None, has_depth=False,
    ):
        """Fullscreen quad with two input textures (indices 0 and 1)."""
        m = self._vulkan
        ctx = self._ctx

        self._ensure_quad_buffers()
        pipeline = self._get_postprocess_pipeline(
            shader_file, vertex_func, fragment_func, has_depth=has_depth
        )
        if pipeline is None:
            return

        m.set_pipeline(context=ctx, pipeline=pipeline)
        m.set_depth_stencil_state(
            context=ctx, depth_write=0, depth_test=0, compare_func=7
        )

        m.set_vertex_buffer(context=ctx, buffer=self._quad_vbuf, index=10, offset=0)
        m.set_vertex_buffer(context=ctx, buffer=self._quad_tcbuf, index=11, offset=0)
        m.set_fragment_texture(context=ctx, texture=texture0, index=0)
        m.set_fragment_texture(context=ctx, texture=texture1, index=1)

        if uniforms_data is not None:
            ubuf = m.create_buffer(context=ctx, data=uniforms_data)
            m.set_descriptor_buffer(context=ctx, buffer=ubuf, index=0, offset=0)

        m.draw_indexed(
            context=ctx,
            index_buffer=self._quad_ibuf,
            primitive_type=0,
            index_count=6,
            index_offset=0,
            instance_count=1,
        )

        self._set_current_depth_state()

    def get_scene_depth_texture(self):
        """Return a handle to the scene depth buffer for sampling."""
        return self._vulkan.get_depth_texture(context=self._ctx)

    def create_color_texture(self, width, height):
        """Create an RGBA8 texture usable as a color attachment AND a
        compute-shader storage image.

        RGBA8 (not R8) because R8_UNORM isn't a core-required STORAGE_IMAGE
        format; RTAO writes into this texture from a compute kernel.
        """
        return self._vulkan.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=0,
            linear_filter=1,
            depth_compare=0,
            storage=1,
        )

    def create_r32f_texture(self, width, height):
        """Create an R32Float storage texture (RTAO ping-pong accumulation)."""
        return self._vulkan.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=4,
            linear_filter=1,
            depth_compare=0,
            storage=1,
        )

    def destroy_texture(self, texture):
        """Backend-agnostic wrapper used by RenderPass cleanup."""
        if texture is not None:
            self._vulkan.destroy_texture(texture=texture)

    def create_data_buffer(self, data):
        """Backend-agnostic wrapper used by RenderPass for color/data SSBOs."""
        return self._vulkan.create_buffer(context=self._ctx, data=data)

    def create_mapped_buffer(self, size_bytes):
        """Allocate a GPU buffer with a persistently mapped pointer.
        Returns (buffer_handle, mapped_ptr_as_int)."""
        return self._vulkan.create_mapped_buffer(
            context=self._ctx, size=size_bytes)

    def destroy_data_buffer(self, buf):
        if buf is not None:
            self._vulkan.destroy_buffer(buffer=buf)

    # The blur pass currently has no Vulkan port; RTAOPass falls back to
    # temporal accumulation alone when this is False.
    supports_rtao_blur = False

    def draw_rtao_composite(self, ao_tex):
        """Multiply ao_tex.r into the bound color attachment via SRC_ALPHA
        blending (output = vec4(0,0,0, 1-ao)). Backend-specific so the
        per-API shader-name conventions stay encapsulated here."""
        self.draw_fullscreen_quad(
            ao_tex,
            "rtao_composite",   # rtao_composite.vert / rtao_composite.frag
            "main",             # GLSL entry is always main; arg ignored
            "main",
        )

    def begin_pt_blit_pass(self):
        """Restart the main render pass with a clear color (so the path-traced
        image fully replaces the rasterised scene) but preserve depth."""
        self._vulkan.end_render_pass(context=self._ctx)
        r, g, b, a = self._background_color
        self._vulkan.begin_render_pass(
            context=self._ctx,
            clear_r=r, clear_g=g, clear_b=b, clear_a=a,
            clear_depth=0, load_color=0, load_depth=1,
        )

    def draw_pt_blit(self, color_tex):
        """Fullscreen blit of the path-tracer color texture."""
        self.draw_fullscreen_quad(
            color_tex,
            "pt_blit",          # pt_blit.vert / pt_blit.frag
            "main", "main",
        )

    def draw_pt_denoise(self, color_tex, depth_tex, uniforms_data):
        """Cross-bilateral denoise of the path-tracer output. Reads
        ``color_tex`` (path-traced rgb) and ``depth_tex`` (rasterised
        impostor depth as the edge-stop), writes the blurred result to
        the currently-bound color attachment."""
        self.draw_fullscreen_quad_two_textures(
            color_tex, depth_tex,
            "pt_denoise",
            "main", "main",
            uniforms_data=uniforms_data,
        )

    def begin_color_pass(self, texture, width, height, clear=True):
        """Begin a render pass targeting a color texture (no depth)."""
        self.begin_frame()
        self._vulkan.end_render_pass(context=self._ctx)
        self._vulkan.begin_color_pass(
            context=self._ctx,
            color_texture=texture,
            width=width,
            height=height,
            do_clear=1 if clear else 0,
        )

    def restart_main_pass(self):
        """Restart the main render pass preserving existing color and depth."""
        self._vulkan.end_render_pass(context=self._ctx)
        r, g, b, a = self._background_color
        self._vulkan.begin_render_pass(
            context=self._ctx,
            clear_r=r,
            clear_g=g,
            clear_b=b,
            clear_a=a,
            clear_depth=0,
            load_color=1,
            load_depth=1,
        )

    # -------------------------------------------------------------------
    # Ray-traced ambient occlusion
    # -------------------------------------------------------------------

    _rtao_accel_struct = None
    _rtao_sphere_buffer = None
    _rtao_shader_source = None
    _rtao_pipeline_compiled = False

    def build_sphere_acceleration_structure(self, sphere_data):
        """Build a ray tracing acceleration structure from sphere positions/radii.

        Parameters
        ----------
        sphere_data : ndarray, shape (N, 4), dtype float32
            Per-sphere (x, y, z, radius) in world space.
        """
        if self._rtao_accel_struct is not None:
            self._rt_backend.destroy_accel_struct(
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
            )
            self._rtao_accel_struct = None
            self._rtao_sphere_buffer = None
            self._rtao_cylinder_buffer = None
            self._rtao_triangle_buffer = None

        data = np.ascontiguousarray(sphere_data, dtype=np.float32)
        as_ptr, sb_ptr = self._rt_backend.build_sphere_accel(
            context=self._rt_ctx, sphere_data=data, sphere_count=len(sphere_data)
        )
        self._rtao_accel_struct = as_ptr
        self._rtao_sphere_buffer = sb_ptr

    _rtao_cylinder_buffer = None

    def build_mixed_acceleration_structure(self, sphere_data, cylinder_data):
        """Build an AS with spheres (geometry 0) and cylinders (geometry 1).

        Parameters
        ----------
        sphere_data : ndarray, shape (N, 4), dtype float32
            Per-sphere (x, y, z, radius) in world space.
        cylinder_data : ndarray, shape (M, 8), dtype float32
            Per-cylinder (p0.x, p0.y, p0.z, radius, p1.x, p1.y, p1.z, 0).
        """
        if self._rtao_accel_struct is not None:
            self._rt_backend.destroy_accel_struct(
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
            )
            self._rtao_accel_struct = None
            self._rtao_sphere_buffer = None
            self._rtao_cylinder_buffer = None
            self._rtao_triangle_buffer = None

        sph = np.ascontiguousarray(sphere_data, dtype=np.float32)
        cyl = np.ascontiguousarray(cylinder_data, dtype=np.float32)
        as_ptr, sb_ptr, cb_ptr = self._rt_backend.build_mixed_accel(
            context=self._rt_ctx,
            sphere_data=sph, sphere_count=len(sphere_data),
            cylinder_data=cyl, cylinder_count=len(cylinder_data),
        )
        self._rtao_accel_struct = as_ptr
        self._rtao_sphere_buffer = sb_ptr
        self._rtao_cylinder_buffer = cb_ptr

    _rtao_triangle_buffer = None

    def build_scene_acceleration_structure(
        self, sphere_data, cylinder_data=None, triangle_data=None
    ):
        """Build an AS with spheres (0), cylinders (1), and triangles (2).

        Parameters
        ----------
        sphere_data : ndarray, shape (N, 4), dtype float32
        cylinder_data : ndarray or None, shape (M, 8), dtype float32
        triangle_data : ndarray or None, shape (T, 12), dtype float32
            Per-triangle: 3 vertices packed as (v0.xyz, 0, v1.xyz, 0, v2.xyz, 0).
        """
        if self._rtao_accel_struct is not None:
            self._rt_backend.destroy_accel_struct(
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
            )
            self._rtao_accel_struct = None
            self._rtao_sphere_buffer = None
            self._rtao_cylinder_buffer = None
            self._rtao_triangle_buffer = None

        sph = np.ascontiguousarray(sphere_data, dtype=np.float32)
        # Provide empty arrays for missing geometries so the C API always
        # receives valid buffer objects.
        if cylinder_data is not None:
            cyl = np.ascontiguousarray(cylinder_data, dtype=np.float32)
            cyl_count = len(cylinder_data)
        else:
            cyl = np.zeros((0, 8), dtype=np.float32)
            cyl_count = 0
        if triangle_data is not None:
            tri = np.ascontiguousarray(triangle_data, dtype=np.float32)
            tri_count = len(triangle_data)
        else:
            tri = np.zeros((0, 12), dtype=np.float32)
            tri_count = 0

        result = self._rt_backend.build_scene_accel(
            context=self._rt_ctx,
            sphere_data=sph, sphere_count=len(sphere_data),
            cylinder_data=cyl, cylinder_count=cyl_count,
            triangle_data=tri, triangle_count=tri_count,
        )
        as_ptr, sb_ptr, cb_ptr, tb_ptr = result
        self._rtao_accel_struct = as_ptr
        self._rtao_sphere_buffer = sb_ptr
        self._rtao_cylinder_buffer = cb_ptr
        self._rtao_triangle_buffer = tb_ptr

    def _ensure_rtao_pipeline(self):
        """Compile the RTAO compute pipeline (once)."""
        if self._rtao_pipeline_compiled:
            return
        import os

        # Metal RT uses .metal shaders; Vulkan RT uses .comp (GLSL).
        if self._rt_backend is self._metal_rt_module:
            shader_path = os.path.join(os.path.dirname(__file__), "shaders", "rtao.metal")
        else:
            shader_path = os.path.join(os.path.dirname(__file__), "shaders", "rtao.comp")
        with open(shader_path, encoding='utf-8') as f:
            source = f.read()
        self._rt_backend.compile_rtao(context=self._rt_ctx, shader_source=source)
        self._rtao_pipeline_compiled = True

    def dispatch_rtao(
        self,
        ao_texture,
        prev_texture,
        accum_texture,
        inv_projection,
        inv_view,
        ao_radius=5.0,
        intensity=1.5,
        num_rays=8,
        frame=0,
        accum_count=1,
    ):
        """Dispatch the RTAO compute kernel."""
        if self._rtao_accel_struct is None:
            return
        self._ensure_rtao_pipeline()

        w, h = self._width, self._height
        uniforms = np.zeros(40, dtype=np.float32)
        uniforms[0:16] = np.asarray(inv_projection, dtype=np.float32).reshape(-1)
        uniforms[16:32] = np.asarray(inv_view, dtype=np.float32).reshape(-1)
        uniforms_int = uniforms.view(np.uint32)
        uniforms_int[32] = w
        uniforms_int[33] = h
        uniforms[34] = np.float32(ao_radius)
        uniforms[35] = np.float32(intensity)
        uniforms_int[36] = num_rays
        uniforms_int[37] = frame
        uniforms_int[38] = accum_count
        # uniforms[39] is padding for 16-byte alignment

        if self._rt_backend is self._metal_rt_module:
            # The Vulkan projection matrix negates Y for clip space, but
            # the Metal RT shader reads the MoltenVK depth buffer which
            # already accounts for this. Undo the Y-flip in the inverse
            # projection so position reconstruction comes out right.
            uniforms[4] = -uniforms[4]   # inv_proj[1][0]
            uniforms[5] = -uniforms[5]   # inv_proj[1][1]
            uniforms[6] = -uniforms[6]   # inv_proj[1][2]
            uniforms[7] = -uniforms[7]   # inv_proj[1][3]

        data = np.ascontiguousarray(uniforms)

        self._vulkan.end_render_pass(context=self._ctx)

        if self._rt_backend is self._metal_rt_module:
            # Submit pending Vulkan work and signal the timeline semaphore
            # so the Metal RT dispatch can wait on it GPU-side.
            self._vulkan.flush_gpu(context=self._ctx)
            depth_tex = self._vulkan.get_depth_texture(context=self._ctx)
            self._rt_backend.dispatch_rtao(
                context=self._rt_ctx,
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
                depth_texture=self._export_vk_texture(depth_tex),
                ao_texture=self._export_vk_texture(ao_texture),
                prev_texture=self._export_vk_texture(prev_texture),
                accum_texture=self._export_vk_texture(accum_texture),
                uniforms=data,
            )
        else:
            self._rt_backend.dispatch_rtao(
                context=self._rt_ctx,
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
                ao_texture=ao_texture,
                prev_texture=prev_texture,
                accum_texture=accum_texture,
                uniforms=data,
            )

    # -------------------------------------------------------------------
    # Path tracer
    # -------------------------------------------------------------------

    _pt_pipeline_compiled = False

    def _ensure_pt_pipeline(self):
        """Compile the path tracer compute pipeline (once)."""
        if self._pt_pipeline_compiled:
            return
        import os

        if self._rt_backend is self._metal_rt_module:
            shader_path = os.path.join(
                os.path.dirname(__file__), "shaders", "pathtracer.metal"
            )
        else:
            shader_path = os.path.join(
                os.path.dirname(__file__), "shaders", "pathtracer.comp"
            )
        with open(shader_path, encoding='utf-8') as f:
            source = f.read()
        self._rt_backend.compile_pathtracer(context=self._rt_ctx, shader_source=source)
        self._pt_pipeline_compiled = True

    def create_rgba_texture(self, width, height):
        """Create an RGBA8 storage texture for path-tracer output.

        storage=1 is required because the path-tracer compute kernel binds
        these as `image2D` and writes directly via imageStore. Without the
        STORAGE_IMAGE usage bit, the writes hit undefined memory and
        produce garbage (a few rows of corruption + black elsewhere).
        """
        return self._vulkan.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=0,
            linear_filter=1,
            depth_compare=0,
            storage=1,
        )

    def dispatch_pathtracer(
        self,
        color_texture,
        prev_texture,
        accum_texture,
        color_buffer,
        cylinder_color_buffer,
        triangle_color_buffer,
        inv_projection,
        inv_view,
        frame=0,
        accum_count=1,
        key_light_dir=(0, 0, 1),
        key_light_intensity=1.0,
        fill_light_dir=(0, 0, 1),
        fill_light_intensity=0.5,
        ambient_color=(0.1, 0.1, 0.1),
        background_color=(0, 0, 0),
        max_bounces=2,
    ):
        """Dispatch the path tracer compute kernel."""
        if self._rtao_accel_struct is None:
            return
        self._ensure_pt_pipeline()

        w, h = self._width, self._height
        uniforms = np.zeros(56, dtype=np.float32)
        uniforms[0:16] = np.asarray(inv_projection, dtype=np.float32).reshape(-1)
        uniforms[16:32] = np.asarray(inv_view, dtype=np.float32).reshape(-1)
        uniforms_int = uniforms.view(np.uint32)
        uniforms_int[32] = w
        uniforms_int[33] = h
        uniforms_int[34] = frame
        uniforms_int[35] = accum_count
        uniforms[36:39] = np.asarray(key_light_dir, dtype=np.float32).ravel()[:3]
        uniforms[39] = np.float32(key_light_intensity)
        uniforms[40:43] = np.asarray(fill_light_dir, dtype=np.float32).ravel()[:3]
        uniforms[43] = np.float32(fill_light_intensity)
        uniforms[44:47] = np.asarray(ambient_color, dtype=np.float32).ravel()[:3]
        uniforms[48:51] = np.asarray(background_color, dtype=np.float32).ravel()[:3]
        uniforms_int[52] = max_bounces

        if self._rt_backend is self._metal_rt_module:
            # Undo the Vulkan Y-flip in the inverse projection (same
            # reason as in dispatch_rtao).
            uniforms[4] = -uniforms[4]
            uniforms[5] = -uniforms[5]
            uniforms[6] = -uniforms[6]
            uniforms[7] = -uniforms[7]

        data = np.ascontiguousarray(uniforms)

        cyl_buf = self._rtao_cylinder_buffer or 0
        cyl_color_buf = cylinder_color_buffer or 0
        tri_buf = self._rtao_triangle_buffer or 0
        tri_color_buf = triangle_color_buffer or 0

        self._vulkan.end_render_pass(context=self._ctx)

        if self._rt_backend is self._metal_rt_module:
            # Submit pending Vulkan work and signal the timeline semaphore.
            self._vulkan.flush_gpu(context=self._ctx)
            # color_buffer and cyl_color_buf are Vulkan SSBOs that need
            # exporting. cylinder_buffer (cyl_buf) is already a native
            # Metal buffer from build_scene_accel — pass it directly.
            # Same for triangle_buffer (tri_buf).
            mtl_color_buf, color_offset = self._export_vk_buffer(color_buffer)
            mtl_cyl_color_buf, cyl_color_offset = self._export_vk_buffer(cyl_color_buf)
            mtl_tri_color_buf, tri_color_offset = self._export_vk_buffer(tri_color_buf)
            self._rt_backend.dispatch_pathtracer(
                context=self._rt_ctx,
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
                color_texture=self._export_vk_texture(color_texture),
                prev_texture=self._export_vk_texture(prev_texture),
                accum_texture=self._export_vk_texture(accum_texture),
                color_buffer=mtl_color_buf,
                color_buffer_offset=color_offset,
                cylinder_buffer=cyl_buf,
                cylinder_color_buffer=mtl_cyl_color_buf,
                cylinder_color_buffer_offset=cyl_color_offset,
                triangle_buffer=tri_buf,
                triangle_color_buffer=mtl_tri_color_buf,
                triangle_color_buffer_offset=tri_color_offset,
                uniforms=data,
            )
        else:
            self._rt_backend.dispatch_pathtracer(
                context=self._rt_ctx,
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
                color_texture=color_texture,
                prev_texture=prev_texture,
                accum_texture=accum_texture,
                color_buffer=color_buffer,
                cylinder_buffer=cyl_buf,
                cylinder_color_buffer=cyl_color_buf,
                triangle_buffer=tri_buf,
                triangle_color_buffer=tri_color_buf,
                uniforms=data,
            )

    # -------------------------------------------------------------------
    # Metal RT interop (macOS / MoltenVK only)
    # -------------------------------------------------------------------

    _metal_rt_module = None
    _metal_rt_ctx = None
    _metal_rt_checked = False

    def _init_metal_rt(self):
        """Lazily import and initialize _metal_rt on macOS."""
        if self._metal_rt_checked:
            return self._metal_rt_module is not None
        self._metal_rt_checked = True
        if sys.platform != "darwin":
            return False
        try:
            from chimerax.graphics import _metal_rt
            ctx_ptr = _metal_rt.init_from_vulkan_context(context=self._ctx)
            self._metal_rt_module = _metal_rt
            self._metal_rt_ctx = ctx_ptr
            return True
        except (ImportError, RuntimeError):
            return False

    @property
    def _rt_backend(self):
        """The C extension module for ray tracing calls.

        Returns _vulkan on platforms with native Vulkan RT, or _metal_rt
        on macOS where MoltenVK lacks RT extensions.
        """
        try:
            if self._vulkan.supports_ray_tracing(context=self._ctx):
                return self._vulkan
        except Exception:
            pass
        if self._init_metal_rt():
            return self._metal_rt_module
        return None

    @property
    def _rt_ctx(self):
        """The context pointer to pass to the RT backend."""
        if self._rt_backend is self._vulkan:
            return self._ctx
        return self._metal_rt_ctx

    def supports_ray_tracing(self):
        """True when either native Vulkan RT or Metal RT (via MoltenVK
        interop) is available."""
        return self._rt_backend is not None

    # Caches of Vulkan -> Metal handle exports.
    _exported_textures = None
    _exported_buffers = None

    def _export_vk_buffer(self, vk_buf_handle):
        """Export a VulkanBuffer handle to an (MTLBuffer, offset) pair.

        Two-step: ask _vulkan for the VkDeviceMemory + offset (VMA may
        sub-allocate), then ask _metal_rt to export as MTLBuffer.
        Returns (mtl_buffer_ptr, byte_offset).

        Cached by buffer handle — invalidated when the handle changes
        (the render pass destroys and recreates buffers on geometry change).
        """
        if vk_buf_handle is None or vk_buf_handle == 0:
            return 0, 0
        if self._exported_buffers is None:
            self._exported_buffers = {}
        cached = self._exported_buffers.get(vk_buf_handle)
        if cached is not None:
            return cached
        device_memory, offset = self._vulkan.get_buffer_device_memory(
            buffer=vk_buf_handle
        )
        mtl_buf = self._metal_rt_module.export_buffer(
            context=self._metal_rt_ctx, device_memory=device_memory
        )
        result = (mtl_buf, offset)
        self._exported_buffers[vk_buf_handle] = result
        return result

    def _export_vk_texture(self, vk_tex_handle):
        """Export a VulkanTexture handle to a MetalRTTexture handle.

        Caches the result so repeated calls with the same handle don't
        re-export. The cache is per-renderer-instance.
        """
        if self._exported_textures is None:
            self._exported_textures = {}
        if vk_tex_handle in self._exported_textures:
            return self._exported_textures[vk_tex_handle]
        rt_tex = self._metal_rt_module.export_texture(
            context=self._metal_rt_ctx, texture=vk_tex_handle
        )
        self._exported_textures[vk_tex_handle] = rt_tex
        return rt_tex

    def max_3d_texture_size(self):
        """Largest 3D texture dimension supported by the device."""
        return self._context._max_3d_texture_size


# ---------------------------------------------------------------------------
# Stub sub-systems (minimal implementations to avoid crashes)
# ---------------------------------------------------------------------------


class VulkanShadow:
    """Single directional shadow map state and binding for Vulkan."""

    def __init__(self, render):
        self._render = render
        self._shadow_texture = None
        self._shadow_transform = None
        self._shadow_view_transform = None
        self.multishadow_update_needed = False

    def compute_shadow_transforms(self, light_direction, center, radius, depth_bias):
        """Compute light-view inverse and scene-to-texture transform."""
        from chimerax.geometry import (
            normalize_vector,
            orthonormal_frame,
            translation,
            scale,
        )

        ld = normalize_vector(light_direction)
        lv = translation(center - radius * ld) * orthonormal_frame(-ld)
        lvinv = lv.inverse(is_orthonormal=True)
        # Vulkan stores texture row 0 at the top of the image (V=0 = top),
        # which is the opposite of OpenGL. Flip Y in the scene-to-texture
        # transform so the shadow lookup samples the correct row. The
        # projection-matrix Y flip in set_projection_matrix is a separate
        # NDC-coordinate adjustment and does not cancel this out.
        stf = (
            translation((0.5, 0.5, -depth_bias))
            * scale((0.5 / radius, -0.5 / radius, -0.5 / radius))
            * lvinv
        )
        return lvinv, stf

    def set_scene_transform(self, stf, texture):
        """Store the scene-to-texture transform and texture handle."""
        self._shadow_transform = stf
        self._shadow_texture = texture

    def set_shadow_view(self, camera_position):
        """Update shadow transform for current camera position."""
        if self._shadow_transform is None:
            return
        self._shadow_view_transform = self._shadow_transform * camera_position

    def bind_shadow_texture(self, render):
        """Bind shadow map for fragment shader sampling."""
        if self._shadow_texture is None:
            return
        m = render._vulkan
        ctx = render._ctx
        m.set_fragment_texture(context=ctx, texture=self._shadow_texture, index=3)


class VulkanMultishadow:
    """Tiled multishadow map state and binding for Vulkan."""

    def __init__(self, render):
        self._render = render
        self._multishadow_texture = None
        self._multishadow_depth = None
        self._directions = None
        self._transforms = []
        self._view_matrices = []
        self._matrix_buffer = None
        self.multishadow_update_needed = False

    def max_multishadows(self):
        return 8192

    def multishadow_directions(self, n):
        """Return *n* uniformly distributed sphere points (cached)."""
        if self._directions is None or len(self._directions) != n:
            from chimerax.geometry import sphere

            self._directions = sphere.sphere_points(n)
        return self._directions

    def transforms_valid(self, expected_count):
        """True if cached transforms match the expected direction count."""
        return len(self._transforms) == expected_count

    def reset_transforms(self):
        """Invalidate cached transforms (parameter change)."""
        self._transforms = []
        self._view_matrices = []

    def compute_multishadow_transforms(
        self,
        light_direction,
        center,
        radius,
        depth_bias,
        x,
        y,
        width,
        height,
        texture_size,
    ):
        """Compute light-view inverse and tiled scene-to-texture transform."""
        from chimerax.geometry import (
            normalize_vector,
            orthonormal_frame,
            translation,
            scale,
        )

        ld = normalize_vector(light_direction)
        lv = translation(center - radius * ld) * orthonormal_frame(-ld)
        lvinv = lv.inverse(is_orthonormal=True)
        # Y flip in the scene-to-texture transform: Vulkan textures store
        # V=0 at the top (opposite of OpenGL). See VulkanShadow for the
        # full explanation.
        stf = (
            translation((0.5, 0.5, -depth_bias))
            * scale((0.5 / radius, -0.5 / radius, -0.5 / radius))
            * lvinv
        )
        sx, sy = width / texture_size, height / texture_size
        tx, ty = x / texture_size, y / texture_size
        stf = translation((tx, ty, 0)) * scale((sx, sy, 1)) * stf
        return lvinv, stf

    def set_scene_transforms(self, transforms, depth, texture):
        """Store all transforms after shadow map generation."""
        self._transforms = transforms
        self._view_matrices = []
        self._multishadow_depth = depth
        self._multishadow_texture = texture

    def set_multishadow_view(self, camera_position):
        """Update transforms for current camera position and upload to GPU."""
        if not self._transforms:
            return
        mats = []
        for stf in self._transforms[: self.max_multishadows()]:
            mats.append((stf * camera_position).opengl_matrix())
        data = np.ascontiguousarray(np.asarray(mats, dtype=np.float32))
        m = self._render._vulkan
        ctx = self._render._ctx
        if self._matrix_buffer is None:
            self._matrix_buffer = m.create_buffer(context=ctx, data=data)
        else:
            try:
                m.update_buffer(buffer=self._matrix_buffer, data=data)
            except RuntimeError:
                m.destroy_buffer(buffer=self._matrix_buffer)
                self._matrix_buffer = m.create_buffer(context=ctx, data=data)
        self._view_matrices = mats

    def bind_multishadow_texture(self, render):
        """Bind tiled depth texture for fragment shader sampling."""
        if self._multishadow_texture is None:
            return
        m = render._vulkan
        ctx = render._ctx
        m.set_fragment_texture(context=ctx, texture=self._multishadow_texture, index=4)


class _StubSilhouette:
    enabled = False

    def start_silhouette_drawing(self, r):
        pass

    def finish_silhouette_drawing(self, r):
        pass

    def draw_silhouette(self, r):
        pass


class _StubOutline:
    offscreen_outline_needed = False

    def set_outline_mask(self):
        pass


class _StubOffscreen:
    enabled = False

    def start(self, r):
        pass

    def finish(self, r):
        pass


class _StubBlend:
    pass


class _StubFramebuffer:
    """Minimal framebuffer stub so View methods don't crash."""

    def __init__(self, render):
        self._render = render
        self.width = 800
        self.height = 600
        self.viewport = (0, 0, 800, 600)
        self.alpha = False
        self.color = True
        self.depth = True
        self.color_texture = None
        self.depth_texture = None

    def activate(self):
        return True

    def copy_from_framebuffer(self, fb, color=True, depth=True):
        pass

    def copy_to_framebuffer(self, fb, color=True, depth=True):
        pass

    def attach_depth_texture(self, tex):
        pass
