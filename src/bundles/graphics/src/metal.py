"""
Metal rendering backend for ChimeraX.

Implements the Renderer protocol using Apple's Metal API via the _metal
C extension module. All Metal API calls go through _metal — this module
handles the Python-level state management, pipeline caching, and
translation between the Drawing/View interface and Metal commands.
"""

import sys
import numpy as np

if sys.platform != "darwin":
    raise ImportError("Metal backend is only available on macOS")


class MetalContext:
    """
    Wraps the Metal device, command queue, and CAMetalLayer.

    Created from a Qt QWindow's native view handle (NSView pointer).
    Equivalent to OpenGLContext in the OpenGL backend.
    """

    def __init__(self, graphics_window, screen, use_stereo=False):
        from chimerax.graphics import _metal

        self.window = graphics_window
        self._screen = screen
        self._metal = _metal

        # Get native NSView pointer from Qt
        nsview = int(graphics_window.winId())
        self._ctx = _metal.create_context(nsview_pointer=nsview)
        self._device_name = _metal.device_name(context=self._ctx)

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
        """Metal doesn't need make_current — context is always available."""
        return True

    def done_current(self):
        pass

    def swap_buffers(self, window=None):
        """Present the current frame."""
        self._metal.end_frame(context=self._ctx)

    def pixel_scale(self):
        return self.window.devicePixelRatio()

    def delete(self):
        if self._ctx is not None:
            self._metal.destroy_context(context=self._ctx)
            self._ctx = None


class MetalShader:
    """
    A Metal pipeline state object, equivalent to an OpenGL shader program.

    Created for a specific combination of capability flags. Cached and
    reused for subsequent draws with the same capabilities.
    """

    def __init__(self, capabilities, pipeline_handle):
        self.capabilities = capabilities
        self._pipeline = pipeline_handle
        self._pipeline_depth_only = None
        self._pipeline_depth_prepass = None
        self._uniforms = {}  # name → value, flushed before draw
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


class MetalBuffer:
    """Wraps a Metal GPU buffer."""

    def __init__(self, metal_module, ctx, data=None):
        self._metal = metal_module
        self._ctx = ctx
        self._handle = None
        if data is not None:
            self.update_buffer_data(data)

    def update_buffer_data(self, data):
        arr = np.ascontiguousarray(data)
        if self._handle is None:
            self._handle = self._metal.create_buffer(context=self._ctx, data=arr)
        else:
            try:
                self._metal.update_buffer(buffer=self._handle, data=arr)
            except RuntimeError:
                # Buffer too small — recreate
                self._metal.destroy_buffer(buffer=self._handle)
                self._handle = self._metal.create_buffer(context=self._ctx, data=arr)

    def delete_buffer(self):
        if self._handle is not None:
            self._metal.destroy_buffer(buffer=self._handle)
            self._handle = None


class MetalTexture:
    """Wraps a Metal texture."""

    def __init__(self):
        self.dimension = 2
        self.size = (0, 0)
        self.is_cubemap = False
        self.linear_interpolation = True
        self._handle = None

    def initialize_rgba(self, size):
        self.size = size
        self.dimension = 2

    def initialize_depth(self, size, depth_compare_mode=True):
        self.size = size
        self.dimension = 2

    def bind_texture(self, texture_unit=None):
        pass

    def unbind_texture(self, texture_unit=None):
        pass

    def reload_texture(self, data, now=False):
        pass

    def delete_texture(self):
        self._handle = None


class MetalRender:
    """
    Metal rendering backend. Implements the Renderer protocol.

    This is the Metal equivalent of the Render class in opengl.py.
    Drawing and View call methods on this object to render the scene.
    """

    def __init__(self, metal_context):
        self._context = metal_context
        self._metal = metal_context._metal
        self._ctx = metal_context._ctx
        # Compatibility: Drawing._draw_geometry accesses renderer.opengl_context
        self.opengl_context = metal_context
        self._width = 800
        self._height = 600
        self._in_frame = False
        self._drawing_buffers = {}  # id(drawing) → {name → buffer_handle}
        self._pipeline_cache = {}  # (shader_name, constants_tuple, depth_only) → pipeline_handle
        self._depth_only_rendering = False
        self._depth_prepass_rendering = False
        self._sphere_ao_active = False

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
        self.shadow = MetalShadow(self)
        self.multishadow = MetalMultishadow(self)
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
                setattr(MetalRender, name, getattr(_r, name))

    def delete(self):
        """Clean up renderer resources."""
        if self._in_frame:
            self.end_frame()

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
        return None  # No OpenGL errors in Metal

    def check_opengl_version(self, *args, **kw):
        pass  # Not applicable to Metal

    def set_default_framebuffer_size(self, width, height):
        self._width = int(width)
        self._height = int(height)
        self._context._metal.resize_drawable(
            context=self._ctx, width=self._width, height=self._height
        )

    def opengl_version(self):
        return (0, 0)  # Not OpenGL

    def opengl_version_number(self):
        return 0

    def opengl_vendor(self):
        return "Apple"

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
        m = self._metal
        ctx = self._ctx

        self.begin_frame()

        # End any active encoder cleanly.
        if ctx:
            m.end_render_pass(context=ctx)

        if descriptor.depth_only and descriptor.depth_texture is not None:
            tex = descriptor.depth_texture
            # Metal depth textures are raw handles (long), not Texture objects.
            # Width/height come from the descriptor viewport or texture size.
            if descriptor.viewport:
                _, _, w, h = descriptor.viewport
            else:
                # Infer from the texture — the multishadow/shadow code knows
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
        m = self._metal
        ctx = self._ctx
        m.end_render_pass(context=ctx)
        self._depth_only_rendering = False
        self.full_viewport()

    # -- Framebuffer stack (stub) ---------------------------------------------

    def default_framebuffer(self):
        return None  # TODO: Metal framebuffer

    def current_framebuffer(self):
        return _StubFramebuffer(self)

    def push_framebuffer(self, fb):
        self._framebuffer_stack.append(fb)

    def pop_framebuffer(self):
        if self._framebuffer_stack:
            return self._framebuffer_stack.pop()
        return None

    def rendering_to_screen(self):
        return len(self._framebuffer_stack) == 0

    def set_offscreen_color_bits(self, bits):
        pass

    # -- Depth texture factory (used by render passes) -----------------------

    def create_depth_texture(self, width, height):
        """Create a depth texture for shadow mapping."""
        return self._metal.create_texture(
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
            self._metal.destroy_texture(texture=texture)

    def enable_shader_shadows(self, enable):
        self.enable_shader_capability(self.SHADER_SHADOW, enable)

    def enable_shader_multishadows(self, enable):
        self.enable_shader_capability(self.SHADER_MULTISHADOW, enable)

    def set_lighting_shader_capabilities(self):
        lp = self.lighting
        # Depth cue is not yet fully supported — uniform values for
        # depth_cue_color and depth_cue_range are placeholders, so
        # enabling it would darken/blacken the scene.
        # self.enable_shader_capability(self.SHADER_DEPTH_CUE, lp.depth_cue)
        self.enable_shader_shadows(lp.shadows)
        self.enable_shader_multishadows(lp.multishadow > 0)

    def initialize_opengl(self, width, height):
        """Metal equivalent of OpenGL initialization."""
        s = self._context.pixel_scale()
        w, h = int(s * width), int(s * height)
        self.set_default_framebuffer_size(w, h)
        self.set_viewport(0, 0, w, h)

    def enable_depth_test(self, enable):
        pass

    def enable_backface_culling(self, enable):
        pass

    def enable_blending(self, enable):
        pass

    def blend_alpha(self, alpha_factor=True):
        pass

    def blend_add(self, f):
        pass

    def blend_max(self, enable):
        pass

    def enable_xor(self, enable):
        pass

    def write_depth(self, write):
        pass

    def depth_invert(self, invert):
        pass

    def draw_transparent(self, draw_depth, draw):
        """Simple single-layer transparency: draw depth first, then color."""
        draw_depth()
        draw()

    def draw_xor_rectangle(self, *args, **kw):
        pass

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
        """Switch Metal rendering to a different QWindow.

        Creates a CAMetalLayer on the target window if needed, swaps the
        context's layer/depth-texture, and updates the viewport.  Returns
        the previous window so the caller can restore it.
        """
        prev_window = self._context.window
        if window is prev_window:
            return prev_window

        # End any in-flight frame before switching layers.
        self.end_frame()

        nsview = int(window.winId())
        self._metal.set_layer_for_view(
            context=self._ctx, nsview_pointer=nsview)

        self._context.window = window

        # Update Python-side size from the pixel-scaled layer size.
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
            p = MetalShader(capabilities, None)  # TODO: actual pipeline creation
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
            # Convert OpenGL projection (NDC z [-1,1]) to Metal (NDC z [0,1]).
            # metal_clip_z = 0.5 * gl_clip_z + 0.5 * gl_clip_w
            # In column-major storage: math row 2 = pm[:, 2], row 3 = pm[:, 3]
            gl_pm = np.array(pm, dtype=np.float32)
            metal_pm = gl_pm.copy()
            metal_pm[:, 2] = 0.5 * gl_pm[:, 2] + 0.5 * gl_pm[:, 3]
            self.current_projection_matrix = metal_pm

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
        """Start a new frame — get drawable and command buffer."""
        if not self._in_frame:
            self._metal.begin_frame(context=self._ctx)
            self._in_frame = True

    def end_frame(self):
        """End the frame — present and commit."""
        if self._in_frame:
            try:
                self._metal.end_frame(context=self._ctx)
            except Exception as e:
                import sys

                print(f"Metal end_frame error: {e}", file=sys.stderr)
            self._in_frame = False

    def draw_background(self, depth=True):
        try:
            self.begin_frame()
            r, g, b, a = self._background_color
            self._metal.begin_render_pass(
                context=self._ctx,
                clear_r=r,
                clear_g=g,
                clear_b=b,
                clear_a=a,
                clear_depth=1 if depth else 0,
            )
        except Exception as e:
            import sys

            print(f"Metal draw_background error: {e}", file=sys.stderr)
            self._in_frame = False

    def _draw_test_triangle(self):
        """Render a hardcoded triangle to verify the Metal pipeline."""
        if not hasattr(self, "_test_pipeline"):
            self._create_test_pipeline()
        if self._test_pipeline is None:
            return

        m = self._metal
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
        shader_source = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut test_vertex(
    const device packed_float3* positions [[buffer(0)]],
    uint vid [[vertex_id]])
{
    VertexOut out;
    float3 pos = positions[vid];
    out.position = float4(pos, 1.0);
    // Color based on vertex position
    out.color = float4(pos.x * 0.5 + 0.5, pos.y * 0.5 + 0.5, 0.5, 1.0);
    return out;
}

fragment float4 test_fragment(VertexOut in [[stage_in]])
{
    return in.color;
}
"""
        try:
            m = self._metal
            ctx = self._ctx
            self._test_pipeline = m.create_pipeline(
                context=ctx,
                shader_source=shader_source,
                vertex_function="test_vertex",
                fragment_function="test_fragment",
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

            print(f"Metal test pipeline error: {e}", file=sys.stderr)
            self._test_pipeline = None

    def allow_equal_depth(self, equal):
        # Compare function values match metal_set_depth_stencil_state:
        # 1 = less, 3 = lessEqual.
        self._depth_compare_func = 3 if equal else 1
        self._set_current_depth_state()

    def _set_current_depth_state(self):
        self._metal.set_depth_stencil_state(
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
        return (self._width, self._height)

    def set_viewport(self, x, y, w, h):
        self._metal.set_viewport(context=self._ctx, x=x, y=y, width=w, height=h)

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

        # Pack LightingParams struct matching common.metal layout exactly:
        #   float3 key_light_direction; float _pad0;       (offset 0)
        #   float3 key_light_diffuse_color; float _pad1;   (offset 4)
        #   float3 key_light_specular_color; float _pad2;  (offset 8)
        #   float specular_exponent; float _pad3,4,5;      (offset 12)
        #   float3 fill_light_direction; float _pad6;       (offset 16)
        #   float3 fill_light_diffuse_color; float _pad7;  (offset 20)
        #   float3 ambient_color; float _pad8;             (offset 24)
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
        # Metal retains the old buffer until its command buffer completes.
        self._lighting_buffer = self._metal.create_buffer(
            context=self._ctx, data=data
        )

    def update_lighting_parameters(self):
        self.activate_lighting()

    def set_depth_cue_parameters(self):
        pass

    def set_clip_parameters(self):
        pass

    def set_near_far_clip(self, near_far):
        self._near_far_clip = near_far

    # -------------------------------------------------------------------
    # Textures
    # -------------------------------------------------------------------

    def set_colormap(self, colormap, colormap_range, data_texture):
        pass

    def set_ambient_texture_transform(self, transform):
        pass

    # -------------------------------------------------------------------
    # Draw commands
    # -------------------------------------------------------------------

    def draw_elements(self, mode, count, offset=0, instance_count=1):
        pass  # TODO: encode draw command

    def draw_drawing(self, drawing, capabilities, highlighted_only=False):
        """
        Metal draw path: takes a Drawing object's raw data and renders it.
        Called from Drawing._draw_geometry when the renderer is Metal.
        """
        if not self._in_frame:
            import sys

            return
        try:
            self._draw_drawing_impl(drawing, capabilities, highlighted_only)
        except Exception as e:
            import sys, traceback

            print(f"Metal draw_drawing error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _draw_drawing_impl(self, drawing, capabilities, highlighted_only=False):
        import sys

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
        """Issue Metal draw call for one scene position."""
        import sys

        m = self._metal
        ctx = self._ctx

        vertices = drawing.vertices
        triangles = drawing.triangles

        # Check for sphere-sphere AO data on this drawing.
        from .renderer import SHADER_IMPOSTOR_SPHERE, SHADER_IMPOSTOR_CYLINDER

        self._sphere_ao_active = (
            not self._depth_only_rendering
            and (capabilities & SHADER_IMPOSTOR_SPHERE)
            and hasattr(drawing, "_ao_offset_buffer")
            and drawing._ao_offset_buffer is not None
        )

        # Get or create Metal pipeline for these capabilities
        shader = self.shader(capabilities)
        if (
            shader._pipeline is None
            or shader._pipeline_depth_only != self._depth_only_rendering
            or shader._pipeline_depth_prepass != self._depth_prepass_rendering
            or getattr(shader, "_pipeline_sphere_ao", False) != self._sphere_ao_active
        ):
            if capabilities & SHADER_IMPOSTOR_SPHERE:
                self._ensure_shader_pipeline(
                    shader, "impostor_sphere", "sphere_vertex", "sphere_fragment"
                )
            elif capabilities & SHADER_IMPOSTOR_CYLINDER:
                self._ensure_shader_pipeline(
                    shader, "impostor_cylinder", "cylinder_vertex", "cylinder_fragment"
                )
            else:
                self._ensure_shader_pipeline(
                    shader, "mesh", "mesh_vertex", "mesh_fragment"
                )
        if shader._pipeline is None:
            return  # Can't render without a pipeline

        m = self._metal
        ctx = self._ctx

        m.set_pipeline(context=ctx, pipeline=shader._pipeline)
        self._set_current_depth_state()
        self._bind_dummy_vertex_buffers()

        # Create or update vertex buffer
        draw_id = id(drawing)
        bufs = self._drawing_buffers.get(draw_id)
        if bufs is None:
            bufs = {}
            self._drawing_buffers[draw_id] = bufs

        # Vertex positions (buffer index 0)
        vdata = np.ascontiguousarray(vertices, dtype=np.float32)
        if "vertices" not in bufs or bufs.get("vcount") != len(vertices):
            if "vertices" in bufs:
                m.destroy_buffer(buffer=bufs["vertices"])
            bufs["vertices"] = m.create_buffer(context=ctx, data=vdata)
            bufs["vcount"] = len(vertices)
        else:
            m.update_buffer(buffer=bufs["vertices"], data=vdata)

        # Index buffer
        idata = np.ascontiguousarray(triangles.reshape(-1), dtype=np.uint32)
        if "indices" not in bufs or bufs.get("icount") != len(idata):
            if "indices" in bufs:
                m.destroy_buffer(buffer=bufs["indices"])
            bufs["indices"] = m.create_buffer(context=ctx, data=idata)
            bufs["icount"] = len(idata)
        else:
            m.update_buffer(buffer=bufs["indices"], data=idata)

        # Vertex data uses buffer indices 10+ to avoid conflict with uniform buffers 0-3
        m.set_vertex_buffer(context=ctx, buffer=bufs["vertices"], index=10, offset=0)

        # Normals (buffer index 2)
        normals = drawing.normals
        if normals is not None:
            ndata = np.ascontiguousarray(normals, dtype=np.float32)
            if "normals" not in bufs or bufs.get("ncount") != len(normals):
                if "normals" in bufs:
                    m.destroy_buffer(buffer=bufs["normals"])
                bufs["normals"] = m.create_buffer(context=ctx, data=ndata)
                bufs["ncount"] = len(normals)
            else:
                m.update_buffer(buffer=bufs["normals"], data=ndata)
            m.set_vertex_buffer(context=ctx, buffer=bufs["normals"], index=12, offset=0)

        # Per-instance colors (buffer index 3)
        colors = drawing.colors
        if colors is not None and len(colors) > 0:
            cdata = np.ascontiguousarray(colors, dtype=np.uint8)
            if "colors" not in bufs or bufs.get("ccount") != len(colors):
                if "colors" in bufs:
                    m.destroy_buffer(buffer=bufs["colors"])
                bufs["colors"] = m.create_buffer(context=ctx, data=cdata)
                bufs["ccount"] = len(colors)
            else:
                m.update_buffer(buffer=bufs["colors"], data=cdata)
            m.set_vertex_buffer(context=ctx, buffer=bufs["colors"], index=13, offset=0)

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
            m.set_vertex_buffer(context=ctx, buffer=ubuf, index=0, offset=0)
            m.set_fragment_buffer(context=ctx, buffer=ubuf, index=0, offset=0)

        # Lighting params at buffer 1 (vertex + fragment)
        if hasattr(self, "_lighting_buffer"):
            m.set_vertex_buffer(
                context=ctx, buffer=self._lighting_buffer, index=1, offset=0
            )
            m.set_fragment_buffer(
                context=ctx, buffer=self._lighting_buffer, index=1, offset=0
            )

        # Multishadow transform block at buffer 2.
        if (
            not self._depth_only_rendering
            and isinstance(self.multishadow, MetalMultishadow)
            and self.multishadow._matrix_buffer is not None
        ):
            m.set_fragment_buffer(
                context=ctx, buffer=self.multishadow._matrix_buffer, index=2, offset=0
            )

        # Shadow map texture at index 3 (skip during depth-only pass to avoid
        # read-while-write on the shadow texture that is the render target).
        if not self._depth_only_rendering and isinstance(self.shadow, MetalShadow):
            self.shadow.bind_shadow_texture(self)
        if not self._depth_only_rendering and isinstance(
            self.multishadow, MetalMultishadow
        ):
            self.multishadow.bind_multishadow_texture(self)

        # Instance data
        pos = drawing.positions
        use_instancing = len(pos) > 1 or pos.shift_and_scale_array() is not None
        instance_count = len(pos) if use_instancing else 1

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

        # Instance matrices (buffer index 5)
        if sas is None and len(pos) > 1:
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

        # Sphere-sphere AO buffers at fragment buffer 3 and 4.
        if self._sphere_ao_active:
            m.set_fragment_buffer(
                context=ctx, buffer=drawing._ao_offset_buffer, index=3, offset=0
            )
            m.set_fragment_buffer(
                context=ctx, buffer=drawing._ao_neighbor_buffer, index=4, offset=0
            )

        # Draw
        index_count = triangles.size
        m.draw_indexed(
            context=ctx,
            index_buffer=bufs["indices"],
            primitive_type=0,
            index_count=index_count,
            index_offset=0,
            instance_count=instance_count,
        )

    def _bind_dummy_vertex_buffers(self):
        """
        Bind defaults for optional vertex attributes.  The Metal pipeline
        descriptor includes texture coordinates, normals, colors, sphere
        shift/scale, and cylinder placement matrices for all shader variants;
        drawings only provide the subset their shader reads.
        """
        m = self._metal
        ctx = self._ctx
        if self._dummy_float_buffer is None:
            fdata = np.zeros((16,), dtype=np.float32)
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
        Build the Uniforms struct matching common.metal layout.
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

        # Pack struct fields in order matching common.metal Uniforms
        parts = []
        parts.append(mat4(mvm))  # model_view_matrix
        parts.append(mat4(pm))  # projection_matrix
        parts.append(mat4(self.current_model_matrix))  # model_matrix
        parts.append(mat4(self.current_view_matrix))  # view_matrix
        # Shadow transform (scene-to-shadow-texture coords)
        if (
            isinstance(self.shadow, MetalShadow)
            and self.shadow._shadow_view_transform is not None
        ):
            parts.append(mat4(self.shadow._shadow_view_transform))
        else:
            parts.append(np.eye(4, dtype=np.float32))
        parts.append(np.eye(4, dtype=np.float32))  # ambient_tex3d_transform

        # color (float4) — use drawing's single color if available
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
        if isinstance(self.multishadow, MetalMultishadow):
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

    def _ensure_shader_pipeline(self, shader, shader_name, vertex_func, fragment_func):
        """Compile a shader pipeline from the shaders/ directory."""
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

        # Cache key includes shader name, constants, and depth-only/prepass flags.
        const_key = tuple(constants)
        depth_only = self._depth_only_rendering
        depth_prepass = self._depth_prepass_rendering
        cache_key = (shader_name, const_key, depth_only, depth_prepass)
        if cache_key in self._pipeline_cache:
            shader._pipeline = self._pipeline_cache[cache_key]
            shader._pipeline_depth_only = depth_only
            shader._pipeline_depth_prepass = depth_prepass
            return
        try:
            import os, sys

            shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
            common_path = os.path.join(shader_dir, "common.metal")
            shader_path = os.path.join(shader_dir, f"{shader_name}.metal")

            with open(common_path) as f:
                common_src = f.read()
            with open(shader_path) as f:
                shader_src = f.read()

            # Replace #include with inline source
            full_src = shader_src.replace('#include "common.metal"', common_src)

            create_pipeline_kw = dict(
                context=self._ctx,
                shader_source=full_src,
                vertex_function=vertex_func,
                fragment_function=fragment_func,
                has_depth=1,
                constants=constants,
                depth_only=1 if depth_only else 0,
            )
            if depth_prepass:
                try:
                    handle = self._metal.create_pipeline(
                        **create_pipeline_kw, color_write=0
                    )
                except TypeError as e:
                    if "keyword" not in str(e):
                        raise
                    handle = self._metal.create_pipeline(**create_pipeline_kw)
            else:
                handle = self._metal.create_pipeline(**create_pipeline_kw)
            self._pipeline_cache[cache_key] = handle
            shader._pipeline = handle
            shader._pipeline_depth_only = depth_only
            shader._pipeline_depth_prepass = depth_prepass
            shader._pipeline_sphere_ao = self._sphere_ao_active
        except Exception as e:
            import sys

            print(
                f"Metal {shader_name} pipeline error: {e}", file=sys.stderr, flush=True
            )
            self._pipeline_cache[cache_key] = None
            shader._pipeline = None
            shader._pipeline_depth_only = depth_only
            shader._pipeline_depth_prepass = depth_prepass
            shader._pipeline_sphere_ao = self._sphere_ao_active

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
        m = self._metal
        ctx = self._ctx
        verts = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32
        )
        tcoords = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
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
        shader_path = os.path.join(shader_dir, shader_file)
        with open(shader_path) as f:
            source = f.read()

        try:
            handle = self._metal.create_pipeline(
                context=self._ctx,
                shader_source=source,
                vertex_function=vertex_func,
                fragment_function=fragment_func,
                has_depth=1 if has_depth else 0,
                constants=[0] * 20,
                depth_only=0,
            )
        except Exception as e:
            import sys

            print(f"Metal postprocess pipeline error: {e}", file=sys.stderr)
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
        m = self._metal
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
            m.set_vertex_buffer(context=ctx, buffer=ubuf, index=0, offset=0)
            m.set_fragment_buffer(context=ctx, buffer=ubuf, index=0, offset=0)

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
        m = self._metal
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
            m.set_vertex_buffer(context=ctx, buffer=ubuf, index=0, offset=0)
            m.set_fragment_buffer(context=ctx, buffer=ubuf, index=0, offset=0)

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
        return self._metal.get_depth_texture(context=self._ctx)

    def create_color_texture(self, width, height):
        """Create an R8 texture for post-processing output."""
        return self._metal.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=1,
            linear_filter=1,
            depth_compare=0,
        )

    def create_r32f_texture(self, width, height):
        """Create an R32Float texture for high-precision accumulation."""
        return self._metal.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=4,
            linear_filter=1,
            depth_compare=0,
        )

    def begin_color_pass(self, texture, width, height, clear=True):
        """Begin a render pass targeting a color texture (no depth)."""
        self.begin_frame()
        self._metal.end_render_pass(context=self._ctx)
        self._metal.begin_color_pass(
            context=self._ctx,
            color_texture=texture,
            width=width,
            height=height,
            do_clear=1 if clear else 0,
        )

    def restart_main_pass(self):
        """Restart the main render pass preserving existing color and depth."""
        self._metal.end_render_pass(context=self._ctx)
        r, g, b, a = self._background_color
        self._metal.begin_render_pass(
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
        # Destroy previous AS.
        if self._rtao_accel_struct is not None:
            self._metal.destroy_accel_struct(
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
            )
            self._rtao_accel_struct = None
            self._rtao_sphere_buffer = None

        data = np.ascontiguousarray(sphere_data, dtype=np.float32)
        as_ptr, sb_ptr = self._metal.build_sphere_accel(
            context=self._ctx, sphere_data=data, sphere_count=len(sphere_data)
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
            self._metal.destroy_accel_struct(
                accel_struct=self._rtao_accel_struct,
                sphere_buffer=self._rtao_sphere_buffer,
            )
            self._rtao_accel_struct = None
            self._rtao_sphere_buffer = None
            self._rtao_cylinder_buffer = None

        sph = np.ascontiguousarray(sphere_data, dtype=np.float32)
        cyl = np.ascontiguousarray(cylinder_data, dtype=np.float32)
        as_ptr, sb_ptr, cb_ptr = self._metal.build_mixed_accel(
            context=self._ctx,
            sphere_data=sph, sphere_count=len(sphere_data),
            cylinder_data=cyl, cylinder_count=len(cylinder_data),
        )
        self._rtao_accel_struct = as_ptr
        self._rtao_sphere_buffer = sb_ptr
        self._rtao_cylinder_buffer = cb_ptr

    def _ensure_rtao_pipeline(self):
        """Compile the RTAO compute pipeline (once)."""
        if self._rtao_pipeline_compiled:
            return
        import os

        shader_path = os.path.join(os.path.dirname(__file__), "shaders", "rtao.metal")
        with open(shader_path) as f:
            source = f.read()
        self._metal.compile_rtao(context=self._ctx, shader_source=source)
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

        data = np.ascontiguousarray(uniforms)

        self._metal.end_render_pass(context=self._ctx)

        self._metal.dispatch_rtao(
            context=self._ctx,
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

        shader_path = os.path.join(
            os.path.dirname(__file__), "shaders", "pathtracer.metal"
        )
        with open(shader_path) as f:
            source = f.read()
        self._metal.compile_pathtracer(context=self._ctx, shader_source=source)
        self._pt_pipeline_compiled = True

    def create_rgba_texture(self, width, height):
        """Create an RGBA8 texture for path tracer output."""
        return self._metal.create_texture(
            context=self._ctx,
            width=width,
            height=height,
            format=0,
            linear_filter=1,
            depth_compare=0,
        )

    def dispatch_pathtracer(
        self,
        color_texture,
        prev_texture,
        accum_texture,
        color_buffer,
        cylinder_color_buffer,
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

        data = np.ascontiguousarray(uniforms)

        cyl_buf = self._rtao_cylinder_buffer or 0
        cyl_color_buf = cylinder_color_buffer or 0

        self._metal.end_render_pass(context=self._ctx)

        self._metal.dispatch_pathtracer(
            context=self._ctx,
            accel_struct=self._rtao_accel_struct,
            sphere_buffer=self._rtao_sphere_buffer,
            color_texture=color_texture,
            prev_texture=prev_texture,
            accum_texture=accum_texture,
            color_buffer=color_buffer,
            cylinder_buffer=cyl_buf,
            cylinder_color_buffer=cyl_color_buf,
            uniforms=data,
        )

    def supports_ray_tracing(self):
        """Check if the Metal device supports ray tracing."""
        try:
            return hasattr(self._metal, "build_sphere_accel")
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Stub sub-systems (minimal implementations to avoid crashes)
# ---------------------------------------------------------------------------


class MetalShadow:
    """Single directional shadow map state and binding for Metal."""

    def __init__(self, render):
        self._render = render
        self._shadow_texture = None
        self._shadow_transform = None
        self._shadow_view_transform = None
        self.multishadow_update_needed = False

    def compute_shadow_transforms(self, light_direction, center, radius, depth_bias):
        """Compute light-view inverse and scene→texture transform."""
        from chimerax.geometry import (
            normalize_vector,
            orthonormal_frame,
            translation,
            scale,
        )

        ld = normalize_vector(light_direction)
        lv = translation(center - radius * ld) * orthonormal_frame(-ld)
        lvinv = lv.inverse(is_orthonormal=True)
        # Metal Y-flip: shadow texture coords use opposite Y from OpenGL.
        stf = (
            translation((0.5, 0.5, -depth_bias))
            * scale((0.5 / radius, -0.5 / radius, -0.5 / radius))
            * lvinv
        )
        return lvinv, stf

    def set_scene_transform(self, stf, texture):
        """Store the scene→texture transform and texture handle."""
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
        m = render._metal
        ctx = render._ctx
        m.set_fragment_texture(context=ctx, texture=self._shadow_texture, index=3)


class MetalMultishadow:
    """Tiled multishadow map state and binding for Metal."""

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
        """Compute light-view inverse and tiled scene→texture transform."""
        from chimerax.geometry import (
            normalize_vector,
            orthonormal_frame,
            translation,
            scale,
        )

        ld = normalize_vector(light_direction)
        lv = translation(center - radius * ld) * orthonormal_frame(-ld)
        lvinv = lv.inverse(is_orthonormal=True)
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
        m = self._render._metal
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
        m = render._metal
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
