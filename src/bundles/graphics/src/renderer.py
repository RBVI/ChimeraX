"""
Abstract renderer protocol for ChimeraX graphics backends.

This module defines the interface that any graphics backend (OpenGL, Vulkan, Metal)
must implement to render the ChimeraX scene graph. Drawing and View never call
graphics API functions directly — they call methods on these protocol objects.

To implement a new backend:
1. Subclass Renderer and implement all abstract methods
2. Subclass ShaderProgram, Buffer, VertexArray, Texture, Framebuffer
3. Register your Renderer factory with the graphics system

The current OpenGL implementation lives in opengl.py (Render class).
"""

from __future__ import annotations
from typing import Protocol, Optional, runtime_checkable
from numpy import ndarray


# ---------------------------------------------------------------------------
# Render pass descriptor
# ---------------------------------------------------------------------------

class PassDescriptor:
    """Describes a render pass's attachments and clear behaviour.

    Attributes
    ----------
    depth_texture :
        GPU depth texture to render into.  *None* means the default
        depth buffer (screen).
    color_texture :
        GPU color texture to render into.  *None* means the screen
        drawable (or no color attachment when *depth_only* is True).
    clear_color :
        ``(r, g, b, a)`` to clear the color buffer, or *None* to
        preserve existing contents.
    clear_depth :
        Depth value to clear to (typically ``1.0``), or *None* to
        preserve existing contents.
    depth_only :
        If True, disable color writes — only depth is rendered.
    viewport :
        ``(x, y, w, h)`` initial viewport, or *None* for full target.
    """

    __slots__ = ('depth_texture', 'color_texture',
                 'clear_color', 'clear_depth',
                 'depth_only', 'viewport')

    def __init__(self, *, depth_texture=None, color_texture=None,
                 clear_color=None, clear_depth=None,
                 depth_only=False, viewport=None):
        self.depth_texture = depth_texture
        self.color_texture = color_texture
        self.clear_color = clear_color
        self.clear_depth = clear_depth
        self.depth_only = depth_only
        self.viewport = viewport


# ---------------------------------------------------------------------------
# Shader / Pipeline
# ---------------------------------------------------------------------------

@runtime_checkable
class ShaderProgram(Protocol):
    """
    A compiled shader program (OpenGL) or pipeline object (Vulkan/Metal).

    In OpenGL, this wraps a linked program with cached uniform locations.
    In Vulkan, this wraps a VkPipeline + VkPipelineLayout + descriptor sets.
    In Metal, this wraps an MTLRenderPipelineState.

    The `capabilities` field is the bitmask that was used to compile/select
    this program. It determines which features (lighting, shadows, impostors,
    etc.) are active.
    """
    capabilities: int

    def set_integer(self, name: str, value: int) -> None: ...
    def set_float(self, name: str, value: float) -> None: ...
    def set_vector2(self, name: str, vector: tuple | ndarray) -> None: ...
    def set_vector3(self, name: str, vector: tuple | ndarray) -> None: ...
    def set_rgba(self, name: str, color: tuple | ndarray) -> None: ...
    def set_float4(self, name: str, v4: ndarray, count: int = 1) -> None: ...
    def set_float3_array(self, name: str, vectors: ndarray) -> None: ...
    def set_matrix(self, name: str, matrix: ndarray) -> None: ...


# ---------------------------------------------------------------------------
# GPU Buffers
# ---------------------------------------------------------------------------

@runtime_checkable
class Buffer(Protocol):
    """
    A GPU buffer for vertex, index, or instance data.

    Buffer types define the attribute binding (position, normal, color, etc.)
    and whether data is per-vertex or per-instance.
    """

    def update_buffer_data(self, data: ndarray) -> None:
        """Upload numpy array to GPU. May reallocate if size changed."""
        ...

    def delete_buffer(self) -> None: ...


@runtime_checkable
class VertexArray(Protocol):
    """
    A collection of buffer bindings (OpenGL VAO, Vulkan vertex input state).

    Associates attribute locations with specific buffers.
    """

    def activate(self) -> None:
        """Bind this vertex array for subsequent draw calls."""
        ...

    def bind_shader_variable(self, buffer: Buffer) -> None:
        """Associate a buffer with its shader attribute location."""
        ...

    def delete_bindings(self) -> None: ...


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

@runtime_checkable
class Texture(Protocol):
    """
    A GPU texture (1D, 2D, 3D, or cubemap).

    Wraps OpenGL texture objects, VkImage+VkSampler, or MTLTexture.
    """
    dimension: int          # 1, 2, or 3
    size: tuple             # (w,) or (w,h) or (w,h,d)
    is_cubemap: bool
    linear_interpolation: bool

    def initialize_rgba(self, size: tuple) -> None: ...
    def initialize_8_bit(self, size: tuple) -> None: ...
    def initialize_depth(self, size: tuple, depth_compare_mode: bool = True) -> None: ...
    def bind_texture(self, texture_unit: Optional[int] = None) -> None: ...
    def unbind_texture(self, texture_unit: Optional[int] = None) -> None: ...
    def reload_texture(self, data: ndarray, now: bool = False) -> None: ...
    def delete_texture(self) -> None: ...


# ---------------------------------------------------------------------------
# Framebuffers
# ---------------------------------------------------------------------------

@runtime_checkable
class Framebuffer(Protocol):
    """
    An offscreen render target with optional color and depth attachments.

    Wraps OpenGL FBO, VkFramebuffer+VkRenderPass, or MTLRenderPassDescriptor.
    """
    name: str
    width: int
    height: int
    color: bool
    depth: bool
    alpha: bool
    color_texture: Optional[Texture]
    depth_texture: Optional[Texture]
    viewport: tuple         # (x, y, w, h)

    def activate(self) -> bool:
        """Bind this framebuffer as the render target. Returns False on failure."""
        ...

    def attach_depth_texture(self, texture: Optional[Texture]) -> None: ...

    def copy_from_framebuffer(self, source: 'Framebuffer',
                              color: bool = True, depth: bool = True) -> None:
        """Blit color and/or depth from source framebuffer to this one."""
        ...

    def copy_to_framebuffer(self, dest: 'Framebuffer',
                            color: bool = True, depth: bool = True) -> None: ...

    def delete(self, make_current: bool = False) -> None: ...


# ---------------------------------------------------------------------------
# Renderer (the main backend interface)
# ---------------------------------------------------------------------------

# Shader capability flags. These are the same across all backends.
# Each flag corresponds to a shader feature that can be independently
# enabled or disabled.
#
# In OpenGL, these map to #define macros in the uber-shader.
# In Vulkan/Metal, these map to pipeline variant selection or
# specialization constants.

SHADER_LIGHTING             = 1 << 0
SHADER_LIGHTING_NORMALS     = 1 << 1
SHADER_DEPTH_CUE            = 1 << 2
SHADER_NO_DEPTH_CUE         = 1 << 3
SHADER_TEXTURE_2D           = 1 << 4
SHADER_TEXTURE_3D           = 1 << 5
SHADER_COLORMAP             = 1 << 6
SHADER_DEPTH_TEXTURE        = 1 << 7
SHADER_TEXTURE_CUBEMAP      = 1 << 8
SHADER_TEXTURE_3D_AMBIENT   = 1 << 9
SHADER_BLEND_TEXTURE_2D     = 1 << 10
SHADER_BLEND_TEXTURE_3D     = 1 << 11
SHADER_BLEND_COLORMAP       = 1 << 12
SHADER_SHADOW               = 1 << 13
SHADER_NO_SHADOW            = 1 << 14
SHADER_MULTISHADOW          = 1 << 15
SHADER_NO_MULTISHADOW       = 1 << 16
SHADER_SHIFT_AND_SCALE      = 1 << 17
SHADER_INSTANCING           = 1 << 18
SHADER_TEXTURE_OUTLINE      = 1 << 19
SHADER_DEPTH_OUTLINE        = 1 << 20
SHADER_VERTEX_COLORS        = 1 << 21
SHADER_FRAME_NUMBER         = 1 << 22
SHADER_TRANSPARENT_ONLY     = 1 << 23
SHADER_OPAQUE_ONLY          = 1 << 24
SHADER_STEREO_360           = 1 << 25
SHADER_CLIP_PLANES          = 1 << 26
SHADER_NO_CLIP_PLANES       = 1 << 27
SHADER_ALPHA_DEPTH          = 1 << 28
SHADER_ALL_WHITE            = 1 << 29
SHADER_VOLUME_RAYCASTING    = 1 << 30
SHADER_SHOW_DEPTH_BUFFER    = 1 << 31
SHADER_IMPOSTOR_SPHERE      = 1 << 32
SHADER_IMPOSTOR_CYLINDER    = 1 << 33
SHADER_GTAO                 = 1 << 34


@runtime_checkable
class Renderer(Protocol):
    """
    The main graphics backend interface.

    Drawing and View call only methods on this object (and on the objects
    it returns). No graphics API calls should exist outside of Renderer
    implementations.
    """

    # --- Active state (read by Drawing/View) ---
    current_shader_program: Optional[ShaderProgram]
    current_projection_matrix: Optional[ndarray]
    current_view_matrix: Optional[ndarray]
    enable_capabilities: int
    disable_capabilities: int

    # --- Sub-systems (managed by the renderer) ---
    lighting: object        # Lighting parameters (key/fill/ambient)
    material: object        # Surface material properties
    shadow: object          # Single shadow map manager
    multishadow: object     # Multi-directional AO manager
    silhouette: object      # Silhouette edge renderer
    outline: object         # Selection outline renderer
    offscreen: object       # Offscreen compositing
    blend: object           # Multi-channel texture blending

    # -----------------------------------------------------------------------
    # Context management
    # -----------------------------------------------------------------------

    def make_current(self) -> None:
        """Make this renderer's context current for the calling thread."""
        ...

    def done_current(self) -> None:
        """Release the context from the calling thread."""
        ...

    def swap_buffers(self) -> None:
        """Present the back buffer to the display."""
        ...

    # -----------------------------------------------------------------------
    # Pipeline / shader selection
    # -----------------------------------------------------------------------

    def shader(self, capabilities: int) -> ShaderProgram:
        """
        Select (and compile if needed) a shader program for the given
        capability flags. Activates the program for subsequent draw calls.

        The capabilities bitmask is OR'd with enable_capabilities and
        AND'd with ~disable_capabilities before lookup.

        OpenGL: compiles vertex+fragment shaders with #defines.
        Vulkan: selects pre-compiled VkPipeline from cache.
        Metal: selects MTLRenderPipelineState.
        """
        ...

    def disable_shader_capabilities(self, mask: int) -> None:
        """Set the disable mask. Capabilities in this mask are removed
        from all subsequent shader() calls."""
        ...

    def enable_shader_capability(self, cap: int, enable: bool) -> None:
        """Add or remove a capability from the enable mask."""
        ...

    def draw_depth_only(self, depth_only: bool = True) -> None:
        """
        Enter or exit depth-only rendering mode. In this mode, only
        geometry and depth are processed — no colors, lighting, or
        textures. Color writes are disabled.
        """
        ...

    # -----------------------------------------------------------------------
    # Transform matrices
    # -----------------------------------------------------------------------

    def set_projection_matrix(self, pm: Optional[ndarray] = None) -> None:
        """Set the 4x4 projection matrix. None reuses the last one."""
        ...

    def set_view_matrix(self, vm: ndarray) -> None:
        """Set the 4x4 camera view matrix (scene → camera coords)."""
        ...

    def set_model_matrix(self, model_matrix: Optional[object] = None) -> None:
        """
        Set the 4x4 model matrix. Internally computes model_view_matrix
        and updates the shader uniform. None uses identity.
        The argument is a Place object with an .opengl_matrix() method.
        """
        ...

    # -----------------------------------------------------------------------
    # Render state
    # -----------------------------------------------------------------------

    def set_background_color(self, rgba: tuple) -> None: ...
    def draw_background(self, depth: bool = True) -> None:
        """Clear the color and/or depth buffers."""
        ...

    def allow_equal_depth(self, equal: bool) -> None:
        """Switch depth function between GL_LESS and GL_LEQUAL."""
        ...

    def set_depth_range(self, near: float, far: float) -> None:
        """Set the depth range mapping from NDC to window coords."""
        ...

    def set_model_color(self, color: Optional[tuple] = None) -> None:
        """Set the uniform color for non-vertex-colored geometry."""
        ...

    def set_frame_number(self, frame: Optional[int] = None) -> None: ...

    # -----------------------------------------------------------------------
    # Render pass management
    # -----------------------------------------------------------------------

    def begin_pass(self, descriptor: 'PassDescriptor') -> None:
        """Begin a render pass targeting the attachments in *descriptor*.

        OpenGL: push a framebuffer, optionally clear, enter depth-only mode.
        Metal:  end current encoder, begin a new render pass with load/store
                actions derived from the descriptor.
        """
        ...

    def end_pass(self) -> None:
        """End the current render pass and restore the previous state.

        OpenGL: exit depth-only mode if active, pop framebuffer.
        Metal:  end the current render command encoder.
        """
        ...

    # -----------------------------------------------------------------------
    # Viewport and framebuffer management
    # -----------------------------------------------------------------------

    def render_size(self) -> tuple:
        """Return (width, height) of the current render target."""
        ...

    def default_framebuffer(self) -> Framebuffer: ...
    def current_framebuffer(self) -> Framebuffer: ...

    def set_viewport(self, x: int, y: int, w: int, h: int) -> None: ...
    def full_viewport(self) -> None:
        """Reset viewport to full framebuffer size."""
        ...

    def push_framebuffer(self, fb: Framebuffer) -> None:
        """Push a framebuffer onto the stack and activate it."""
        ...

    def pop_framebuffer(self) -> Framebuffer:
        """Pop the top framebuffer and activate the one beneath it."""
        ...

    # -----------------------------------------------------------------------
    # Lighting and effects
    # -----------------------------------------------------------------------

    def activate_lighting(self) -> None:
        """Bind the lighting uniform buffer for subsequent draws."""
        ...

    def update_lighting_parameters(self) -> None:
        """Upload lighting parameters to the GPU."""
        ...

    def set_depth_cue_parameters(self) -> None:
        """Update depth cue (fog) uniforms from current settings."""
        ...

    def set_clip_parameters(self) -> None:
        """Update clip plane uniforms."""
        ...

    # -----------------------------------------------------------------------
    # Texture binding
    # -----------------------------------------------------------------------

    def set_colormap(self, colormap: Texture, colormap_range: tuple,
                     data_texture: Texture) -> None: ...

    def set_ambient_texture_transform(self, transform: ndarray) -> None: ...

    # -----------------------------------------------------------------------
    # Stereo and special rendering
    # -----------------------------------------------------------------------

    def set_stereo_buffer(self, eye_num: Optional[int]) -> None: ...
    def set_stereo_360_params(self, origin: ndarray, vertical: ndarray,
                              shift: float) -> None: ...

    # -----------------------------------------------------------------------
    # Draw commands
    # -----------------------------------------------------------------------

    def draw_elements(self, mode: int, count: int, offset: int = 0,
                      instance_count: int = 1) -> None:
        """
        Issue an indexed draw call.

        mode: primitive type
            TRIANGLES = 0
            LINES = 1
            POINTS = 2
        count: number of indices to draw
        offset: byte offset into the index buffer
        instance_count: number of instances (1 = non-instanced)

        OpenGL: glDrawElementsInstanced(mode, count, GL_UNSIGNED_INT, offset, instance_count)
        Vulkan: vkCmdDrawIndexed(cmd, count, instance_count, offset, 0, 0)
        Metal: drawIndexedPrimitives(type, count, indexType, indexBuffer, offset, instance_count)
        """
        ...


# ---------------------------------------------------------------------------
# Primitive types (backend-agnostic constants)
# ---------------------------------------------------------------------------

TRIANGLES = 0
LINES = 1
POINTS = 2
