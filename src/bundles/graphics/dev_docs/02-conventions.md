# 02 — Conventions

Coordinate systems, matrix storage, depth, winding, color. Most rendering
bugs that "look wrong but don't crash" are violations of something on this
page.

## Matrices

### ChimeraX storage is transposed

ChimeraX stores matrices in numpy arrays where **storage row *i* holds math
column *i***. A `Place` (chimerax.geometry) is a 3×4 array whose **last
column** `[:, 3]` is the translation; its `opengl_matrix()` returns a 4×4
where the **last row** holds the translation — i.e. the OpenGL-style
column-major data laid out in a row-major numpy array.

Why this works out in shaders: uploading that numpy array's bytes into a
GLSL `mat4` (std140, column-major) lands math column *i* in GLSL column *i*,
so shaders write the natural `u.projection_matrix * vec4(pos, 1.0)`.

Rules of thumb:

- Building a projection by hand? Look at `camera.py` `ortho()`/`frustum()`
  and `_shadow_projection()` in `render_passes.py` — copy their layout.
- Math element (row r, col c) lives at numpy `[c][r]`. The diagonal is safe.
- A perspective matrix has numpy `[3][3] == 0`; orthographic has `1`.
  Shaders use `u.projection_matrix[3][3] != 0.0` to detect ortho (the
  diagonal survives the transpose).

### GL → Vulkan projection conversion

ChimeraX cameras produce OpenGL-convention projections: NDC x,y,z all in
[-1, 1], Y up. Vulkan wants z in [0, 1] and has Y pointing **down** in NDC.
`VulkanRender.set_projection_matrix` converts every incoming matrix:

```python
vk_pm = gl_pm.copy()
vk_pm[:, 1] = -gl_pm[:, 1]                       # flip Y (math row 1)
vk_pm[:, 2] = 0.5 * gl_pm[:, 2] + 0.5 * gl_pm[:, 3]   # z: [-1,1] -> [0,1]
```

(Remember the transpose: numpy **column** j is math **row** j, which is why
the slicing is by column.)

Every matrix that reaches the Vulkan backend through
`set_projection_matrix` gets this treatment — camera projections, the
shadow-map ortho, the overlay ortho. Code that *inverts* the projection
(`raycast`, RTAO, path tracer) inverts the already-converted matrix, so the
shader math stays internally consistent.

### Consequences of the Y flip

1. **Triangle winding mirrors.** Geometry authored CCW-front (the OpenGL
   convention) rasterizes **clockwise** after the flip. Vulkan pipelines
   therefore use `VK_FRONT_FACE_CLOCKWISE` (`vulkan_pipeline.cpp`). If you
   port any culling logic from `opengl.py`, the facing is inverted.
2. **Texture V origin.** Vulkan/Metal put texel row 0 at the *top*. The
   fullscreen-quad texcoords (`vulkan.py _ensure_quad_buffers`) and the
   shadow scene-to-texture transform (`VulkanShadow
   .compute_shadow_transforms`, note the `-0.5/radius` Y scale) already
   account for this. If a post-process output is vertically flipped, suspect
   the UV mapping, not the projection.

## Depth

- Vulkan depth range is [0, 1]; clear value 1.0; compare op
  `LESS_OR_EQUAL` everywhere. `LESS_OR_EQUAL` (not `LESS`) is required so a
  depth pre-pass followed by an identical color pass doesn't z-fight itself.
- `common.glsl` provides `ndc_to_depth` — with the converted projection the
  NDC z already *is* the [0,1] depth, so it just clamps.
- The impostor fragment shaders write `gl_FragDepth` (the ray-traced surface
  depth, not the proxy quad's) and declare `layout(depth_greater)`: a promise
  to the driver that written depth ≥ rasterized quad depth, preserving
  early-z. The vertex shaders uphold the promise by placing the proxy quad
  in front of the analytic surface — don't move the quad without rechecking.
- The near/far clip values used by depth cue and impostor near-plane clamps
  flow through `r.set_near_far_clip(near_far)` into the uniform block
  (`depth_range_pack.zw`).

## Color space (macOS)

Shaders compute colors in a nominal sRGB-ish space and write them raw —
there is **no** linear-workflow/gamma conversion anywhere, matching the
OpenGL backend.

On macOS there is one extra subtlety: the compositor color-matches *tagged*
surfaces. Qt's OpenGL drawable is untagged, so its pixels are interpreted in
the display's native gamut (usually P3 — slightly oversaturated, but it is
the look users know). MoltenVK tags the `CAMetalLayer` according to the
swapchain's `VkColorSpace`; an sRGB-tagged layer gets converted into P3 and
looks ~10% desaturated next to the GL window. The backend therefore requests
`VK_COLOR_SPACE_PASS_THROUGH_EXT` (via the `VK_EXT_swapchain_colorspace`
instance extension) with an sRGB fallback, leaving the layer untagged to
match OpenGL. See `build_vulkan_instance` and `create_swapchain` in
`vulkan_context.cpp`.

## Texture format codes

`_vulkan.create_texture(format=N)` (and `vulkan.py`'s `_FORMAT_*`
constants):

| Code | VkFormat | Used for |
|---|---|---|
| 0 | `R8G8B8A8_UNORM` | color textures, labels, GTAO/RTAO outputs |
| 1 | `R8_UNORM` | single-channel volume data |
| 2 | `D32_SFLOAT` | shadow maps, scene depth copies |
| 3 | `D32_SFLOAT_S8_UINT` | (reserved; no current caller) |
| 4 | `R32_SFLOAT` | float volume data, RTAO accumulation |
| 5–8 | `R16_UNORM/SFLOAT/SNORM`, `R8_SNORM` | volume data variants |
| 9 | `R16G16B16A16_SFLOAT` | path-tracer output/accumulation (8-bit would quantize the temporal average; RGBA16F is the float format with *mandatory* support for both storage images and linear sampling) |
| 10 | the swapchain's own format | render targets that **scene pipelines** draw into outside the main pass (outline mask). Render-pass compatibility requires matching attachment formats. |

Single-channel color formats get an `R→GB` component swizzle so `.rgb`
samples read (L, L, L) — mirroring OpenGL's texture swizzle that colormap
lookups rely on. Storage-capable textures skip the swizzle (Vulkan requires
identity swizzle on storage image views).

Depth textures created with `depth_compare=1` get a comparison sampler
(`LESS_OR_EQUAL`) — that is `sampler2DShadow` in GLSL. `linear_filter=1` on
a comparison sampler enables hardware PCF (each lookup bilinearly filters
the comparison results) and is what makes shadow edges smooth; the samplers
also use `CLAMP_TO_BORDER` with a white border so out-of-map lookups read
"fully lit", matching OpenGL.

## Descriptor layout (Vulkan)

All graphics pipelines share one pipeline layout with two descriptor sets
(`vulkan_internal.h`, `create_descriptor_infrastructure`):

```
set 0 — buffers (7 bindings):
  0  UBO   Uniforms          (matrices, color, clip planes, depth cue — common.glsl)
  1  UBO   LightingParams    (uploaded by activate_lighting)
  2  SSBO  multishadow matrices
  3  SSBO  sphere-AO offsets
  4  SSBO  sphere-AO neighbors
  5  UBO   ColormapRange
  6  UBO   VolumeRaycast params

set 1 — combined image samplers (8 bindings):
  0  primary 2D texture        4  multishadow map (sampler2DShadow)
  1  3D texture (volumes)      5  ambient 3D texture
  2  cubemap                   6  1D colormap
  3  shadow map (2DShadow)     7  reserved (GTAO)
```

Two invariants:

1. **Every binding a shader statically declares must hold a valid
   descriptor**, even if the runtime path never samples it (specialization
   constants don't remove the declaration). `flush_descriptors`
   (`vulkan_buffer.cpp`) fills unbound slots with typed dummies from
   `vulkan_dummies.cpp`. If you add a binding to `common.glsl`, add a dummy.
2. **Descriptor sets are never reused within a frame's command buffer.**
   Each draw with dirty bindings allocates fresh sets from a per-frame pool
   (reset in `begin_frame`). Updating an already-bound set is illegal in
   Vulkan without an extension we don't use.

The compute pipelines (RTAO, path tracer) have their own separate set
layouts — see [05-raytracing.md](05-raytracing.md).

## Vertex input layout (Vulkan)

Every graphics pipeline declares the same six vertex bindings
(`vulkan_pipeline.cpp`); drawings bind the subset they have and
`_bind_dummy_vertex_buffers` fills the rest (a draw with any declared
binding unbound is invalid):

| Binding | Rate | Content | Attribute locations |
|---|---|---|---|
| 10 | vertex | position vec3 | 0 |
| 11 | vertex | texcoord vec3 (xy for 2D, xyz for 3D) | 1 |
| 12 | vertex | normal vec3 | 2 |
| 13 | vertex *or* instance | color RGBA8 (rate = per-instance for impostors) | 3 |
| 14 | instance | shift_and_scale vec4 (xyz offset + radius) | 4 |
| 15 | instance | 4×4 placement matrix | 5,6,7,8 (one per column) |

If a frame's *first* draw is a fullscreen quad, the quad helpers also bind
the dummies — bindings reset with each new command buffer.
