# 04 — Features, end to end

Each section: what the feature looks like, which files implement it, and how
the pieces talk to each other. Paths are relative to
`src/bundles/graphics/` unless noted.

## Impostors

**What**: atoms and bonds are drawn as tiny screen-facing quads whose
fragment shaders ray-trace a perfect sphere/cylinder per pixel. Compared to
tessellated meshes: ~6 vertices per atom instead of hundreds, pixel-perfect
silhouettes at any zoom, and exact analytic normals.

**Files**: `shaders/impostor_sphere.{vert,frag}`,
`shaders/impostor_cylinder.{vert,frag}` (Vulkan GLSL),
`shaders/impostor_*.metal` (Metal), the `USE_IMPOSTOR_*` sections of
`fragmentShader.txt` (OpenGL); geometry setup in
`atomic/src/structure.py` (`_set_impostor_quad_geometry`) and
`atomic/src/pbgroup.py`.

**How (spheres)**: the drawing's geometry is one unit quad; per-atom
instance data is the `shift_and_scale` array (xyz center + radius, vertex
binding 14). The vertex shader positions the quad in eye space *in front of*
the sphere (this ordering backs the `layout(depth_greater)` early-z promise)
and sizes it to cover the sphere's screen extent, with separate math for
perspective (rays from the eye) and orthographic (parallel rays —
`projection_matrix[3][3] != 0` detects ortho). The fragment shader solves
the ray/sphere quadratic; misses `discard`, hits compute the eye-space
intersection, shade with the standard lighting (plus shadows, sphere-AO),
and write the true `gl_FragDepth`. Because the depth is correct, impostors
compose seamlessly with meshes, volumes, and each other.

**How (cylinders/bonds)**: one instance **per bond** (placement matrix at
binding 15): rotation aligns the unit cylinder, translation is the bond
*midpoint*, and the second half-bond color rides in the matrix's 4th
column. Dashed pseudobonds don't generate dash geometry — the fragment
shader modulates along the axis using `impostor_dashes` from the uniform
block. Anything that consumes bond drawing `positions` (picking!) must know
this layout — see `_bonds_planes_pick` in `atomic/src/structure.py`.

**Sphere-AO**: per-pixel analytic ambient occlusion from neighboring atoms.
The atomic bundle attaches `_ao_offset_buffer`/`_ao_neighbor_buffer` SSBOs
(set 0, bindings 3/4) listing each atom's neighbors; the fragment shader
accumulates occluding solid angles at the hit point. Enabled via the
`USE_SPHERE_AO` specialization constant when the buffers exist.

## Lighting & depth cue

`activate_lighting` (per frame) packs key/fill/ambient light parameters into
the LightingParams UBO (set 0 binding 1) — the layout must match
`common.glsl` exactly. `compute_lighting` in `common.glsl` mirrors the
OpenGL fragment shader: key diffuse+specular (shadowed), fill diffuse,
ambient (multishadow-modulated).

Depth cue (distance fog toward `depth_cue_color`) is **on by default** in
ChimeraX. The cue range interpolates `lighting.depth_cue_start/end` across
the near/far clip distances; `_build_uniforms` packs it, shaders apply it
last (`USE_DEPTH_CUE`). If Vulkan output ever looks "flatter" than GL,
check this first — it shipped disabled once and the difference reads as
washed-out color rather than missing fog.

## Shadows & multishadow

**Single shadow** (`lighting shadows true`): `ShadowMapPass` renders shadow
casters into a depth texture from the key light's view (orthographic
projection sized to the scene bounding sphere), then scene shaders sample
it. `VulkanShadow.compute_shadow_transforms` builds the light view and the
scene→texture-coords transform (with the Vulkan Y flip and the depth bias
folded in). Scene shaders recompute the shadow coordinate per pixel
(impostors must use the ray-traced hit point, not the quad position) and do
a 16-tap Poisson PCF through a `sampler2DShadow`.

Shadow map quality knobs that must stay aligned with OpenGL:
comparison samplers use **linear** filtering (hardware PCF per tap — nearest
gives blocky edges) and **white border** clamping (outside the map = lit).

**Multishadow** (`lighting soft`): the same idea with N light directions
tiled into one big depth atlas (each tile rendered with its own
viewport+transform within a single depth pass). Per-tile scene→texture
transforms go into an SSBO (set 0 binding 2); shaders sum
diffuse-weighted shadow lookups over the tiles to approximate ambient
occlusion. The tile count/size logic lives in
`ShadowMapPass._render_multishadow`.

The **depth-only pass** (`begin_depth_pass`) renders with pipelines created
`depth_only=1` (no color attachment, into `depth_only_render_pass`); the
fragment shaders still run for impostors so the shadow map contains
ray-traced sphere depths, not quad depths.

## Transparency (OIT)

`TransparentPass` → `r.begin_oit_pass` / draw transparent / `r.end_oit_pass`
(`vulkan_oit.cpp`, `shaders/oit_composite.*`).

Weighted-blended OIT (McGuire–Bavoil): transparent fragments accumulate into
two attachments — `accum` (RGBA16F, additive, weighted by a depth-falloff
function) and `reveal` (R8, multiplicative transmittance) — sharing the
scene depth read-only so opaque geometry still occludes. `end_oit_pass`
resumes the main pass with `load_color/load_depth` and composites a
fullscreen quad: `avg = accum.rgb / accum.a`, blended with
`alpha = 1 - reveal`.

Pipelines for the accumulation use `oit_mode=1` (two-attachment blend state,
depth writes off, the `USE_OIT` specialization constant routes the fragment
output). The OIT framebuffer is rebuilt per `begin_oit_pass` so it always
attaches the *current* target's depth (capture vs. swapchain).

## Clip planes

`clipping.py` defines scene-space planes; when any are active the view sets
`SHADER_CLIP_PLANES` and calls `r.set_clip_parameters(planes)` with
`opengl_vec4()` vectors (nx, ny, nz, -d). Vulkan packs up to 8 planes plus
the count into the Uniforms block; vertex shaders compute per-vertex
signed distances against *scene-space* positions and fragments `discard`
where any distance is negative. The overlay/outline paths disable the
capability so 2D draws aren't clipped.

## Selection outlines

`HighlightOutlinePass` → `drawing.draw_highlight_outline` →
`r.outline.start_rendering_outline()` … draw highlighted …
`finish_rendering_outline(color, pixel_width)`.

Vulkan implementation (`VulkanOutline` in vulkan.py, `vulkan_begin_mask_pass`
in vulkan_context.cpp, `shaders/outline.*`):

1. **Mask pass**: a swapchain-format color texture (format 10 — required for
   pipeline compatibility) is cleared to transparent and the highlighted
   geometry is drawn **all white, unlit, depth-writes off**, depth-testing
   against the **live scene depth** attachment — so only visible parts mask.
   `LESS_OR_EQUAL` depth compare means re-drawing identical geometry passes
   at equal depth (no OpenGL-style depth-range bias needed).
2. **Subset drawing**: `draw_drawing(..., highlighted_only=True)` restricts
   to `highlighted_positions` (instance subset — sliced shift_and_scale /
   matrices / colors uploaded under separate cache keys) and
   `highlighted_displayed_triangles_mask` (triangle subset), so selecting
   three atoms outlines three atoms.
3. **Composite**: resume the main pass (load color+depth) and draw a
   fullscreen quad: a pixel gets the outline color iff its own mask sample
   is empty but a 4-neighbor within `pixel_width` is masked; everything else
   discards.

OpenGL does the same thing with a mask framebuffer + depth copy
(`opengl.Outline`); the Vulkan version skips the copy by attaching the live
depth. Outlines are skipped during offscreen captures (the mask shares the
swapchain-sized depth buffer).

## Silhouettes

`graphics silhouettes true`. Parameters (enabled, thickness, color,
depth_jump, perspective_near_far_ratio) live on the **view's**
`opengl.Silhouette` object regardless of backend; the render graph wires the
backend implementation to it each frame
(`render_graph._resolve_subsystems` → `r._silhouette_impl.params = view.silhouette`).

Vulkan implementation (`VulkanSilhouette`, `vulkan_copy_depth_texture`,
`shaders/silhouette.*`): the scene depth can't be sampled while attached, so
`SilhouetteFinishPass` copies it into a D32 texture (`vkCmdCopyImage`; the
scene depth image carries `TRANSFER_SRC` usage for this), resumes the main
pass, and runs a fullscreen quad that ports OpenGL's `USE_DEPTH_OUTLINE`
shader: take the min depth in a disc of radius `thickness`; where the center
depth jumps past it by more than `depth_jump` (perspective-corrected by the
near/far ratio) draw the silhouette color, else discard.
`SilhouetteDrawPass` runs the same composite before transparency so
transparent surfaces draw over silhouettes, as on GL.

## Drag-selection rectangle

OpenGL draws the rubber band by XOR-ing into the *front buffer* between
frames — a technique with no Vulkan/Metal equivalent. Instead,
`View.draw_xor_rectangle` detects non-front-buffer renderers and **toggles**
`view._overlay_rectangle` (toggle = the caller's XOR erase semantics work
unchanged; the color is XOR-ed back against the background to recover the
intended color). `View.draw` composites the active rectangle each frame via
`drawing.draw_overlay_rectangle` (ortho projection, depth test off, Mesh
style outline). The actual *picking* on mouse-up is geometric
(`view.rectangle_pick`) and renderer-independent.

## Volume rendering

Two paths, chosen by the volume code:

- **Plane stacks / multitexture**: textured quads; `multitexture` holds one
  texture per slice and `_draw_single` draws each slice's quad range with
  its texture bound. Blend variants: standard alpha, `blend_max` (maximum
  intensity projection → `VK_BLEND_OP_MAX`), `blend_alpha(False)`
  (premultiplied source factor).
- **Ray casting** (`SHADER_VOLUME_RAYCASTING`, `shaders/raycast.*`): draw
  the volume's bounding box (back faces — note culling is *enabled* here and
  front face is CLOCKWISE); the fragment shader reconstructs the eye ray
  from `gl_FragCoord` using the inverse view/projection, intersects the box,
  and marches the 3D texture, compositing with the colormap. Parameters
  arrive via `set_volume_parameters`/`set_bounding_box_planes` (cached on
  the renderer, packed into the raycast UBO at set 0 binding 6 by
  `_upload_raycast_uniforms`).

Colormaps: 1D texture at set 1 binding 6 plus a (offset, scale) UBO at
set 0 binding 5 (`set_colormap`), matching OpenGL's normalization math.

## Overlays, labels, on-top drawings

`draw_overlays` / `draw_on_top` (drawing.py): identity/ortho projection,
**depth test disabled** (`enable_depth_test(False)` — on Vulkan this is a
pipeline-key flag, so overlays genuinely ignore the scene depth), blending
on for transparent label textures.

## Known feature gaps (Vulkan)

Tracked in more detail in [06-debugging.md](06-debugging.md):
crossfade/motion blur (needs swapchain readback), stereo cameras and 360
modes, GTAO (Metal-only shader), `offscreen`/`blend` subsystems (stubs),
selection outlines and silhouettes during image capture.
