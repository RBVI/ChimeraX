# 01 — Architecture

How a ChimeraX frame goes from "the user rotated the molecule" to pixels.

## The layers

```
┌─────────────────────────────────────────────────────────────────┐
│ ChimeraX application (models, commands, tools)                  │
│   models create/update Drawing objects                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ View (view.py)                                                  │
│   owns: camera, root Drawing, Lighting/Material/Silhouette,     │
│         the renderer, the render graph                          │
│   View.draw() runs once per frame                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ RenderGraph (render_graph.py) + passes (render_passes.py)      │
│   an ordered list of passes, each with enabled(ctx)/execute(ctx)│
└──────────────────────────┬──────────────────────────────────────┘
                           │  renderer protocol calls
┌──────────────────────────▼──────────────────────────────────────┐
│ Renderer — one of:                                              │
│   opengl.Render          (reference implementation)            │
│   metal.MetalRender      (macOS native)                        │
│   vulkan.VulkanRender    (Linux/Windows; macOS via MoltenVK)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │  ctypes-free C extension calls
┌──────────────────────────▼──────────────────────────────────────┐
│ C extensions: PyOpenGL / _metal / _vulkan                       │
│   record GPU command buffers, manage memory, present frames     │
└─────────────────────────────────────────────────────────────────┘
```

The key design idea: everything above the renderer is backend-agnostic.
A pass never calls Vulkan directly; it calls renderer methods like
`begin_pass`, `draw_fullscreen_quad`, `enable_depth_test`. The OpenGL
renderer is the reference: when in doubt about what a protocol method should
do, read `opengl.py`.

## Backend selection

`ui/src/graphics.py` chooses the windowing context at startup based on the
`CHIMERAX_GRAPHICS_BACKEND` environment variable (`vulkan`, `metal`, or
default OpenGL). It constructs the matching context class — e.g.
`vulkan.VulkanContext`, which creates the Vulkan instance/device/swapchain
from the Qt window's native handle (on macOS, `_metal_rt.create_surface`
makes the `CAMetalLayer`-backed `VkSurfaceKHR` first).

`View.initialize_rendering` (view.py) then sniffs the context object:

```python
if hasattr(graphics_context, "_vulkan"):   -> VulkanRender
elif hasattr(graphics_context, "_metal"):  -> MetalRender
else:                                      -> opengl.Render
```

The renderer object lives for the lifetime of the window. Lighting, material,
and silhouette parameter objects are owned by the `View` and assigned onto
the renderer, so user commands (`lighting`, `graphics silhouettes`, …) work
identically across backends.

## The Drawing tree

`Drawing` (drawing.py) is the scene graph node. Important fields:

- `vertices`, `normals`, `triangles`, `texture_coordinates` — numpy arrays.
- `positions` — a `Places` collection: one or more 3×4 placement matrices.
  More than one position (or a `shift_and_scale_array`) means **instancing**:
  the geometry is drawn once per position in a single draw call.
- `colors` (per instance) and `vertex_colors` (per vertex).
- `display_style` — `Solid` (triangles), `Mesh` (edges), `Dot` (points).
- highlight state: `highlighted_positions` (per-instance bool mask) and
  `highlighted_triangles_mask` (per-triangle) drive selection outlines.

Drawing dispatch: `Drawing._draw_geometry` checks whether the renderer has a
`draw_drawing` method. The Metal/Vulkan renderers do — they receive the
Drawing object directly and manage their own GPU buffers
(`VulkanRender._draw_single`). OpenGL instead uses the Drawing's own
`_vertex_buffers` / vertex-array machinery.

### Impostor geometry

In this branch, atoms and bonds are not tessellated meshes. The drawings
carry a tiny screen-facing quad as geometry and per-atom/per-bond instance
data; the fragment shader ray-traces the analytic sphere/cylinder per pixel
(see [04-features.md](04-features.md#impostors)). Two layout consequences
that other code must respect:

- **Atoms**: `positions.shift_and_scale_array()` is an N×4 array of
  (x, y, z, radius) — one instance per atom.
- **Bonds**: one placement matrix **per bond** (not the classic two
  half-bond cylinders). The translation is the bond midpoint and the second
  half-bond color is packed into the matrix's 4th column. Picking code that
  assumed 2-per-bond placements has to branch
  (see `_bonds_planes_pick` in `atomic/src/structure.py`).

## Anatomy of a frame

`View.draw()` (view.py) does, in order:

1. `check_for_drawing_change` — decides if a redraw is needed.
2. `r.set_background_color`, `r.update_viewport`, lighting refresh,
   `r.activate_lighting()` (uploads the lighting uniform buffer).
3. `_draw_scene` → builds a `RenderContext` (camera, categorized drawings)
   and executes the render graph.
4. Overlays (`draw_overlays`) and, if a drag-selection is in progress, the
   rubber-band rectangle (`draw_overlay_rectangle`).
5. `r.end_frame()` / `r.swap_buffers()` — submit and present.

Drawings are pre-sorted into categories per frame (`_drawings_by_pass`):
opaque, transparent, highlighted, on-top.

### Render graph pass order

Built once in `View._build_render_graph`:

```
pre-view:   ShadowMapPass          render single-shadow + multishadow depth maps
per-view:   OffscreenStartPass     (OpenGL offscreen workarounds)
            SilhouetteStartPass    (no-op on Vulkan; GL redirects to offscreen FB)
            BackgroundPass         clear color+depth
            ViewSetupPass          projection/view matrices, shadow view transforms
            DepthPrepass           depth-only pre-pass (multishadow/GTAO only)
            GTAOPass               screen-space AO (Metal only)
            OpaquePass             the main opaque geometry
            RTAOPass               ray-traced AO compute + composite
            PathTracePass          full path tracer (replaces raster output)
            OutlineMaskPass        (OpenGL outline depth mask setup)
            SilhouetteDrawPass     silhouettes behind transparency
            TransparentPass        transparent geometry (OIT when available)
            DepthResetPass
            SilhouetteFinishPass   silhouette edge composite
            HighlightOutlinePass   green selection outlines
            OnTopPass              labels and always-on-top geometry
            OffscreenFinishPass
```

Each pass implements `enabled(ctx)` and `execute(ctx)`. `ctx` is the
`RenderContext`: renderer, view, camera, categorized drawing lists, plus
per-frame resolved subsystems (`ctx.silhouette`, `ctx.offscreen`, shadow
flags). `RenderGraph._resolve_subsystems` is where backend-specific
subsystem objects get wired — e.g. on Vulkan, `ctx.silhouette` is the
renderer's `VulkanSilhouette` implementation configured with the view's
parameter object.

**Gating passes by capability, not by backend name.** Passes test for
renderer abilities (`hasattr(r, "draw_fullscreen_quad")`,
`hasattr(r, "_metal")`, `r.supports_ray_tracing()`,
`hasattr(outline, "start_rendering_outline")`) so a backend that grows a
feature starts getting the pass automatically.

## The capability system

`renderer.py` defines ~35 `SHADER_*` bit flags. A draw's capability mask is
assembled from three sources:

1. The Drawing's own needs (`Drawing._shader_options`): vertex colors,
   textures, instancing, impostor flags, transparency pass selection.
2. The renderer's `enable_capabilities` — bits passes turn on globally
   (e.g. `SHADER_SHADOW` while shadows are active, `SHADER_ALL_WHITE` during
   the outline mask pass).
3. The renderer's `disable_capabilities` — bits passes force off (e.g.
   `draw_depth` disables lighting/shadows for shadow-map rendering).

Effective mask: `(drawing_caps | enable) & ~disable`.

What the mask *means* differs per backend:

- **OpenGL** generates a shader per mask by `#define`-ing `USE_*` macros into
  the template shaders (`vertexShader.txt` / `fragmentShader.txt`).
- **Vulkan** maps each bit to a **specialization constant**
  (`layout(constant_id = N) const bool USE_*`) passed at pipeline creation —
  same GLSL source, the driver compiles the branches away. The mapping lives
  in `VulkanRender._ensure_shader_pipeline` and must match the
  `constant_id` declarations in `shaders/*.vert|.frag`.

## The renderer protocol (abridged)

There is no formal interface class; `opengl.Render` defines it by example.
The essential surface, grouped:

**Frame & passes** — `begin_frame` (implicit), `end_frame`, `swap_buffers`,
`begin_pass(PassDescriptor)`, `end_pass`, `begin_oit_pass`/`end_oit_pass`.

**Matrices** — `set_projection_matrix`, `set_view_matrix`,
`set_model_matrix`, `set_near_far_clip`. All take ChimeraX-convention
matrices; backends convert internally (see 02-conventions).

**State toggles** — `enable_depth_test`, `enable_blending`, `blend_alpha`,
`blend_max`, `write_depth`, `enable_backface_culling`,
`allow_equal_depth`, `set_viewport`, `full_viewport`.
On Vulkan most of these set Python-side flags that feed the *pipeline cache
key* rather than issuing GPU commands.

**Drawing** — `draw_drawing(drawing, capabilities, highlighted_only)`
(Metal/Vulkan direct path), `draw_fullscreen_quad(texture, shader, ...)`
(post-processing), `draw_transparent`, `draw_volume_raycast`.

**Subsystems** — `r.shadow`, `r.multishadow`, `r.outline`, `r.silhouette`,
`r.offscreen`, `r.blend`. Each backend supplies its own implementations
(or stubs). The render graph consults them.

**Capture** — `create_framebuffer`/`push_framebuffer`/`pop_framebuffer` and
`frame_buffer_image` implement "save image" by redirecting a frame into an
offscreen target.

When adding a renderer method, add it to `opengl.py` first (or confirm it
exists there), then mirror it in `metal.py`/`vulkan.py`. A missing method is
an `AttributeError` mid-frame, which aborts the whole draw.
