# 06 — Building, debugging, and known gaps

## Building & running

The graphics bundle builds like any ChimeraX bundle (`make` in
`src/bundles/graphics/`, driven by `bundle_info.xml`, which lists the
`_vulkan` C++ sources). Dependencies for the Vulkan extension:

- **Vulkan SDK** (loader headers/libs; `${VULKAN_SDK}` on Linux/Windows,
  Homebrew `vulkan-loader` + `shaderc` on macOS).
- **shaderc** (`shaderc_combined`) for runtime GLSL→SPIR-V.
- **MoltenVK** on macOS (bundled in `src/lib/` with the ICD json).
- vk-bootstrap and VMA are vendored in `_vulkan/` — never hand-edit them;
  update by replacing the files wholesale.

Select the backend at launch:

```sh
CHIMERAX_GRAPHICS_BACKEND=vulkan ./ChimeraX.app/...   # or metal; unset = OpenGL
```

## Diagnostics you already have

- **`VK_TRACE(...)`** (`vulkan_internal.h`): printf-tracing compiled into
  debug builds (`NDEBUG` unset). Writes to stderr *and* `vulkan_trace.log`
  in the working directory, flushing per line — so the last line before a
  hard crash is trustworthy. Most lifecycle functions already trace; add
  traces freely while debugging, they cost nothing in release.
- **Validation layers**: enabled automatically in debug builds on
  Linux/Windows (`build_vulkan_instance`; never on macOS — MoltenVK doesn't
  ship them). Validation output goes to stderr via the vk-bootstrap default
  messenger. *Always* run validation after touching synchronization,
  layouts, or render passes; also consider
  `VK_LAYER_KHRONOS_validation`'s sync-validation feature
  (`VK_LAYER_ENABLES`) for hazard hunting.
- **`CHIMERAX_VK_DRAW_DEBUG=1`**: per-draw prints for impostor cylinder
  draws (buffer shapes, instance counts, capability masks) from both
  drawing.py and vulkan.py.
- **Python-side failures** print tracebacks to stderr (`draw_drawing`
  catches per-draw exceptions so one bad drawing doesn't kill the app —
  check the terminal, not just the ChimeraX log).

## Common failure patterns

**A frame aborts with `AttributeError: 'VulkanRender' object has no
attribute ...`** — a code path called a protocol method the backend lacks.
Find the OpenGL implementation of that method, decide whether Vulkan needs a
real implementation or a documented no-op, and add it. Grep `opengl.py` for
the method; the call site usually tells you what behavior is load-bearing.

**Steady memory growth while rendering** — almost always an object created
per frame and not routed through the deferred-delete machinery. Checklist:
every implicit render-pass end goes through `end_active_render_pass`;
every `create_buffer` has a matching `destroy_buffer` (deferred) or lives in
a cache with eviction; per-frame UBOs (lighting, fullscreen-quad uniforms,
RT dispatch) are destroyed right after use. `vulkan_trace.log` plus a
counter in `flush_pending_deletes` localizes what's accumulating.

**SIGBUS / EXC_BAD_ACCESS inside `flush_pending_deletes` (often right
after a window resize)** — a deferred-deleted object outlived something it
references. The canonical case: a queued per-pass `VkFramebuffer` whose
attachment views were destroyed directly by swapchain/OIT/offscreen
teardown before the queue drained. Every such teardown must follow
idle-wait → `flush_all_pending_deletes` → destroy; check that any new
teardown path you added does. (A double-queued handle gives the same
signature — the first destroy frees the wrapper, the second reads freed
memory.)

**Garbage/flicker that changes with timing** — a GPU/CPU hazard. Usual
suspects: writing a buffer in place instead of `update_buffer`'s orphaning;
an image written by compute and sampled by fragment without
`FRAGMENT_SHADER` in the barrier's destination scope; a stale
`current_layout` after adding a pass whose `finalLayout` changed an image.
Sync-validation finds these mechanically.

**Black/garbage output after adding a pipeline-affecting state** — the new
state isn't in the pipeline cache key (or not in `_draw_single`'s staleness
check), so an old pipeline gets reused. See
[03-vulkan-backend.md](03-vulkan-backend.md#pipelines).

**Validation: "render pass is not compatible with pipeline"** — you drew
scene geometry into a pass whose attachment formats differ from the main
pass. Either match formats (the outline mask's format-10 trick) or create
per-target pipeline variants.

**Wrong colors only on macOS, only vs OpenGL** — color-space tagging on the
CAMetalLayer; see [02-conventions.md](02-conventions.md#color-space-macos).
Verify the swapchain actually negotiated `PASS_THROUGH_EXT` by logging the
chosen surface format in `create_swapchain`.

**Selection/picking misbehaves with impostors** — picking is geometric and
renderer-independent, but it must understand the impostor *drawing layouts*
(one placement per bond, midpoint translation, color in the 4th matrix
column). `_bonds_planes_pick` in `atomic/src/structure.py` shows the
required branching.

**Shaders fail to compile at runtime** — shaderc errors surface as Python
`RuntimeError`s with the compiler message, and post-process pipeline
failures are echoed to stderr and `vulkan_trace.log`. Remember
`common.glsl` is textually inlined by Python (`#include` is not handled by
shaderc here), and specialization-constant IDs must match
`_ensure_shader_pipeline`'s `constants[]` indices.

## Verifying changes (manual QA checklist)

There is no automated rendering test suite; verify visually against OpenGL
(run two ChimeraX instances side by side, same session):

1. Open a structure with sticks + a molecular surface; orbit, zoom, resize
   the window (exercises swapchain recreation).
2. `lighting soft` (multishadow), `lighting shadows true` (single shadow) —
   compare edge softness and direction with GL.
3. Transparent surface (`transparency 50`) — OIT compositing, then
   "save image" with transparency (offscreen capture path).
4. Drag-select with right mouse — rubber band visible, selection works,
   green outlines hug only the selected atoms.
5. `graphics silhouettes true` — edges match GL's thickness/placement.
6. Clip (`clip front`), volume data if available (plane + raycast modes,
   maximum intensity projection).
7. `lighting rtao 8` and a trajectory (`coordset` playback) — AO follows
   the moving atoms; path tracer converges without banding.
8. Switch tabs/windows, minimize/restore, quit — no validation errors at
   teardown in a debug build.

For sync/lifetime changes, additionally run a debug build under validation
for the whole checklist; "no new validation messages" is the bar.

## Known gaps (Vulkan), roughly by priority

1. **Pipeline ↔ render-target compatibility during image capture**: scene
   pipelines are baked against the main pass (swapchain format) but capture
   draws into an RGBA8 offscreen pass — works on tested drivers, formally
   undefined. Proper fix: per-target pipeline variants keyed in the
   pipeline cache, or `VK_KHR_dynamic_rendering`.
2. **Crossfade / motion blur**: needs `frame_buffer_image` to read the
   swapchain (currently returns zeros for the front buffer); the scene
   crossfades to black.
3. **Outlines & silhouettes during image capture**: both share the
   swapchain-sized scene depth and skip themselves when an offscreen
   target of a different size is active.
4. **GTAO**: Metal-only (the shader is `gtao.metal`; the pass is gated to
   renderers with `_metal`). A GLSL port + the existing color-pass plumbing
   would enable it.
5. **Stereo / 360 cameras**: `set_stereo_buffer` is a no-op; the stereo
   camera classes also touch OpenGL framebuffer objects directly.
6. **`offscreen` / `blend` subsystems**: stubs (OpenGL uses them for
   16-bit compositing workarounds that may not be needed on Vulkan).
7. **Zero-copy buffer lifetime**: numpy views over persistently-mapped
   buffers don't keep the allocation alive; the producing code (map bundle)
   owns that contract. A generation/epoch mechanism would make it safe.
8. **Per-instance state**: `set_cull_mode` (C) records nothing —
   culling is a pipeline-key flag on the Python side only;
   `set_depth_range`, `depth_invert`, `enable_xor` are no-ops.
9. **`Texture` class swap is process-global**: `VulkanRender.__init__`
   points `chimerax.graphics.Texture` at `VulkanTexture`; code that did
   `from chimerax.graphics import Texture` at import time keeps whatever
   was bound then. Fine for a single renderer per process; fragile for
   runtime backend switching.
