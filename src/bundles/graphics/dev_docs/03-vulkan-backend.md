# 03 — The Vulkan backend

The Vulkan backend is split across two layers:

- **`src/vulkan.py`** — `VulkanRender` implements the renderer protocol,
  owns Python-side caches (pipelines, per-drawing buffers), packs uniform
  data with numpy, and decides *what* to draw.
- **`_vulkan/*.cpp`** — the `_vulkan` C extension records command buffers,
  owns every Vulkan object, and decides *how* to draw. Python refers to C
  objects by opaque pointer-sized handles (`Py_ssize_t` of a C++ pointer).

If you're new to Vulkan itself: the model is *explicit*. Nothing happens
implicitly — you record commands into a command buffer, submit it to a
queue, and synchronize manually with fences (GPU→CPU) and semaphores
(GPU→GPU). Most of the complexity documented here exists to hide that from
the renderer-agnostic layers above.

Start by reading `_vulkan/vulkan_internal.h` top to bottom. Every shared
struct is defined and commented there; this chapter explains how they fit
together.

## Context creation

`vulkan.VulkanContext.__init__` (Python) →
`vulkan_create_context` (`vulkan_context.cpp`):

1. **Instance** via vk-bootstrap. Validation layers + debug messenger are
   enabled only in debug builds (`!NDEBUG`) and never on macOS (MoltenVK
   doesn't ship them). On macOS the `VK_EXT_swapchain_colorspace` extension
   is enabled when available (color matching — see 02-conventions).
2. **Surface**: on macOS, Python pre-creates it through
   `_metal_rt.create_surface` (needs Obj-C to attach a `CAMetalLayer`);
   on Windows the C side creates a Win32 surface itself (and owns it,
   `ctx->owns_surface`); on Linux a pre-created surface is required.
3. **Physical/logical device** via vk-bootstrap, preferring discrete GPUs.
   Ray-tracing extensions (`VK_KHR_acceleration_structure`,
   `VK_KHR_ray_query`) are enabled *if present*; `raytracing_supported` is
   true only when all the function pointers also resolve.
4. **VMA allocator**, initial swapchain, command pool, per-frame objects,
   descriptor infrastructure, dummy resources.

Teardown (`vulkan_destroy_context`) waits for device idle, drains the
deferred-delete queues, and destroys everything in reverse order. If you add
a context-owned object, add its destruction there — `vkDestroyDevice` with
live child objects is a validation error and a leak.

## Frame lifecycle

Two frames can be "in flight" (`MAX_FRAMES_IN_FLIGHT = 2`): the CPU records
frame N+1 while the GPU executes frame N. Each frame slot (`FrameData`)
owns a command buffer, a spare command buffer for post-ray-tracing work, an
`image_available` semaphore, and an `in_flight_fence`.

```
begin_frame:
  wait frame slot's fence            (GPU done with this slot's previous use)
  flush this slot's deferred deletes (now safe to vkDestroy*)
  acquire swapchain image            (signals image_available)
  reset descriptor pool, command buffers; begin recording
  set image_acquired_unconsumed = true

…passes record draws…

[optional] flush_gpu:                (used before Metal-RT interop dispatch)
  end active render pass, end + submit the command buffer
  if first submit of the frame: wait image_available  ← consumed here
  swap to the post-RT command buffer and resume recording

end_frame:
  end active render pass; end command buffer
  submit, waiting image_available iff still unconsumed,
          signaling the per-IMAGE render_finished semaphore,
          fenced with the frame slot's fence
  present (waits the per-image semaphore)
  advance current_frame
```

Things that look odd but are load-bearing:

- **`image_acquired_unconsumed`**: whichever submit happens *first* in a
  frame must wait the acquire semaphore (a binary semaphore can only be
  waited once). `flush_gpu` may or may not run, so the flag tracks it.
- **Per-image `render_finished` semaphores** (`render_finished_per_image`):
  present holds its wait semaphore until the image is re-acquired, so the
  semaphore must be tied to the *image*, not the frame slot.
- **Swapchain recreation** happens on `OUT_OF_DATE`/`SUBOPTIMAL` results and
  on resize. If recreation fails, the swapchain is left null and
  `begin_frame` retries creating it rather than acquiring from a null
  handle.
- Blocking waits (`vkWaitForFences`, acquire, present) release the GIL.

Offscreen capture frames (image save) skip acquire/present entirely; the
fence still keeps slot bookkeeping consistent. See
[offscreen capture](#offscreen-capture).

## Render-pass model

Unlike OpenGL's "just draw", every Vulkan draw must be inside a
`VkRenderPass` instance bound to a `VkFramebuffer`. The backend creates
**fresh render pass + framebuffer objects per `begin_*_pass` call**, because
the load/store ops (clear vs. preserve) and the target views vary call to
call. Pass types:

| C entry point | Targets | Notes |
|---|---|---|
| `vulkan_begin_render_pass` | swapchain image + scene depth | `load_color`/`load_depth` flags; clears otherwise. Demotes a load on a never-rendered swapchain image to a clear (its contents are undefined). During capture, targets the offscreen framebuffer instead, honoring the load flags. |
| `vulkan_begin_depth_pass` | a shadow-map depth texture | uses the shared `depth_only_render_pass`; ends with the texture in `SHADER_READ_ONLY` for sampling |
| `vulkan_begin_color_pass` | an arbitrary color texture | post-processing targets (RTAO blur, PT denoise) |
| `vulkan_begin_oit_pass` | accum + reveal + scene depth | transparency accumulation (vulkan_oit.cpp) |
| `vulkan_begin_mask_pass` | swapchain-format mask texture + **live scene depth** | selection-outline mask; formats deliberately match the main pass so scene pipelines are compatible |

### The one invariant: `end_active_render_pass`

Per-pass `VkRenderPass`/`VkFramebuffer` objects are *owned* and must be
destroyed — but only after the GPU finishes the frame. `end_render_pass`
(the explicit path) and `end_active_render_pass(ctx)` (the helper every
implicit "end whatever is active" path calls) both push owned objects onto
the **deferred-delete queue**. If you add any code that ends a pass, call
the helper; a bare `vkCmdEndRenderPass` leaks one pass + framebuffer *per
frame* and you will not notice until memory climbs.

### Render-pass compatibility

A `VkPipeline` is created *against* a specific render pass and may only be
used in **compatible** ones (same attachment formats/counts/sample counts;
load/store ops don't matter). Scene pipelines are built against the main
pass (swapchain format + D32). This is why:

- the outline mask texture uses **format code 10** (= swapchain format), and
- the offscreen capture pass currently relies on driver tolerance
  (RGBA8 vs. swapchain BGRA8 — a known gap, see 06-debugging).

If you add a pass that scene pipelines draw into, match the main pass's
attachment formats or budget for per-target pipeline variants.

## Pipelines

`vulkan_create_pipeline` compiles GLSL→SPIR-V at runtime with **shaderc**
(`#include "common.glsl"` is inlined by Python beforehand) and bakes in:

- shader pair + ~20 boolean **specialization constants** (the capability
  bits — see 01-architecture),
- topology (triangles/lines/points), cull mode (front face = **CLOCKWISE**),
- depth test/write + compare op, color-write mask,
- blend state (standard alpha, MAX for maximum-intensity projection,
  premultiplied variant, or the dual-attachment OIT blend),
- the target render pass (main / depth-only / OIT).

Python caches pipelines in `VulkanRender._pipeline_cache` keyed by
`(shader_name, constants, depth_only, depth_prepass, face_culling,
blend_op_max, blend_premult, depth_write, depth_test, topology, oit_mode)`.
**If you make any new state affect pipeline creation, it must join this
key**, and `_draw_single`'s staleness check must compare it — otherwise
you'll get a stale pipeline for the new state.

State-toggle protocol methods (`enable_depth_test`, `blend_max`,
`write_depth`, `enable_backface_culling`, …) therefore just set Python
flags; the next draw picks the right pipeline. `set_depth_stencil_state` on
the C side is a no-op by design.

Post-processing pipelines (fullscreen quads) have a separate per-instance
cache, `_postprocess_pipelines`. Per-*instance* matters: pipeline handles
point into one `VkDevice`, and a class-level cache would hand stale handles
to a renderer created after a context teardown.

## Buffers

`vulkan_buffer.cpp`. All allocations go through VMA. The wrapper is
`BufferWithAllocator` (handle exposed to Python).

Three lifetime rules:

1. **Destroying**: `destroy_buffer` never frees immediately. It queues the
   wrapper on `pending_deletes[deferred_delete_slot(ctx)]`; the queue drains
   in `begin_frame` after the slot's fence wait, when the GPU provably can't
   reference it. `deferred_delete_slot` picks the in-frame slot while
   recording and the previous slot between frames — see the comment block in
   `vulkan_internal.h`.

   **Corollary invariant**: queued per-pass framebuffers hold image *views*
   (swapchain, scene depth, OIT accum/reveal, capture targets) as
   attachments. Any code path that destroys those views after a
   `vkDeviceWaitIdle` (swapchain recreation, OIT resize, offscreen
   teardown) must call `flush_all_pending_deletes(ctx)` **first** —
   otherwise the framebuffers' deferred destruction runs one or two frames
   after their attachments are gone, which is a use-after-free inside the
   driver (observed as a SIGBUS in MoltenVK on window resize). Idle-wait →
   flush → destroy, in that order, always.
2. **Updating**: `update_buffer` *orphans*. With two frames in flight the
   GPU may still be reading the old contents, so the C side allocates a
   fresh `VkBuffer` inside the same wrapper and defers the old one. Future
   binds read `wrapper.base.buffer` at record time, so they pick up the new
   buffer automatically. (Persistently-mapped buffers are exempt — Python
   holds their raw pointer — and required to be `HOST_COHERENT`.)
3. **Host visibility**: every host write is followed by
   `vmaFlushAllocation` (no-op on coherent memory, required on
   non-coherent); readbacks `vmaInvalidateAllocation` first.

**Zero-copy path**: `create_mapped_buffer` returns (handle, raw pointer) so
C-level producers (the flying-edges isosurface code in the map bundle) write
vertex data straight into GPU-visible memory; drawings carry these in
`drawing._vk_buffers` and `_draw_single` binds them directly. The numpy
views wrapping the raw pointer do **not** keep the buffer alive — freeing
the buffer while views exist is a real use-after-free hazard; the map bundle
must manage that lifetime.

Python-side per-drawing caching: `VulkanRender._drawing_buffers` maps
`id(drawing)` → a dict of GPU buffers + memoized arrays. Uploads
short-circuit when the source array identity is unchanged. Entries are
evicted (and buffers freed) by a `weakref.finalize` on the drawing —
necessary both to avoid leaks and because `id()` values get recycled.

## Textures

`vulkan_texture.cpp`. `TextureWithDevice` wraps image + view + sampler.
Uploads go through a staging buffer with a one-shot command buffer
(transition → copy → transition back to `SHADER_READ_ONLY`), and validate
the data size against the texture extent. Format codes and sampler rules are
in [02-conventions.md](02-conventions.md#texture-format-codes). Destruction
is deferred like buffers.

Python's `VulkanTexture` class (vulkan.py) adapts the ChimeraX `Texture`
interface (`bind_texture(tex_unit)` → `set_fragment_texture(index)`), and
`__init__.py` swaps the module-level `Texture` class so backend-agnostic
code constructing textures gets the right one.

Layout tracking: `VulkanTexture.current_layout` (C struct field) records
where the image *will be* after the recorded commands run; barriers use it
as `oldLayout`. If you add a pass that changes an image's layout via a
render pass `finalLayout`, update `current_layout` to that final layout (see
`begin_depth_pass` for the pattern).

## Offscreen capture

"Save image" pushes a `VulkanFramebuffer` (Python) →
`vulkan_create_offscreen_framebuffer` + `set_active_offscreen` (C). While an
offscreen target is active:

- `begin_frame` skips swapchain acquire; `end_frame` skips present and
  *waits the fence immediately* so `vulkan_read_framebuffer` can copy the
  color image out through a staging buffer.
- `begin_render_pass` redirects to the offscreen target, building per-call
  passes that honor `load_color`/`load_depth` (a capture frame has multiple
  passes — opaque, OIT resume, overlays — and clearing between them would
  wipe the scene).
- `render_size()` returns the capture size, so projection/viewport code
  follows automatically.

The OIT pass attaches whatever depth buffer belongs to the *current* target
(offscreen depth during capture, swapchain depth otherwise) — it builds its
framebuffer per-begin for exactly this reason.

## Where to add things — quick recipes

**A new uniform**: add to the `Uniforms` block in `common.glsl` *and* to
`VulkanRender._build_uniforms` in the same position (it's packed by
concatenation, byte-order is the contract). Mind std140: vec3s pad to 16
bytes.

**A new texture binding**: bump `MAX_TEXTURE_BINDINGS`
(`vulkan_internal.h`), add the GLSL declaration, add a dummy in
`vulkan_dummies.cpp` (`dummy_image_info_for_binding`), bind via
`set_fragment_texture(index=N)`.

**A new post-process pass**: write `shaders/foo.{vert,frag}` (copy
`pt_denoise.*` for the conventions: UBO at set 0 binding 0, textures at
set 1), render with `begin_color_pass` + `draw_fullscreen_quad(texture,
"foo", ...)`, then `end_render_pass`. Uniforms go in as a packed numpy
array.

**A new C entry point**: implement `vulkan_foo(PyObject*, args, kwds)` in
the matching `.cpp`, declare it in `vulkan_api.h`, register it in
`module.cpp`'s method table. Parse a context with `get_vulkan_context` or
`"n"` format units for handles. Set a Python exception and return `NULL` on
failure — never fail silently.
