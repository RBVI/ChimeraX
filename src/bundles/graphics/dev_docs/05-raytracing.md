# 05 — Ray tracing (RTAO and the path tracer)

Two GPU ray-tracing features sit on top of the raster renderer:

- **RTAO** (`lighting rtao N` / `lighting.rtao`): ray-traced ambient
  occlusion. Keeps the normal raster image; a compute kernel shoots N
  short AO rays per pixel from the depth buffer and darkens the ambient
  term.
- **Path tracer** (`lighting.pathtrace`): replaces the raster output
  entirely with a progressive Monte-Carlo render (direct + indirect light,
  soft shadows), accumulated over frames while the camera is still.

Both follow the same shape: build an acceleration structure over the scene's
primitives, dispatch a compute shader that traverses it with **ray queries**,
temporally accumulate, composite. There is no ray-tracing *pipeline* (no
shader binding tables): a single compute shader uses
`VK_KHR_ray_query` inline, with analytic sphere/cylinder intersection — much
simpler, and it mirrors how the Metal version works.

## Backends

`VulkanRender._rt_backend` resolves to either:

- the native Vulkan path (`_vulkan`, `vulkan_raytracing.cpp`,
  `shaders/rtao.comp` / `pathtracer.comp`) when the device exposes
  `VK_KHR_acceleration_structure` + `VK_KHR_ray_query`, or
- the **Metal interop** path on macOS (`_metal_rt`): MoltenVK exposes the
  underlying `MTLDevice` via `VK_EXT_metal_objects`; `_metal_rt` runs Metal
  compute kernels (`rtao.metal`, `pathtracer.metal`) on textures shared with
  the Vulkan swapchain world. GPU↔GPU ordering uses a timeline semaphore
  exported as an `MTLSharedEvent`: `flush_gpu` submits pending Vulkan work
  signaling value V, the Metal dispatch waits V and signals V+1, and
  `end_frame`'s submit waits V+1 (`rt_sync_pending` in the context).

The Python entry points (`build_*_acceleration_structure`, `dispatch_rtao`,
`dispatch_pathtracer` in vulkan.py) hide which backend is active.

The `_metal_rt` build functions validate every geometry buffer's byte
length against its count×stride contract and nil-check the exported
device/queue before use — a shape or context mismatch raises a Python
`ValueError`/`RuntimeError` naming the culprit instead of segfaulting in
`newBufferWithBytes:`. Keep that invariant when adding geometry kinds.
Destroying an AS must release *all four* bridge-retained handles (AS,
sphere, cylinder, triangle buffers) — `_destroy_rt_accel_struct` in
vulkan.py does this per backend.

## Acceleration structures (`vulkan_raytracing.cpp`)

All primitives are **AABB geometries** (procedural), not triangles-as-
triangles — the compute shader does its own analytic intersection inside
each candidate AABB. One BLAS holds up to three geometry slots with fixed
IDs the shaders rely on:

```
geometry 0: spheres    (AABBs from center±radius, slightly padded)
geometry 1: cylinders  (AABBs from endpoint boxes)
geometry 2: triangles  (AABBs per triangle — molecular surfaces)
```

Geometry IDs are positional. A scene with triangles but no cylinders still
declares slot 1 with `primitiveCount = 0` — and its AABB data address must
still be valid (a 1-entry dummy buffer), because
`vkGetBufferDeviceAddress(VK_NULL_HANDLE)` is invalid usage.

A 1-instance TLAS wraps the BLAS (identity transform) — ray queries require
a TLAS. Builds are one-shot submissions with `vkQueueWaitIdle` between BLAS
and TLAS (slow but simple; rebuilds happen only when geometry changes).
Scratch buffers are allocated with
`minAccelerationStructureScratchOffsetAlignment` (queried at context
creation) via `vmaCreateBufferWithAlignment`.

Alongside the AS, per-primitive data buffers (sphere xyz+r, cylinder
endpoints+radius, triangle vertices) are uploaded as SSBOs for the shader's
analytic intersection and shading; colors are packed `uchar4` SSBOs
uploaded by the passes.

**Rebuild policy** (render_passes.py): the passes compare the *contents* of
the geometry arrays (`np.array_equal`, memcmp speed), not just the counts —
trajectory playback moves every atom without changing counts, and tracing
against a stale BVH while accumulating produces convincing-looking wrong
images. Any rebuild also resets the accumulation counter.

## RTAO

`RTAOPass.execute` each frame:

1. Collect visible atom spheres; (re)build the sphere AS if changed.
2. Ensure textures: `rtao_ao` (RGBA8 storage), `rtao_accum_0/1`
   (R32F ping-pong pair).
3. Reset accumulation if the camera/projection moved.
4. `dispatch_rtao`: uploads a per-frame UBO (inverse view/projection, ray
   count, radius, intensity, frame index — std140 layout matched by hand in
   vulkan.py), binds the per-frame-slot descriptor set, transitions the
   scene depth to read-only and the outputs to `GENERAL`, dispatches 8×8
   tiles, transitions outputs to sampleable (the barrier's destination
   scope covers **fragment shaders too** — the composite is raster), and
   restores the depth layout.
5. Optional bilateral blur (`supports_rtao_blur` — currently Metal only),
   then `draw_rtao_composite`: a fullscreen quad multiplying AO into the
   scene color in the resumed main pass.

The kernel (`rtao.comp`, set layout in `create_rtao_descriptor_layout`:
UBO, TLAS, sphere SSBO, depth sampler, AO storage image, previous
accumulation sampler, accumulation storage image) reconstructs the
eye-space position from depth, shoots cosine-weighted hemisphere rays via
ray query, and blends with the previous frame's accumulation
(`1/accum_count` exponential-ish average across the R32F ping-pong).

## Path tracer

`PathTracePass.execute` mirrors RTAO but goes further:

- Collects spheres **and** cylinders **and** surface triangles
  (`build_scene_acceleration_structure`), plus per-primitive color SSBOs.
- Output/accumulation textures are **RGBA16F** (format code 9). This is
  load-bearing: a running average's per-frame delta drops below 1/255 after
  a few dozen samples, so an 8-bit accumulation target freezes convergence
  with visible banding. RGBA16F is the float format with *mandatory* Vulkan
  support for both storage images and linear sampling.
- Accumulation resets on camera, projection, lighting, *or geometry*
  change; otherwise `accum_count` grows toward `lighting.pt_max_accum` and
  the pass keeps requesting redraws until converged.
- `pathtracer.comp` (11 bindings — see `create_pt_descriptor_layout`):
  primary ray from the camera through the pixel, analytic hit against
  spheres/cylinders/triangles, direct light with any-hit shadow rays,
  cosine-weighted bounces with Russian roulette (guarded against
  zero-throughput divide — NaNs blended into the accumulation never leave),
  background on miss.
- Composite: optional cross-bilateral denoise (`pt_denoise.frag`, edge-
  stopped by the *raster* depth buffer, which aligns since both reflect the
  same atoms) into a texture, then `pt_blit` replaces the framebuffer
  contents.

## Synchronization notes specific to RT

- Per-frame-slot descriptor sets (`s_rtao.sets[frame]`, `s_pt.sets[frame]`)
  are rewritten each dispatch; the `begin_frame` fence wait makes that safe
  — but only for **one dispatch per frame slot**. Multi-view/stereo would
  need per-dispatch sets.
- Compute→raster visibility: any image a compute kernel writes and a
  fragment shader later samples needs the barrier's `dstStageMask` to
  include `FRAGMENT_SHADER` (see `transition_for_sampled`). A compute-only
  destination scope is the classic missing-barrier bug here.
- Per-dispatch UBOs ride the deferred-delete queue
  (`deferred_delete_slot`), like every other transient buffer.
- AS destruction does a full `vkDeviceWaitIdle` — acceptable because it only
  happens on geometry changes.
