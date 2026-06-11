# ChimeraX Graphics — Developer Documentation

This directory documents the ChimeraX graphics bundle for engineers working
on the rendering code, with particular depth on the **Vulkan backend**. It is
written so that a junior engineer with general programming experience — but no
prior exposure to ChimeraX internals or low-level graphics APIs — can become
productive.

These are *developer* docs. User-facing API documentation lives in
`docs/devel/graphics.rst`.

## Reading order

| Chapter | What it covers | Read it when |
|---|---|---|
| [01-architecture.md](01-architecture.md) | The big picture: how a frame gets drawn, the layers from `View` down to the GPU, how the three rendering backends coexist | First. Everything else assumes it. |
| [02-conventions.md](02-conventions.md) | Matrix storage, coordinate systems, depth ranges, winding, color space, format codes | Before touching any math or shader code. Most "why does this look wrong" bugs are convention bugs. |
| [03-vulkan-backend.md](03-vulkan-backend.md) | The `_vulkan` C extension and `vulkan.py`: context, frame lifecycle, render passes, descriptors, pipelines, resource management | Before changing anything under `_vulkan/` or in `vulkan.py`. |
| [04-features.md](04-features.md) | How each visual feature works end to end: impostors, shadows, transparency, clipping, outlines, silhouettes, volume rendering, … | When working on a specific feature. |
| [05-raytracing.md](05-raytracing.md) | RTAO and the path tracer: acceleration structures, compute dispatch, temporal accumulation, the macOS Metal interop | When working on ray tracing. |
| [06-debugging.md](06-debugging.md) | Building, tracing, validation layers, common failure patterns, known gaps | When something breaks (so, often). |

## The one-paragraph summary

ChimeraX draws molecular scenes through a renderer-agnostic core: a `View`
walks a tree of `Drawing` objects and executes a **render graph** of passes
(shadows, opaque, transparent, outlines, …). Each pass talks to a **renderer**
object that implements a common protocol. There are three renderers: the
original OpenGL one (`opengl.py`), a native Metal one for macOS (`metal.py`),
and a Vulkan one (`vulkan.py`) targeting Linux/Windows (and macOS via
MoltenVK). The Metal and Vulkan renderers push actual GPU work down into C
extension modules (`_metal/`, `_vulkan/`) and compile GLSL shaders at runtime
with shaderc. The backend is selected at startup with the
`CHIMERAX_GRAPHICS_BACKEND` environment variable.

## File map

```
src/bundles/graphics/
├── src/                          Python package (chimerax.graphics)
│   ├── view.py                   View: owns camera, drawing tree, drives frames
│   ├── render_graph.py           RenderGraph: ordered pass execution + context
│   ├── render_passes.py          The passes (shadow, opaque, OIT, RTAO, …)
│   ├── renderer.py               SHADER_* capability bits, PassDescriptor
│   ├── drawing.py                Drawing tree, draw_* free functions, picking
│   ├── camera.py / camera360.py  Cameras, projection matrix construction
│   ├── opengl.py                 OpenGL renderer (the reference implementation)
│   ├── metal.py                  Metal renderer (macOS)
│   ├── vulkan.py                 Vulkan renderer  ← most of these docs
│   ├── clipping.py               Clip plane models
│   ├── crossfade.py              Cross-fade / motion blur overlays
│   └── shaders/                  Runtime-compiled GLSL (Vulkan) + MSL (Metal)
│       ├── common.glsl           Shared uniform block + lighting for Vulkan
│       ├── mesh.{vert,frag}      General triangle geometry
│       ├── impostor_sphere.*     Ray-traced sphere billboards (atoms)
│       ├── impostor_cylinder.*   Ray-traced cylinder billboards (bonds)
│       ├── raycast.*             Volume ray casting
│       ├── oit_composite.*       Transparency composite
│       ├── outline.*             Selection outline composite
│       ├── silhouette.*          Depth-edge silhouette composite
│       ├── rtao.comp / pathtracer.comp     Ray-tracing compute kernels
│       └── *.metal               Metal shading language equivalents
├── _graphics/                    Legacy C++ helpers (blending, mesh edges)
├── _metal/                       Metal C/Obj-C backend
├── _metal_rt/                    Metal ray tracing + macOS surface helper
├── _vulkan/                      Vulkan C++ backend
│   ├── vulkan_internal.h         Shared structs (VulkanContext etc.) — START HERE
│   ├── vulkan_context.cpp        Instance/device/swapchain, frame & pass lifecycle
│   ├── vulkan_buffer.cpp         Buffers, descriptor flush, draw calls
│   ├── vulkan_texture.cpp        Texture create/upload/destroy
│   ├── vulkan_pipeline.cpp       GLSL→SPIR-V compile, VkPipeline creation
│   ├── vulkan_framebuffer.cpp    Offscreen capture (image save)
│   ├── vulkan_oit.cpp            Order-independent transparency passes
│   ├── vulkan_raytracing.cpp     Acceleration structures, RTAO/PT dispatch
│   ├── vulkan_dummies.cpp        Placeholder resources for unbound descriptors
│   ├── vulkan_api.h / module.cpp Python bindings (_vulkan module)
│   └── VkBootstrap*, vk_mem_alloc.h   Vendored third-party (do not hand-edit)
└── dev_docs/                     ← you are here
```

## Glossary

- **Drawing** — a node in the scene graph: vertex/triangle arrays, colors,
  textures, and zero or more *positions* (instances). `chimerax.graphics.Drawing`.
- **Renderer / Render** — the object implementing the drawing protocol for one
  backend (`opengl.Render`, `metal.MetalRender`, `vulkan.VulkanRender`).
- **Capabilities** — a bitmask of `SHADER_*` flags (`renderer.py`) describing
  what a draw needs (lighting, shadows, instancing, …). Selects/parameterizes
  the shader.
- **Render pass (graph sense)** — one step of the per-frame render graph
  (`render_passes.py`), e.g. `OpaquePass`. Don't confuse with…
- **Render pass (Vulkan sense)** — a `VkRenderPass`: a GPU construct declaring
  attachments and load/store behavior that draws are recorded inside.
- **Impostor** — a screen-facing quad whose fragment shader ray-traces a
  perfect sphere/cylinder per pixel, replacing tessellated meshes for atoms
  and bonds. See [04-features.md](04-features.md).
- **OIT** — order-independent transparency (weighted blended, McGuire-Bavoil).
- **RTAO** — ray-traced ambient occlusion (compute shader + ray queries).
- **VMA** — Vulkan Memory Allocator, the vendored library managing GPU memory.
- **vk-bootstrap** — vendored library handling instance/device/swapchain
  creation boilerplate.
- **MoltenVK** — Vulkan implemented on top of Metal; how the Vulkan backend
  runs on macOS.

## House rules (the things that bite)

These are expanded in the chapters, but if you remember nothing else:

1. **Matrices are stored transposed.** ChimeraX numpy matrices store math
   column *i* in storage row *i*. Read
   [02-conventions.md](02-conventions.md#matrices) before writing any matrix
   code, including in shaders.
2. **Never end a Vulkan render pass by hand.** Always go through
   `end_active_render_pass()` (C side) or `end_render_pass` (Python side);
   ad-hoc `vkCmdEndRenderPass` calls leak the per-pass objects.
   See [03-vulkan-backend.md](03-vulkan-backend.md#render-pass-model).
3. **Never write to a buffer the GPU might still read.** Use
   `update_buffer` (which orphans) or create-new + `destroy_buffer` (which
   defers). See [03-vulkan-backend.md](03-vulkan-backend.md#buffers).
4. **Pipelines bake state.** Depth test/write, blending, culling, topology and
   render-pass identity are all *pipeline creation parameters* in Vulkan, not
   dynamic state. Changing them means a different pipeline cache key.
5. **The front face is CLOCKWISE.** The GL→Vulkan Y flip mirrors triangle
   winding. If you port culling code from OpenGL, flip it.
