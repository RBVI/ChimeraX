# Vulkan renderer parity status

These changes have been reviewed in source. They have **not** been built,
run through the automated tests, or validated on a headset/display. The
repository's `AGENTS.md` forbids compilation, build, and test commands during
agent work. This document is an acceptance checklist, not a claim that full
OpenGL parity has been demonstrated.

## Implemented in this change

- Offscreen depth allocations permit sampling. Metal RTAO and path-tracer
  denoising use a separate depth copy in a sampling layout; exported Metal
  textures are invalidated when the underlying Vulkan image/view changes.
- Vulkan cameras and high-precision scene targets are selected before shadow
  passes begin a submission. Camera targets are submitted before presentation.
- Offscreen attachments are retired after the recording frame's fence, after
  temporary framebuffers that reference their views have been destroyed.
- Camera caches release resources before backend replacement. The new renderer
  reapplies the camera's special shader modes.
- Sphere and cylinder impostors use the same stereo-360 eye shift as meshes,
  including the scene depth prepass.
- Half-float framebuffer readback retains float values for texture reads and
  converts to the image API's 8-bit RGBA format when requested.
- Dome, mono-360, and stereo-360 projection samples GPU cube-face images,
  avoiding CPU readback and preserving high-precision color through composition.
- OpenXR negotiates Vulkan runtime extensions and the runtime-selected GPU
  before creating the device. Explicit ctypes layouts preserve 64-bit handles.
  Each eye is rendered to an owned target, copied into the acquired sRGB
  swapchain image, restored to the runtime-required layout, and released.
- SteamVR negotiates its Vulkan extensions and GPU, and submits actual Vulkan
  image/device/queue metadata. Both VR paths have native desktop/room-camera
  framebuffers and release sessions before device teardown.
- Looking Glass renders individual Vulkan views into a quilt and uses a port
  of the existing HoloPlay calibration shader. Its window and camera resources
  are recreated on backend changes.

## Manual acceptance checks

Compare each result against OpenGL using the same scene, lighting, resolution,
camera, and display. Use a scene containing mesh geometry, sphere and cylinder
impostors, transparent surfaces, volumes, clipping, labels, and highlights.

| Area | Checks |
| --- | --- |
| Offscreen | Save opaque/transparent PNGs at window and non-window sizes; repeat at 8/16-bit framebuffer precision and with supersampling. Verify orientation, alpha, colors, and subsequent interactive frames. |
| Depth effects | Enable silhouettes and AO, resize repeatedly, alternate screen rendering with captures, and change precision. Check Vulkan validation for sampled-image usage/layout errors. |
| Cameras | Exercise mono, orthographic, side-by-side, top-bottom, dome, mono-360, and both stereo-360 layouts. Change cube size and framebuffer precision; switch Vulkan→OpenGL→Vulkan with each camera active. |
| Stereo impostors | Compare eye disparity for atoms/bonds and equivalent mesh geometry near the viewer, at cube seams, and near the vertical axis. Check depth-tested intersections and clipping. |
| OpenXR | Use a real runtime to check required-extension negotiation, GPU selection, both eyes, gamma, controller poses, room-camera display, mirror, stop/restart, backend switching, and rendering-error cleanup. |
| SteamVR | Check both eye submissions, image orientation, gamma, controllers, mirror/room camera, runtime restart, resizing, and device teardown. |
| Looking Glass | Compare a saved quilt and the calibrated physical display against OpenGL. Verify row order, view order, RGB subpixels, unused quilt borders, and backend switching with the display window open. |

Regression tests are in `graphics/tests/test_vulkan_camera_targets.py`,
`graphics/tests/test_multiview_pathtrace.py`, `vive/tests/test_vulkan_runtime.py`,
and `looking_glass/tests/test_vulkan_quilt.py`.

## Remaining parity limits

- Hardware shutter-glasses stereo still requires the OpenGL stereo context.
  Vulkan side-by-side output is not a substitute for synchronized sequential
  stereo; that needs a platform/display-specific presentation implementation.
- Vulkan presentation still uses FIFO. VR desktop mirroring can consequently
  be limited by desktop refresh, and the existing VR setup may disable mirroring
  when disabling vsync is unavailable.
- Multi-view cameras currently use raster rendering when path tracing or RTAO
  is enabled. Independent per-view progressive histories and dispatch resources
  are still needed before enabling those effects for stereo/cube views.
- OpenXR currently accepts an RGBA8 sRGB swapchain. A runtime offering only
  other formats needs an additional format-conversion path.
- Offscreen image capture uses the existing window-backed Vulkan device; a
  headless device/surface initialization path is not implemented here.

Full feature parity remains open until these limits are addressed and the
hardware acceptance checks pass.
