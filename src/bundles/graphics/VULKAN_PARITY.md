# Vulkan renderer parity status

The source review found gaps in stereo presentation, headless rendering,
OpenXR format negotiation, GPU volume blending, and low-level renderer
methods. The paths listed below still need runtime and hardware acceptance
checks.

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
| Presentation | Toggle `graphics rate waitForVsync` on each platform. Confirm that the active present mode changes when supported, and check desktop mirrors on devices that only offer FIFO. |

Regression tests are in `graphics/tests/test_vulkan_camera_targets.py`,
`graphics/tests/test_multiview_pathtrace.py`, `vive/tests/test_vulkan_runtime.py`,
and `looking_glass/tests/test_vulkan_quilt.py`.

## OpenGL parity gaps

- Hardware shutter-glasses stereo still requires the OpenGL stereo context.
  Synchronized sequential stereo needs platform-specific presentation support.
- OpenGL has an EGL headless context. Vulkan can capture offscreen images
  using the window-backed device, but has no headless initialization path.
- OpenXR with Vulkan requires an RGBA8 sRGB swapchain. The OpenGL path
  negotiates among several formats.
- Volume image blending runs on the CPU under Vulkan, including when
  `blendOnGpu` is requested. The OpenGL path supports GPU blending.
- Some low-level renderer methods, including depth-range changes and direct
  indexed drawing, remain stubs. The main scene path issues native draws
  through `VulkanRender.draw_drawing`.

## Other Vulkan limits

- Disabling vsync requests Mailbox, then Immediate. On a FIFO-only surface,
  desktop mirroring can still be limited by the refresh rate, and VR setup
  may disable the mirror.
- Multi-view cameras fall back to raster rendering when path tracing or RTAO
  is enabled. These effects need independent per-view histories and dispatch
  resources before they can run for stereo or cube views.
