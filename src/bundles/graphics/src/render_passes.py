"""
Concrete render pass implementations for the ChimeraX render graph.

Each pass wraps existing rendering code verbatim.  The pass objects are
thin orchestrators; the actual draw calls live in drawing.py free
functions and renderer subsystem classes (Shadow, Multishadow, etc.).
"""

from .render_graph import RenderPass


# ---------------------------------------------------------------------------
#  Shared utilities
# ---------------------------------------------------------------------------


def _shadow_bounds(drawings):
    """Bounding box of shadow-casting drawings."""
    sdrawings = [d for d in drawings if d.casts_shadows]
    from chimerax.geometry import bounds

    b = bounds.union_bounds(
        d.bounds() for d in sdrawings if not getattr(d, "skip_bounds", False)
    )
    if b is None:
        return None, 0, sdrawings
    return b.center(), b.radius(), sdrawings


def _shadow_projection(radius):
    """Orthographic projection matrix for shadow rendering."""
    from numpy import array, float64

    rinv = 1.0 / radius
    return array(
        ((rinv, 0, 0, 0), (0, rinv, 0, 0), (0, 0, -rinv, 0), (0, 0, -1, 1)), float64
    )


# ---------------------------------------------------------------------------
#  Pre-view passes (run once per frame)
# ---------------------------------------------------------------------------


class ShadowMapPass(RenderPass):
    """Compute single-shadow and multishadow depth maps."""

    name = "shadow_maps"
    _multishadow_params = None

    def enabled(self, ctx):
        lp = ctx.renderer.lighting
        return lp.shadows or lp.multishadow > 0

    def execute(self, ctx):
        r = ctx.renderer
        mp = r.material

        shadow_drawings = list(ctx.opaque_drawings)
        if mp.transparent_cast_shadows:
            shadow_drawings += ctx.transparent_drawings
        if not mp.meshes_cast_shadows:
            shadow_drawings = [d for d in shadow_drawings if d.display_style != d.Mesh]

        center, radius, sdrawings = _shadow_bounds(shadow_drawings)
        if center is None or radius == 0:
            return

        proj = _shadow_projection(radius)

        self._render_single_shadow(ctx, r, sdrawings, center, radius, proj)
        r.enable_shader_shadows(ctx.shadow_enabled)

        self._render_multishadow(ctx, r, sdrawings, center, radius, proj)
        r.enable_shader_multishadows(ctx.multishadow_enabled)

    # -- Single directional shadow -------------------------------------------

    def _render_single_shadow(self, ctx, r, sdrawings, center, radius, proj):
        lp = r.lighting
        if not lp.shadows:
            return

        size = lp.shadow_map_size
        tex = self._ensure_texture(r, ctx.resources, "shadow_depth", size)

        if r.recording_opengl:
            light_dir = (
                lambda c=ctx.camera,
                kl=lp.key_light_direction: c.position.transform_vector(kl)
            )
        else:
            light_dir = ctx.camera.position.transform_vector(lp.key_light_direction)
        lvinv, stf = r.shadow.compute_shadow_transforms(
            light_dir, center, radius, lp.shadow_depth_bias
        )
        r.shadow.set_scene_transform(stf, tex)

        from .renderer import PassDescriptor

        r.begin_pass(
            PassDescriptor(
                depth_texture=tex,
                depth_only=True,
                clear_depth=1.0,
                viewport=(0, 0, size, size),
            )
        )
        r.set_projection_matrix(proj)
        r.set_view_matrix(lvinv)

        from .drawing import draw_depth

        draw_depth(r, sdrawings, opaque_only=not r.material.transparent_cast_shadows)
        r.end_pass()

        ctx.shadow_enabled = True

    # -- Multi-directional ambient occlusion shadow --------------------------

    def _render_multishadow(self, ctx, r, sdrawings, center, radius, proj):
        lp = r.lighting
        if lp.multishadow == 0:
            return

        ms = r.multishadow
        mat = r.material

        # Parameter-change detection.
        params = (
            lp.multishadow,
            lp.multishadow_map_size,
            lp.multishadow_depth_bias,
            mat.transparent_cast_shadows,
            mat.meshes_cast_shadows,
        )
        if params != self._multishadow_params:
            ms.multishadow_update_needed = True
        if ms.multishadow_update_needed:
            ms.reset_transforms()
            ms.multishadow_update_needed = False

        directions = ms.multishadow_directions(lp.multishadow)
        if ms.transforms_valid(len(directions)):
            ctx.multishadow_enabled = True
            return

        size = lp.multishadow_map_size
        tex = self._ensure_texture(r, ctx.resources, "multishadow_depth", size)

        from .renderer import PassDescriptor

        r.begin_pass(
            PassDescriptor(
                depth_texture=tex,
                depth_only=True,
                clear_depth=1.0,
                viewport=(0, 0, size, size),
            )
        )

        from .drawing import draw_depth
        from math import ceil, sqrt

        nl = min(len(directions), ms.max_multishadows())
        d = int(ceil(sqrt(nl)))
        tile = size // d
        bias = lp.multishadow_depth_bias
        transforms = []

        for l in range(nl):
            x, y = (l % d), (l // d)
            r.set_viewport(x * tile, y * tile, tile, tile)
            lvinv, stf = ms.compute_multishadow_transforms(
                directions[l],
                center,
                radius,
                bias,
                x * tile,
                y * tile,
                tile,
                tile,
                size,
            )
            r.set_projection_matrix(proj)
            r.set_view_matrix(lvinv)
            transforms.append(stf)
            draw_depth(r, sdrawings, opaque_only=not mat.transparent_cast_shadows)

        r.end_pass()

        ms.set_scene_transforms(transforms, 2 * radius, tex)
        self._multishadow_params = params
        ctx.multishadow_enabled = True

    # -- Resource helpers ----------------------------------------------------

    @staticmethod
    def _ensure_texture(r, resources, name, size):
        """Get or create a depth texture via the resource pool."""
        if resources.needs_resize(name, size, size):
            old = resources.remove(name)
            r.destroy_depth_texture(old)
            tex = r.create_depth_texture(size, size)
            resources.put(name, tex, width=size, height=size)
        return resources.get(name)


# ---------------------------------------------------------------------------
#  Per-view passes (run once per camera view)
# ---------------------------------------------------------------------------


class OffscreenStartPass(RenderPass):
    """Push offscreen framebuffer for 16-bit color / outline compositing."""

    name = "offscreen_start"

    def enabled(self, ctx):
        return ctx.offscreen is not None

    def execute(self, ctx):
        ctx.offscreen.start(ctx.renderer)


class SilhouetteStartPass(RenderPass):
    """Push silhouette framebuffer for depth-based edge capture."""

    name = "silhouette_start"

    def enabled(self, ctx):
        sil = ctx.silhouette
        return sil is not None and sil.enabled

    def execute(self, ctx):
        ctx.silhouette.start_silhouette_drawing(ctx.renderer)


class BackgroundPass(RenderPass):
    """Clear color and depth buffers with the background."""

    name = "background"

    def execute(self, ctx):
        ctx.camera.draw_background(ctx.view_num, ctx.renderer)


class ViewSetupPass(RenderPass):
    """Set projection, view matrix, and shadow view transforms."""

    name = "view_setup"

    def execute(self, ctx):
        r = ctx.renderer
        view = ctx.view
        camera = ctx.camera
        vnum = ctx.view_num

        view._update_projection(camera, vnum)

        if r.recording_opengl:
            from . import gllist

            cp = gllist.ViewMatrixFunc(view, vnum)
        else:
            cp = camera.get_position(vnum)

        ctx.camera_position = cp
        view._view_matrix = vm = cp.inverse(is_orthonormal=True)
        r.set_view_matrix(vm)

        if ctx.shadow_enabled:
            r.shadow.set_shadow_view(cp)
        if ctx.multishadow_enabled:
            r.multishadow.set_multishadow_view(cp)

        view._start_timing()


class DepthPrepass(RenderPass):
    """Early depth pass to avoid lighting on hidden geometry."""

    name = "depth_prepass"

    def enabled(self, ctx):
        r = ctx.renderer
        gtao_on = hasattr(r, "draw_fullscreen_quad") and getattr(r.lighting, "gtao", 0) > 0
        return (ctx.multishadow_enabled or gtao_on) and len(ctx.opaque_drawings) > 0

    def execute(self, ctx):
        from .drawing import draw_depth

        draw_depth(ctx.renderer, ctx.opaque_drawings)
        ctx.renderer.allow_equal_depth(True)


class OpaquePass(RenderPass):
    """Render opaque geometry with full lighting and shadows."""

    name = "opaque"

    def enabled(self, ctx):
        return len(ctx.opaque_drawings) > 0

    def execute(self, ctx):
        r = ctx.renderer
        # Clear stale GTAO state from previous frames.  If the GTAO pass
        # ran this frame it will have already set these.
        if not hasattr(r, "_gtao_ao_texture") or r._gtao_ao_texture is None:
            from .renderer import SHADER_GTAO
            r.enable_shader_capability(SHADER_GTAO, False)

        from .drawing import draw_opaque

        draw_opaque(r, ctx.opaque_drawings)

        # Clear GTAO texture reference so it does not persist to the next
        # frame if the pass becomes disabled.
        r._gtao_ao_texture = None


class OutlineMaskPass(RenderPass):
    """Copy depth to the outline mask framebuffer for selection outlines."""

    name = "outline_mask"

    def enabled(self, ctx):
        return len(ctx.highlight_drawings) > 0

    def execute(self, ctx):
        ctx.renderer.outline.set_outline_mask()


class SilhouetteDrawPass(RenderPass):
    """Draw silhouette edges behind transparent surfaces."""

    name = "silhouette_draw"

    def enabled(self, ctx):
        sil = ctx.silhouette
        return sil is not None and sil.enabled and len(ctx.transparent_drawings) > 0

    def execute(self, ctx):
        ctx.silhouette.draw_silhouette(ctx.renderer)


def _is_volume(d):
    """A volume image drawing (full 3D raycast or plane-based)."""
    return hasattr(d, 'draw_volume_raycast') or \
        getattr(d, '_draw_planes', None) is not None


def _pt_volume_drawing(ctx):
    """The VolumeRaycastDrawing the path tracer renders as a participating
    medium, or None.

    Scans both the opaque and transparent lists (a volume image lands in
    whichever matches its color mode). The flat list contains both the Image3d
    container AND its child drawings. The actual raycast drawing only exists in
    ``projectionMode rays``; it is reachable as the container's
    ``_volume_raycast_drawing`` and is the only class with BOTH
    ``set_shader_options`` and ``_update_coloring`` (the container has the
    former via the base Drawing but not the latter; plane drawings have the
    latter but not the former)."""
    for d in list(ctx.transparent_drawings) + list(ctx.opaque_drawings):
        # Image3d container -> its rays-mode raycast child (None otherwise).
        vrd = getattr(d, '_volume_raycast_drawing', None)
        if vrd is not None:
            return vrd
        # Or the VolumeRaycastDrawing itself, if it appears directly.
        if (hasattr(d, 'set_shader_options')
                and hasattr(d, '_update_coloring')
                and hasattr(d, 'colormap_range')):
            return d
    return None


class TransparentPass(RenderPass):
    """Render transparent geometry with OIT when available."""

    name = "transparent"

    def enabled(self, ctx):
        return len(ctx.transparent_drawings) > 0

    def execute(self, ctx):
        from .drawing import draw_transparent

        r = ctx.renderer
        drawings = ctx.transparent_drawings

        # When the path tracer rendered a volume this frame as a participating
        # medium, skip its rasterized draw so it isn't composited a second
        # time over the path-traced image.
        handled = getattr(r, '_pt_handled_volume_id', None)
        r._pt_handled_volume_id = None
        if handled:
            drawings = [d for d in drawings if id(d) not in handled]
            if not drawings:
                return

        # Use OIT if the renderer supports and enables it.  Volume raycasting
        # drawings use VK_BLEND_OP_MAX which is incompatible with
        # WBOIT, so they go through the standard transparency path.
        use_oit = (hasattr(r, 'begin_oit_pass') and
                   getattr(r, 'order_independent_transparency', True))
        if use_oit:
            from .drawing import _draw_multiple, Drawing

            oit_drawings = [d for d in drawings if not _is_volume(d)]
            vol_drawings = [d for d in drawings if _is_volume(d)]

            if oit_drawings:
                r.begin_oit_pass()
                _draw_multiple(oit_drawings, r, Drawing.TRANSPARENT_DRAW_PASS)
                r.end_oit_pass()

            if vol_drawings:
                draw_transparent(r, vol_drawings)
        else:
            draw_transparent(r, drawings)


class DepthResetPass(RenderPass):
    """Finish geometry timing and restore strict depth test."""

    name = "depth_reset"

    def execute(self, ctx):
        ctx.view._finish_timing()
        if ctx.multishadow_enabled:
            ctx.renderer.allow_equal_depth(False)


class SilhouetteFinishPass(RenderPass):
    """Composite silhouette edges into the scene."""

    name = "silhouette_finish"

    def enabled(self, ctx):
        sil = ctx.silhouette
        return sil is not None and sil.enabled

    def execute(self, ctx):
        ctx.silhouette.finish_silhouette_drawing(ctx.renderer)


class HighlightOutlinePass(RenderPass):
    """Draw selection highlight outlines."""

    name = "highlight_outline"

    def enabled(self, ctx):
        # Outline rendering needs renderer support (OpenGL's mask
        # framebuffer technique). The Metal/Vulkan renderers carry a stub
        # outline object without these methods; running the pass there
        # raised AttributeError on the first selection, aborting the frame.
        outline = getattr(ctx.renderer, "outline", None)
        return (len(ctx.highlight_drawings) > 0
                and hasattr(outline, "start_rendering_outline"))

    def execute(self, ctx):
        from .drawing import draw_highlight_outline

        view = ctx.view
        draw_highlight_outline(
            ctx.renderer,
            ctx.highlight_drawings,
            color=view._highlight_color,
            pixel_width=view._highlight_width,
        )


class OnTopPass(RenderPass):
    """Render always-on-top geometry (labels, arrows, etc.)."""

    name = "on_top"

    def enabled(self, ctx):
        return len(ctx.on_top_drawings) > 0

    def execute(self, ctx):
        from .drawing import draw_on_top

        draw_on_top(ctx.renderer, ctx.on_top_drawings)


class GTAOPass(RenderPass):
    """Horizon-based screen-space ambient occlusion.

    Runs after the depth pre-pass but before the opaque pass.  Computes
    a per-pixel AO texture from the depth buffer and stores it on the
    renderer so the opaque lighting shaders can apply it to the ambient
    term only (matching multishadow behaviour).
    """

    name = "gtao"

    def enabled(self, ctx):
        # Metal-only for now: the pass draws with the gtao.metal shader
        # and drives the encoder through r._metal directly. The Vulkan
        # renderer has draw_fullscreen_quad but neither the shader nor a
        # _metal handle, so enabling GTAO there crashed mid-frame.
        r = ctx.renderer
        return (hasattr(r, "_metal")
                and hasattr(r, "draw_fullscreen_quad")
                and getattr(r.lighting, "gtao", 0) > 0)

    def execute(self, ctx):
        import numpy as np

        r = ctx.renderer
        lp = r.lighting
        w, h = r.render_size()
        if w == 0 or h == 0:
            return

        # Get or create AO texture (screen-sized, R8).
        res = ctx.resources
        if res.needs_resize("gtao_ao", w, h):
            old = res.remove("gtao_ao")
            if old is not None:
                r._metal.destroy_texture(texture=old)
            res.put("gtao_ao", r.create_color_texture(w, h), width=w, height=h)
        ao_tex = res.get("gtao_ao")

        # Get the scene depth texture from the depth pre-pass.
        depth_tex = r.get_scene_depth_texture()
        if depth_tex is None:
            return

        # Build inverse projection matrix for position reconstruction.
        pm = r.current_projection_matrix
        if pm is None:
            return
        inv_proj = np.linalg.inv(np.asarray(pm, dtype=np.float32))

        # Pack GTAO uniforms matching the shader's GTAOUniforms struct.
        radius = getattr(lp, "gtao_radius", 10.0)       # view-space Angstroms
        intensity = getattr(lp, "gtao_intensity", 1.5)
        directions = int(lp.gtao)
        steps = int(getattr(lp, "gtao_steps", 8))
        near, far = r._near_far_clip

        bias_frac = float(getattr(lp, "gtao_bias_frac", 0.02))

        uniforms = np.zeros(28, dtype=np.float32)  # 112 bytes (16-byte aligned)
        uniforms[0:16] = inv_proj.reshape(-1)  # float4x4 inv_projection
        uniforms[16] = 1.0 / w  # texel_size.x
        uniforms[17] = 1.0 / h  # texel_size.y
        uniforms[18] = radius  # radius
        uniforms[19] = intensity  # intensity
        uniforms.view(np.int32)[20] = directions
        uniforms.view(np.int32)[21] = steps
        uniforms[22] = np.float32(near)  # near
        uniforms[23] = np.float32(far)  # far
        uniforms[24] = np.float32(bias_frac)  # bias_frac

        uniforms_data = np.ascontiguousarray(uniforms)

        # Compute AO into a separate texture.
        r.begin_color_pass(ao_tex, w, h, clear=True)
        r.draw_fullscreen_quad(
            depth_tex,
            "gtao.metal",
            "gtao_vertex",
            "gtao_fragment",
            uniforms_data=uniforms_data,
        )
        r._metal.end_render_pass(context=r._ctx)

        # Restart the main render pass so the opaque pass can continue
        # with the existing depth buffer.
        r.restart_main_pass()

        # Store the AO texture on the renderer and enable the GTAO
        # shader capability so the opaque pass binds it at texture(7).
        r._gtao_ao_texture = ao_tex
        from .renderer import SHADER_GTAO
        r.enable_shader_capability(SHADER_GTAO, True)


class RTAOPass(RenderPass):
    """Ray-Traced Ambient Occlusion.

    Builds a Metal acceleration structure from all visible atom spheres,
    then dispatches a compute kernel that casts short AO rays from each
    visible surface point.  Produces true 3D occlusion — no faceting,
    no screen-space artifacts, correct burial darkening.
    """

    name = "rtao"
    _frame_counter = 0
    _accum_count = 0
    _last_atom_count = 0
    _last_sphere_data = None
    _last_view_matrix = None
    _last_proj_matrix = None
    _ping = 0  # ping-pong index: 0 or 1

    def enabled(self, ctx):
        r = ctx.renderer
        return (
            hasattr(r, "supports_ray_tracing")
            and r.supports_ray_tracing()
            and getattr(r.lighting, "rtao", 0) > 0
        )

    def execute(self, ctx):
        import numpy as np

        r = ctx.renderer
        lp = r.lighting
        w, h = r.render_size()
        if w == 0 or h == 0:
            return

        sphere_data = self._collect_spheres(ctx)
        if sphere_data is None or len(sphere_data) == 0:
            return

        # Rebuild when the geometry *content* changes, not just the count:
        # trajectory playback and model dragging keep the atom count
        # constant while moving every sphere, which previously left a
        # stale BVH and blended occlusion into a stale accumulation
        # history. The array comparison is memcmp-speed and only runs
        # while RTAO is enabled.
        if (len(sphere_data) != self._last_atom_count
                or self._last_sphere_data is None
                or not np.array_equal(sphere_data, self._last_sphere_data)):
            r.build_sphere_acceleration_structure(sphere_data)
            self._last_atom_count = len(sphere_data)
            self._last_sphere_data = np.array(sphere_data, copy=True)
            self._accum_count = 0  # scene changed, reset accumulation

        res = ctx.resources
        if res.needs_resize("rtao_ao", w, h):
            r.destroy_texture(res.remove("rtao_ao"))
            res.put("rtao_ao", r.create_color_texture(w, h), width=w, height=h)
            self._accum_count = 0
        ao_tex = res.get("rtao_ao")

        for name in ("rtao_accum_0", "rtao_accum_1"):
            if res.needs_resize(name, w, h):
                r.destroy_texture(res.remove(name))
                res.put(name, r.create_r32f_texture(w, h), width=w, height=h)
                self._accum_count = 0
        prev_tex = res.get(f"rtao_accum_{self._ping}")
        accum_tex = res.get(f"rtao_accum_{1 - self._ping}")

        pm = r.current_projection_matrix
        cp = ctx.camera_position  # camera position = inverse view matrix
        if pm is None or cp is None:
            return
        inv_proj = np.linalg.inv(np.asarray(pm, dtype=np.float32))
        inv_view = np.asarray(cp.opengl_matrix(), dtype=np.float32)

        # Reset accumulation when the camera or projection changes.
        if (
            self._last_view_matrix is None
            or not np.allclose(inv_view, self._last_view_matrix, atol=1e-6)
            or not np.allclose(inv_proj, self._last_proj_matrix, atol=1e-6)
        ):
            self._accum_count = 0
        self._last_view_matrix = inv_view.copy()
        self._last_proj_matrix = inv_proj.copy()

        num_rays = int(getattr(lp, "rtao", 8))
        ao_radius = getattr(lp, "rtao_radius", 5.0)
        intensity = getattr(lp, "rtao_intensity", 1.5)
        self._frame_counter += 1
        self._accum_count += 1

        r._ensure_rtao_pipeline()

        r.dispatch_rtao(
            ao_tex,
            prev_tex,
            accum_tex,
            inv_proj,
            inv_view,
            ao_radius=ao_radius,
            intensity=intensity,
            num_rays=num_rays,
            frame=self._frame_counter,
            accum_count=self._accum_count,
        )
        self._ping = 1 - self._ping

        # --- Optional bilateral blur pass: denoise the AO ---
        # The Vulkan backend doesn't yet have a blur fragment shader, so it
        # falls back to temporal-accumulation-only denoise (controlled by
        # renderer.supports_rtao_blur).
        composite_tex = ao_tex
        if getattr(r, "supports_rtao_blur", True):
            if res.needs_resize("rtao_blurred", w, h):
                r.destroy_texture(res.remove("rtao_blurred"))
                res.put("rtao_blurred", r.create_color_texture(w, h),
                        width=w, height=h)
            blurred_tex = res.get("rtao_blurred")

            depth_tex = r.get_scene_depth_texture()
            if depth_tex is not None:
                blur_uniforms = np.zeros(4, dtype=np.float32)
                blur_uniforms[0] = 1.0 / w   # texel_size.x
                blur_uniforms[1] = 1.0 / h   # texel_size.y
                blur_uniforms[2] = 0.005      # depth_threshold
                blur_data = np.ascontiguousarray(blur_uniforms)

                r.begin_color_pass(blurred_tex, w, h, clear=True)
                r.draw_fullscreen_quad_two_textures(
                    ao_tex, depth_tex,
                    "rtao.metal", "rtao_blur_vertex", "rtao_blur_fragment",
                    uniforms_data=blur_data)
                r._metal.end_render_pass(context=r._ctx)
                composite_tex = blurred_tex

        # --- Composite: multiply AO into the scene color buffer ---
        r.restart_main_pass()
        r.draw_rtao_composite(composite_tex)

        # Request another frame while accumulating.  Setting the drawing
        # manager's flag works because dm.clear_changes() already ran at
        # the start of this frame, so the flag persists to the next
        # check_for_drawing_change() call.
        max_accum = max(8192 // num_rays, 1)
        if self._accum_count < max_accum:
            ctx.view._drawing_manager.redraw_needed = True

    @staticmethod
    def _collect_spheres(ctx):
        """Collect all visible atom spheres as (x, y, z, radius) in world space."""
        import numpy as np

        all_spheres = []
        for drawing in ctx.opaque_drawings:
            sas = drawing.positions.shift_and_scale_array()
            if sas is None or len(sas) == 0:
                continue
            coords = sas[:, :3].copy()
            radii = sas[:, 3:4].copy()

            # shift_and_scale positions are in parent-local space.
            # The rendering code sets model_matrix = parent.scene_position
            # for instanced drawings, so we must apply that same transform
            # to get world-space coordinates for the acceleration structure.
            if drawing.parent is not None:
                scene_pos = drawing.parent.get_scene_positions(displayed_only=True)
                if not scene_pos.is_identity():
                    p = scene_pos[0]
                    rot = np.asarray(p.matrix[:3, :3], dtype=np.float64)
                    trans = np.asarray(p.matrix[:3, 3], dtype=np.float64)
                    coords = (coords @ rot.T) + trans

            all_spheres.append(np.hstack([coords, radii]))

        if not all_spheres:
            return None
        return np.vstack(all_spheres).astype(np.float32)


class PathTracePass(RenderPass):
    """Full path tracer for impostor spheres.

    Replaces rasterized rendering with a compute-kernel path tracer
    that casts primary rays, shadow rays, and bounce rays against the
    sphere acceleration structure.  Accumulates over frames for
    progressive refinement.
    """

    name = "pathtrace"
    _frame_counter = 0
    _accum_count = 0
    _last_atom_count = 0
    _last_view_matrix = None
    _last_proj_matrix = None
    _last_lighting_key = None
    _last_geom_key = None
    _last_geom_data = None
    _ping = 0
    _color_buffer = None
    _cyl_color_buffer = None
    _tri_color_buffer = None
    _pt_colormap_tex = None   # 2D Nx1 transfer-function texture for the medium
    _pt_cmap_key = None       # cache key: rebuild only when the colormap changes

    def enabled(self, ctx):
        r = ctx.renderer
        return (
            hasattr(r, "supports_ray_tracing")
            and r.supports_ray_tracing()
            and r.lighting.pathtrace
        )

    def execute(self, ctx):
        import numpy as np

        r = ctx.renderer
        lp = r.lighting
        # Cleared each frame; set below only when a volume is path traced, so
        # an early return (or pathtrace off) leaves the rasterized volume to
        # the TransparentPass.
        r._pt_handled_volume_id = None
        w, h = r.render_size()
        if w == 0 or h == 0:
            return

        sphere_data, sphere_colors, cyl_data, cyl_colors, \
            tri_data, tri_colors = self._collect_geometry(ctx)
        if sphere_data is None or len(sphere_data) == 0:
            # No atoms. If a density volume is shown, still path trace it: the
            # acceleration structure needs at least one primitive, so inject a
            # single zero-radius sphere that no ray can intersect.
            if _pt_volume_drawing(ctx) is None:
                return
            sphere_data = np.zeros((1, 4), dtype=np.float32)
            sphere_colors = np.zeros((1, 4), dtype=np.uint8)
            cyl_data = cyl_colors = tri_data = tri_colors = None

        geom_key = (len(sphere_data),
                     len(cyl_data) if cyl_data is not None else 0,
                     len(tri_data) if tri_data is not None else 0)
        # Compare content, not just counts: trajectory playback / model
        # dragging move geometry without changing counts, and tracing
        # against the stale BVH accumulates wrong images. memcmp-speed.
        geom_data = (sphere_data, cyl_data, tri_data)
        changed = (geom_key != self._last_geom_key
                   or self._last_geom_data is None)
        if not changed:
            for a, b in zip(geom_data, self._last_geom_data):
                if (a is None) != (b is None) or (
                        a is not None and not np.array_equal(a, b)):
                    changed = True
                    break
        if changed:
            has_cyls = cyl_data is not None and len(cyl_data) > 0
            has_tris = tri_data is not None and len(tri_data) > 0
            if has_cyls or has_tris:
                r.build_scene_acceleration_structure(
                    sphere_data,
                    cyl_data if has_cyls else None,
                    tri_data if has_tris else None,
                )
            else:
                r.build_sphere_acceleration_structure(sphere_data)
            self._last_geom_key = geom_key
            self._last_geom_data = tuple(
                None if a is None else np.array(a, copy=True)
                for a in geom_data)
            self._accum_count = 0

        # Upload sphere colors.
        color_bytes = np.ascontiguousarray(sphere_colors, dtype=np.uint8)
        r.destroy_data_buffer(self._color_buffer)
        self._color_buffer = r.create_data_buffer(color_bytes)

        # Upload cylinder colors.
        if cyl_colors is not None and len(cyl_colors) > 0:
            cyl_color_bytes = np.ascontiguousarray(cyl_colors, dtype=np.uint8)
            r.destroy_data_buffer(self._cyl_color_buffer)
            self._cyl_color_buffer = r.create_data_buffer(cyl_color_bytes)
        else:
            r.destroy_data_buffer(self._cyl_color_buffer)
            self._cyl_color_buffer = None

        # Upload triangle colors.
        if tri_colors is not None and len(tri_colors) > 0:
            tri_color_bytes = np.ascontiguousarray(tri_colors, dtype=np.uint8)
            r.destroy_data_buffer(self._tri_color_buffer)
            self._tri_color_buffer = r.create_data_buffer(tri_color_bytes)
        else:
            r.destroy_data_buffer(self._tri_color_buffer)
            self._tri_color_buffer = None

        # Ensure RGBA textures for output and ping-pong accumulation.
        res = ctx.resources
        for name in ("pt_color", "pt_accum_0", "pt_accum_1"):
            if res.needs_resize(name, w, h):
                r.destroy_texture(res.remove(name))
                res.put(name, r.create_rgba_texture(w, h), width=w, height=h)
                self._accum_count = 0
        color_tex = res.get("pt_color")
        prev_tex = res.get(f"pt_accum_{self._ping}")
        accum_tex = res.get(f"pt_accum_{1 - self._ping}")

        pm = r.current_projection_matrix
        cp = ctx.camera_position
        if pm is None or cp is None:
            return
        inv_proj = np.linalg.inv(np.asarray(pm, dtype=np.float32))
        inv_view = np.asarray(cp.opengl_matrix(), dtype=np.float32)

        # Light directions: ChimeraX convention is direction light travels
        # (away from source); shader expects direction toward the light.
        kld = -np.array(lp.key_light_direction, dtype=np.float64)
        key_light_world = cp.transform_vector(kld)
        key_light_world = key_light_world / np.linalg.norm(key_light_world)

        fld = -np.array(lp.fill_light_direction, dtype=np.float64)
        fill_light_world = cp.transform_vector(fld)
        fill_light_world = fill_light_world / np.linalg.norm(fill_light_world)

        bg = r._background_color[:3]
        ambient = tuple(
            lp.ambient_light_intensity * c for c in lp.ambient_light_color
        )

        # Participating-medium volume. PathTracePass runs before the volume's
        # own rasterized draw (TransparentPass), so drive its per-frame GPU
        # prep here and read back the world-space box / transfer function.
        vol_kwargs, vol_key = self._collect_volume(ctx, r, lp)
        r._pt_handled_volume_id = vol_key[0] if vol_key else None

        # Reset accumulation on camera, projection, lighting, or volume change.
        lighting_key = (
            tuple(float(x) for x in key_light_world),
            float(lp.key_light_intensity),
            tuple(float(x) for x in fill_light_world),
            float(lp.fill_light_intensity),
            tuple(float(x) for x in ambient),
            tuple(float(x) for x in bg),
            vol_key,
        )
        if (
            self._last_view_matrix is None
            or not np.allclose(inv_view, self._last_view_matrix, atol=1e-6)
            or not np.allclose(inv_proj, self._last_proj_matrix, atol=1e-6)
            or lighting_key != self._last_lighting_key
        ):
            self._accum_count = 0
        self._last_view_matrix = inv_view.copy()
        self._last_proj_matrix = inv_proj.copy()
        self._last_lighting_key = lighting_key

        self._frame_counter += 1
        self._accum_count += 1

        r._ensure_pt_pipeline()
        r.dispatch_pathtracer(
            color_tex,
            prev_tex,
            accum_tex,
            self._color_buffer,
            self._cyl_color_buffer,
            self._tri_color_buffer,
            inv_proj,
            inv_view,
            frame=self._frame_counter,
            accum_count=self._accum_count,
            key_light_dir=tuple(key_light_world),
            key_light_intensity=lp.key_light_intensity,
            fill_light_dir=tuple(fill_light_world),
            fill_light_intensity=lp.fill_light_intensity,
            ambient_color=ambient,
            background_color=bg,
            max_bounces=int(lp.pt_max_bounces),
            **vol_kwargs,
        )
        self._ping = 1 - self._ping

        # Cross-bilateral denoise: smooths Monte-Carlo speckle inside
        # each atom while the rasterised depth buffer keeps silhouettes
        # sharp. Falls back to a direct blit if the renderer doesn't
        # expose a denoise pass (Metal — TODO) or the scene depth tex
        # isn't available.
        depth_tex = (r.get_scene_depth_texture()
                     if hasattr(r, "get_scene_depth_texture") else None)
        denoise_tex = None
        if (depth_tex is not None
                and hasattr(r, "draw_pt_denoise")):
            if res.needs_resize("pt_denoised", w, h):
                r.destroy_texture(res.remove("pt_denoised"))
                res.put("pt_denoised", r.create_color_texture(w, h),
                        width=w, height=h)
            denoise_tex = res.get("pt_denoised")

            denoise_uniforms = np.zeros(4, dtype=np.float32)
            denoise_uniforms[0] = 1.0 / w     # texel_size.x
            denoise_uniforms[1] = 1.0 / h     # texel_size.y
            denoise_uniforms[2] = float(lp.pt_denoise_threshold)
            denoise_data = np.ascontiguousarray(denoise_uniforms)

            r.begin_color_pass(denoise_tex, w, h, clear=True)
            r.draw_pt_denoise(color_tex, depth_tex, denoise_data)
            r._vulkan.end_render_pass(context=r._ctx)

        # Blit the result to the framebuffer.  Clear color (don't load
        # the rasterized scene) so the path-traced image fully replaces it.
        r.begin_pt_blit_pass()
        r.draw_pt_blit(denoise_tex if denoise_tex is not None else color_tex)

        # Keep accumulating.
        max_accum = int(lp.pt_max_accum)
        if self._accum_count < max_accum:
            ctx.view._drawing_manager.redraw_needed = True

    @staticmethod
    def _collect_geometry(ctx):
        """Collect spheres, cylinders, and triangle meshes with colors.

        Returns (sphere_data, sphere_colors, cylinder_data, cylinder_colors,
                 triangle_data, triangle_colors).
        """
        import numpy as np
        from .renderer import SHADER_IMPOSTOR_SPHERE, SHADER_IMPOSTOR_CYLINDER

        all_spheres = []
        all_sphere_colors = []
        all_cyls = []
        all_cyl_colors = []
        all_tris = []
        all_tri_colors = []

        for drawing in ctx.opaque_drawings:
            # Determine world transform for this drawing.
            parent_rot = None
            parent_trans = None
            if drawing.parent is not None:
                scene_pos = drawing.parent.get_scene_positions(
                    displayed_only=True
                )
                if not scene_pos.is_identity():
                    p = scene_pos[0]
                    parent_rot = np.asarray(p.matrix[:3, :3], dtype=np.float64)
                    parent_trans = np.asarray(p.matrix[:3, 3], dtype=np.float64)

            sas = drawing.positions.shift_and_scale_array()
            if sas is not None and len(sas) > 0:
                # Sphere impostor drawing.
                coords = sas[:, :3].copy()
                radii = sas[:, 3:4].copy()
                if parent_rot is not None:
                    coords = (coords @ parent_rot.T) + parent_trans

                all_spheres.append(np.hstack([coords, radii]))

                colors = drawing.colors
                if colors is not None and len(colors) == len(sas):
                    all_sphere_colors.append(colors)
                else:
                    all_sphere_colors.append(
                        np.full((len(sas), 4), 255, dtype=np.uint8)
                    )
                continue

            # Check for cylinder impostor (bond/pseudobond drawings have
            # impostor_dashes attribute and use instance matrices).
            if hasattr(drawing, 'impostor_dashes'):
                matrices = drawing.positions.opengl_matrices()
                if matrices is not None and len(matrices) > 0:
                    # Each 4x4 matrix transforms a unit cylinder (Z, -0.5..+0.5).
                    # Column 3 row 0-2 = translation (center).
                    # Column 2 row 0-2 = Z axis (cylinder direction * length).
                    # Column 0 row 0-2 length = radius.
                    # The W components of each column carry the second atom
                    # color as floats that encode packed RGBA bytes.
                    n = len(matrices)
                    dashes = getattr(drawing, 'impostor_dashes', 0)
                    cyl_data = np.zeros((n, 8), dtype=np.float32)

                    for i in range(n):
                        m = matrices[i]
                        center = m[3, :3].copy()
                        axis = m[2, :3].copy()
                        radius = np.linalg.norm(m[0, :3])

                        p0 = center - 0.5 * axis
                        p1 = center + 0.5 * axis

                        if parent_rot is not None:
                            p0 = (parent_rot @ p0) + parent_trans
                            p1 = (parent_rot @ p1) + parent_trans

                        cyl_data[i, :3] = p0
                        # Shrink the bond radius slightly so the cylinder
                        # surface doesn't protrude past the atom sphere at
                        # the junction.
                        cyl_data[i, 3] = radius * 0.995
                        cyl_data[i, 4:7] = p1
                        # Pack dash count into the unused .w of p1.
                        cyl_data[i, 7] = float(dashes)

                    all_cyls.append(cyl_data)

                    # Cylinder colors: two RGBA8 colors per cylinder packed
                    # into a single 8-byte row (color1, color2).
                    colors = drawing.colors
                    if colors is None or len(colors) != n:
                        colors = np.full((n, 4), 200, dtype=np.uint8)

                    # Extract second-atom color from the W column of the
                    # instance matrix.  The values are floats in 0.0–1.0.
                    color2_floats = matrices[:, :4, 3]  # shape (n, 4)
                    color2 = np.clip(color2_floats * 255.0, 0, 255).astype(np.uint8)

                    # Interleave: [color1_0, color2_0, color1_1, color2_1, ...]
                    paired = np.empty((n, 2, 4), dtype=np.uint8)
                    paired[:, 0, :] = colors
                    paired[:, 1, :] = color2
                    all_cyl_colors.append(paired.reshape(n * 2, 4))
                continue

            # Check for triangle mesh drawing (e.g. SNFG sugar symbols).
            verts = drawing.vertices
            tris = drawing.triangles
            if verts is None or tris is None:
                continue
            if len(verts) == 0 or len(tris) == 0:
                continue

            verts = np.asarray(verts, dtype=np.float64)
            if parent_rot is not None:
                verts = (verts @ parent_rot.T) + parent_trans

            # Flatten triangles into per-triangle vertex data.
            # Layout: 12 floats per triangle (3 × vec4, w=0 padding).
            idx = np.asarray(tris, dtype=np.int32)
            n_tri = len(idx)
            tri_verts = verts[idx]  # (n_tri, 3, 3)
            tri_data = np.zeros((n_tri, 12), dtype=np.float32)
            tri_data[:, 0:3] = tri_verts[:, 0, :]
            # tri_data[:, 3] = 0  (padding)
            tri_data[:, 4:7] = tri_verts[:, 1, :]
            # tri_data[:, 7] = 0
            tri_data[:, 8:11] = tri_verts[:, 2, :]
            # tri_data[:, 11] = 0
            all_tris.append(tri_data)

            # Per-triangle color: use vertex color of the first vertex
            # of each triangle, or the drawing's single color.
            vc = drawing.vertex_colors
            if vc is not None and len(vc) == len(drawing.vertices):
                tri_colors = np.asarray(vc, dtype=np.uint8)[idx[:, 0]]
            else:
                c = drawing.colors
                if c is not None and len(c) > 0:
                    tri_colors = np.tile(
                        np.asarray(c[0], dtype=np.uint8), (n_tri, 1)
                    )
                else:
                    tri_colors = np.full((n_tri, 4), 200, dtype=np.uint8)
            all_tri_colors.append(tri_colors)

        sphere_data = (np.vstack(all_spheres).astype(np.float32)
                       if all_spheres else None)
        sphere_colors = (np.vstack(all_sphere_colors).astype(np.uint8)
                         if all_sphere_colors else None)
        cyl_data = (np.vstack(all_cyls).astype(np.float32)
                    if all_cyls else None)
        cyl_colors = (np.vstack(all_cyl_colors).astype(np.uint8)
                      if all_cyl_colors else None)
        tri_data = (np.vstack(all_tris).astype(np.float32)
                    if all_tris else None)
        tri_colors = (np.vstack(all_tri_colors).astype(np.uint8)
                      if all_tri_colors else None)
        return (sphere_data, sphere_colors, cyl_data, cyl_colors,
                tri_data, tri_colors)

    def _collect_volume(self, ctx, r, lp):
        """Prepare the displayed density volume for path tracing.

        Returns ``(vol_kwargs, vol_key)`` where ``vol_kwargs`` is the set of
        ``volume_*`` keyword arguments for ``dispatch_pathtracer`` (empty when
        no path-traceable volume is shown) and ``vol_key`` is a hashable
        accumulation-reset key that changes whenever the medium changes.

        Requires a full-raycast volume (``projectionMode rays``), which builds
        a VolumeRaycastDrawing with a 3D density texture. In rays mode the
        density is scalar and a transfer function colors it; the path tracer
        samples the density (3D) and a transfer function rebuilt here as a 2D
        Nx1 texture (a 1D texture doesn't survive export to Metal).

        PathTracePass runs before the volume's own rasterized draw, so this
        drives the drawing's per-frame update (coloring, textures, bounding
        box) directly rather than relying on that draw having happened.
        """
        vol = _pt_volume_drawing(ctx)
        if vol is None:
            return {}, None

        try:
            vol._update_coloring()
            density = getattr(vol, "texture", None)
            if density is None:
                return {}, None
            # Force lazy allocation/upload so the C handles exist.
            density.bind_texture()
            density_handle = getattr(density, "_handle", None)
            if density_handle is None:
                return {}, None
            # Populates r._raycast_bbox_* / _raycast_full_region_* (world space).
            vol.set_shader_options(r)
            bbox_min = tuple(float(x) for x in r._raycast_bbox_min)
            bbox_max = tuple(float(x) for x in r._raycast_bbox_max)
            region_min = tuple(float(x) for x in r._raycast_full_region_min)
            region_max = tuple(float(x) for x in r._raycast_full_region_max)

            # Transfer function. With a GPU colormap the 3D texture is scalar
            # density and a separate transfer function colors it; otherwise
            # the 3D texture is already RGBA (the shader branches on whether a
            # colormap is passed). The volume's own colormap is a 1D texture,
            # which doesn't survive export to Metal, so build a 2D (Nx1)
            # texture from the same transfer-function data and rebuild it only
            # when it changes.
            import numpy as np
            colormap_handle = None
            cmap_off, cmap_inv = 0.0, 1.0
            cmap_data, crange = vol._image_render._color_table(vol._axis)
            if cmap_data is not None:
                arr = np.ascontiguousarray(cmap_data)
                if arr.ndim == 2:                       # (N, C) -> (1, N, C)
                    arr = arr.reshape(1, arr.shape[0], arr.shape[1])
                key = (arr.shape, arr.dtype.str, arr.tobytes())
                if self._pt_cmap_key != key:
                    from chimerax.graphics import Texture
                    if self._pt_colormap_tex is not None:
                        self._pt_colormap_tex.delete_texture()
                    t = Texture(arr, dimension=2, clamp_to_edge=True)
                    t.bind_texture()
                    self._pt_colormap_tex = t
                    self._pt_cmap_key = key
                colormap_handle = getattr(self._pt_colormap_tex, "_handle", None)
            if colormap_handle is not None:
                # Match vulkan.set_colormap: pre-scale (v0, v1) by the density
                # texture's sampler normalization so the shader's
                # (sample - offset) * inv_range recovers (value - v0)/(v1 - v0).
                v0, v1 = crange
                s = (density.normalization()
                     if hasattr(density, "normalization") else 1.0)
                denom = s * (v1 - v0)
                cmap_off = s * float(v0)
                cmap_inv = (1.0 / denom) if denom != 0.0 else 0.0
        except Exception:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            return {}, None

        vol_kwargs = dict(
            volume_density_texture=density_handle,
            volume_colormap_texture=colormap_handle,
            volume_bbox_min=bbox_min,
            volume_bbox_max=bbox_max,
            volume_region_min=region_min,
            volume_region_max=region_max,
            volume_sigma_max=float(lp.pt_volume_density),
            volume_phase_g=float(lp.pt_volume_g),
            volume_colormap_range=(cmap_off, cmap_inv),
        )
        # Suppress the rasterized volume so it isn't composited over the PT
        # blit. Cover both the raycast drawing and its Image3d container, since
        # either may be the unit the TransparentPass draws.
        container = getattr(vol, "parent", None)
        suppress = frozenset(id(x) for x in (vol, container) if x is not None)
        vol_key = (suppress, bbox_min, bbox_max, region_min, region_max,
                   colormap_handle is not None,
                   round(cmap_off, 6), round(cmap_inv, 6),
                   float(lp.pt_volume_density), float(lp.pt_volume_g))
        return vol_kwargs, vol_key


class OffscreenFinishPass(RenderPass):
    """Composite offscreen framebuffer back to the screen."""

    name = "offscreen_finish"

    def enabled(self, ctx):
        return ctx.offscreen is not None

    def execute(self, ctx):
        ctx.offscreen.finish(ctx.renderer)
