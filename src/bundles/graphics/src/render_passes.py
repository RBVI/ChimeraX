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
        return ctx.multishadow_enabled and len(ctx.opaque_drawings) > 0

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
        from .drawing import draw_opaque

        draw_opaque(ctx.renderer, ctx.opaque_drawings)


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


class TransparentPass(RenderPass):
    """Render transparent geometry."""

    name = "transparent"

    def enabled(self, ctx):
        return len(ctx.transparent_drawings) > 0

    def execute(self, ctx):
        from .drawing import draw_transparent

        draw_transparent(ctx.renderer, ctx.transparent_drawings)


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
        return len(ctx.highlight_drawings) > 0

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
    """Ground Truth Ambient Occlusion — screen-space post-process.

    Reads the depth buffer after the opaque pass, computes per-pixel
    occlusion, and composites the result into the color buffer with
    multiply blending.
    """

    name = "gtao"

    def enabled(self, ctx):
        r = ctx.renderer
        return hasattr(r, "draw_fullscreen_quad") and getattr(r.lighting, "gtao", 0) > 0

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

        # Get the scene depth texture for sampling.
        depth_tex = r.get_scene_depth_texture()
        if depth_tex is None:
            return

        # Build inverse projection matrix for position reconstruction.
        pm = r.current_projection_matrix
        if pm is None:
            return
        inv_proj = np.linalg.inv(np.asarray(pm, dtype=np.float32))

        # Pack GTAO uniforms matching the shader's GTAOUniforms struct.
        radius = getattr(lp, "gtao_radius", 16.0)
        intensity = getattr(lp, "gtao_intensity", 1.5)
        directions = int(lp.gtao)
        steps = int(getattr(lp, "gtao_steps", 4))
        near, far = r._near_far_clip

        bias_frac = float(getattr(lp, "gtao_bias_frac", 0.02))

        uniforms = np.zeros(28, dtype=np.float32)  # 112 bytes (16-byte aligned)
        uniforms[0:16] = inv_proj.reshape(-1)  # float4x4 inv_projection
        uniforms[16] = 1.0 / w  # texel_size.x
        uniforms[17] = 1.0 / h  # texel_size.y
        uniforms[18] = radius  # radius
        uniforms[19] = intensity  # intensity
        uniforms[20] = np.float32(
            directions
        )  # directions (as float, reinterpret in shader)
        uniforms[21] = np.float32(steps)  # steps
        uniforms[22] = np.float32(near)  # near
        uniforms[23] = np.float32(far)  # far
        uniforms[24] = np.float32(bias_frac)  # bias_frac
        # Reinterpret directions/steps as int in the uniform buffer.
        uniforms.view(np.int32)[20] = directions
        uniforms.view(np.int32)[21] = steps

        uniforms_data = np.ascontiguousarray(uniforms)

        # --- Pass 1: compute AO from depth ---
        r.begin_color_pass(ao_tex, w, h, clear=True)
        r.draw_fullscreen_quad(
            depth_tex,
            "gtao.metal",
            "gtao_vertex",
            "gtao_fragment",
            uniforms_data=uniforms_data,
        )
        r._metal.end_render_pass(context=r._ctx)

        # --- Restart main render pass (load existing color + depth) ---
        r.restart_main_pass()

        # --- Pass 2: multiply AO into color buffer ---
        r.draw_fullscreen_quad(
            ao_tex, "gtao.metal", "gtao_vertex", "gtao_composite_fragment"
        )


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

        if len(sphere_data) != self._last_atom_count:
            r.build_sphere_acceleration_structure(sphere_data)
            self._last_atom_count = len(sphere_data)
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
    _ping = 0
    _color_buffer = None
    _cyl_color_buffer = None

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
        w, h = r.render_size()
        if w == 0 or h == 0:
            return

        sphere_data, sphere_colors, cyl_data, cyl_colors = \
            self._collect_geometry(ctx)
        if sphere_data is None or len(sphere_data) == 0:
            return

        geom_key = (len(sphere_data), len(cyl_data) if cyl_data is not None else 0)
        if geom_key != self._last_geom_key:
            if cyl_data is not None and len(cyl_data) > 0:
                r.build_mixed_acceleration_structure(sphere_data, cyl_data)
            else:
                r.build_sphere_acceleration_structure(sphere_data)
            self._last_geom_key = geom_key
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

        # Reset accumulation on camera, projection, or lighting change.
        lighting_key = (
            tuple(float(x) for x in key_light_world),
            float(lp.key_light_intensity),
            tuple(float(x) for x in fill_light_world),
            float(lp.fill_light_intensity),
            tuple(float(x) for x in ambient),
            tuple(float(x) for x in bg),
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
        """Collect spheres and cylinders with colors in world space.

        Returns (sphere_data, sphere_colors, cylinder_data, cylinder_colors).
        """
        import numpy as np
        from .renderer import SHADER_IMPOSTOR_SPHERE, SHADER_IMPOSTOR_CYLINDER

        all_spheres = []
        all_sphere_colors = []
        all_cyls = []
        all_cyl_colors = []

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
            if not hasattr(drawing, 'impostor_dashes'):
                continue
            matrices = drawing.positions.opengl_matrices()
            if matrices is not None and len(matrices) > 0:
                # Each 4x4 matrix transforms a unit cylinder (Z, -0.5..+0.5).
                # Column 3 row 0-2 = translation (center).
                # Column 2 row 0-2 = Z axis (cylinder direction * length).
                # Column 0 row 0-2 length = radius.
                # The 4th row carries the second atom color (packed).
                n = len(matrices)
                cyl_data = np.zeros((n, 8), dtype=np.float32)

                for i in range(n):
                    m = matrices[i]
                    # opengl_matrices() stores columns as rows:
                    # m[0] = X axis, m[1] = Y axis, m[2] = Z axis, m[3] = translation
                    # (the w-component m[j,3] carries packed color2, not geometry)
                    center = m[3, :3].copy()
                    axis = m[2, :3].copy()
                    radius = np.linalg.norm(m[0, :3])

                    p0 = center - 0.5 * axis
                    p1 = center + 0.5 * axis

                    if parent_rot is not None:
                        p0 = (parent_rot @ p0) + parent_trans
                        p1 = (parent_rot @ p1) + parent_trans

                    cyl_data[i, :3] = p0
                    cyl_data[i, 3] = radius
                    cyl_data[i, 4:7] = p1

                all_cyls.append(cyl_data)

                # Cylinder colors: use drawing.colors (first atom color).
                colors = drawing.colors
                if colors is not None and len(colors) == n:
                    all_cyl_colors.append(colors)
                else:
                    all_cyl_colors.append(
                        np.full((n, 4), 200, dtype=np.uint8)
                    )

        sphere_data = (np.vstack(all_spheres).astype(np.float32)
                       if all_spheres else None)
        sphere_colors = (np.vstack(all_sphere_colors).astype(np.uint8)
                         if all_sphere_colors else None)
        cyl_data = (np.vstack(all_cyls).astype(np.float32)
                    if all_cyls else None)
        cyl_colors = (np.vstack(all_cyl_colors).astype(np.uint8)
                      if all_cyl_colors else None)
        return sphere_data, sphere_colors, cyl_data, cyl_colors


class OffscreenFinishPass(RenderPass):
    """Composite offscreen framebuffer back to the screen."""

    name = "offscreen_finish"

    def enabled(self, ctx):
        return ctx.offscreen is not None

    def execute(self, ctx):
        ctx.offscreen.finish(ctx.renderer)
