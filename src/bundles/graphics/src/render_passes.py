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


class OffscreenFinishPass(RenderPass):
    """Composite offscreen framebuffer back to the screen."""

    name = "offscreen_finish"

    def enabled(self, ctx):
        return ctx.offscreen is not None

    def execute(self, ctx):
        ctx.offscreen.finish(ctx.renderer)
