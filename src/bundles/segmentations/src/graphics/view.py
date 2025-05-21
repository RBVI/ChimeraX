from chimerax.graphics import View
from chimerax.graphics.drawing import (
    Drawing,
    draw_depth,
    draw_opaque,
    draw_transparent,
    draw_highlight_outline,
    draw_on_top,
    draw_overlays,
)

class OrthoplaneView(View):
    """OrthoplaneView splits the draw() method of view in two, which gives us more fine-grained
    control over how draw() and _draw_scene() works."""

    def __init__(self, drawing, axis, *, window_size=(256, 256), trigger_set=None):
        super().__init__(drawing, window_size=window_size, trigger_set=trigger_set)
        self.segmentation_overlays = []
        self.cursor_overlays = []
        self.guideline_overlays = []
        self.axis = axis

    def add_guideline_overlay(self, overlay):
        self.guideline_overlays.append(overlay)

    def remove_guideline_overlays(self, overlays):
        for overlay in overlays:
            self.guideline_overlays.remove(overlay)

    def add_segmentation_overlay(self, overlay):
        self.segmentation_overlays.append(overlay)

    def add_cursor_overlay(self, overlay):
        self.cursor_overlays.append(overlay)

    def remove_cursor_overlays(self, overlays):
        for overlay in overlays:
            self.cursor_overlays.remove(overlay)

    def remove_segmentation_overlay(self, overlay):
        self.segmentation_overlays.remove(overlay)

    def background_contrast_color(self):
        from chimerax.core.colors import contrast_with
        return contrast_with(self.background_color)

    def prepare_scene_for_drawing(self, camera=None, check_for_changes=True):
        if not self._use_opengl():
            return

        if check_for_changes:
            self.check_for_drawing_change()

        if camera is None:
            camera = self.camera
        self._render.set_frame_number(self.frame_number)
        self._render.set_background_color(self.background_color)
        self._render.update_viewport()

        if self.update_lighting:
            self._render.set_lighting_shader_capabilities()
            self._render.update_lighting_parameters()
            self.update_lighting = False
        self._render.activate_lighting()

    def finalize_draw(self, camera=None, swap_buffers=True, should_draw_overlays=True):
        if camera is None:
            camera = self.camera
        camera.combine_rendered_camera_views(self._render)
        # We want to draw the overlays in a specific order: segmentations first, then the guidelines, then the cursor
        #
        if self._overlays and should_draw_overlays:
            odrawings = sum(
                [
                    o.all_drawings(displayed_only=True)
                    for o in self.segmentation_overlays
                ],
                [],
            )
            draw_overlays(odrawings, self._render)
            odrawings = sum(
                [o.all_drawings(displayed_only=True) for o in self.guideline_overlays],
                [],
            )
            draw_overlays(odrawings, self._render)
            odrawings = sum(
                [o.all_drawings(displayed_only=True) for o in self.cursor_overlays], []
            )
            draw_overlays(odrawings, self._render)
            odrawings = sum(
                [o.all_drawings(displayed_only=True) for o in self._overlays], []
            )
            draw_overlays(odrawings, self._render)

        if swap_buffers:
            if camera.do_swap_buffers():
                self._render.swap_buffers()
            self.redraw_needed = False
            self._render.done_current()

    def _draw_scene(self, camera, drawings):
        if drawings is None:
            drawings = [self.drawing]
        self.clip_planes.enable_clip_plane_graphics(self._render, camera.position)
        (
            opaque_drawings,
            transparent_drawings,
            highlight_drawings,
            on_top_drawings,
        ) = self._drawings_by_pass(drawings)
        no_drawings = (
            len(opaque_drawings) == 0
            and len(transparent_drawings) == 0
            and len(highlight_drawings) == 0
            and len(on_top_drawings) == 0
        )

        offscreen = self._render.offscreen if self._render.offscreen.enabled else None
        if highlight_drawings and self._render.outline.offscreen_outline_needed:
            offscreen = self._render.offscreen
        if (
            offscreen
            and self._render.current_framebuffer()
            is not self._render.default_framebuffer()
        ):
            offscreen = None  # Already using an offscreen framebuffer

        silhouette = self.silhouette

        shadow, multishadow = self._compute_shadowmaps(
            opaque_drawings, transparent_drawings, camera
        )

        for vnum in range(camera.number_of_views()):
            camera.set_render_target(vnum, self._render)
            if no_drawings:
                camera.draw_background(vnum, self._render)
                continue
            if offscreen:
                offscreen.start(self._renderr)
            if silhouette.enabled:
                silhouette.start_silhouette_drawing(self._render)
            camera.draw_background(vnum, self._render)
            self._update_projection(camera, vnum)
            cp = camera.get_position(vnum)
            self._view_matrix = vm = cp.inverse(is_orthonormal=True)
            self._render.set_view_matrix(vm)
            if shadow:
                self._render.shadow.set_shadow_view(cp)
            if multishadow:
                self._render.multishadow.set_multishadow_view(cp)
                # Initial depth pass optimization to avoid lighting
                # calculation on hidden geometry
                if opaque_drawings:
                    draw_depth(self._render, opaque_drawings)
                    self._render.allow_equal_depth(True)
            self._start_timing()
            if opaque_drawings:
                for d in drawings:
                    # d.draw(self._render, Drawing.OPAQUE_DRAW_PASS)
                    d._update_blend_groups()
                    bi = d._blend_image
                    if bi:
                        if d is bi.master_image:
                            bi.draw(self._render, Drawing.OPAQUE_DRAW_PASS)
                        continue

                    pd = d._update_planes(self._render)
                    if pd._update_region:
                        pd.update_region()
                        pd._update_region = False
                    pd._update_coloring()
                    d._planes_2d._draw_geometry(self._render)
            if highlight_drawings:
                self._render.outline.set_outline_mask()  # copy depth to outline framebuffer
            if transparent_drawings:
                if silhouette.enabled:
                    # Draw opaque object silhouettes behind transparent surfaces
                    silhouette.draw_silhouette(self._render)
                draw_transparent(self._render, transparent_drawings)
            self._finish_timing()
            if multishadow:
                self._render.allow_equal_depth(False)
            if silhouette.enabled:
                silhouette.finish_silhouette_drawing(self._render)
            if highlight_drawings:
                draw_highlight_outline(
                    self._render,
                    highlight_drawings,
                    color=self._highlight_color,
                    pixel_width=self._highlight_width,
                )
            if on_top_drawings:
                draw_on_top(self._render, on_top_drawings)
            if offscreen:
                offscreen.finish(self._render)
