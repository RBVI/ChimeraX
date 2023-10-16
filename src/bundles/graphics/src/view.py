# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

'''
View
====
'''

class View:
    '''
    A View is the graphics windows that shows 3-dimensional drawings.
    It manages the camera and renders the drawing when needed.
    '''
    def __init__(self, drawing, *, window_size = (256,256), trigger_set = None):

        self.triggers = trigger_set
        self.drawing = drawing
        self.window_size = window_size		# pixels
        self._render = None
        self._opengl_initialized = False

        self.set_default_parameters()

        # Graphics overlays, used for example for crossfade
        self._overlays = []

        # Redrawing
        self.frame_number = 1
        self.redraw_needed = True
        self._time_graphics = False
        self.update_lighting = True

        self._drawing_manager = dm = _RedrawNeeded()
        if trigger_set:
            self.drawing.set_redraw_callback(dm)

    def set_default_parameters(self):
        # Lights and material properties
        from .opengl import Lighting, Material, Silhouette
        self.lighting = Lighting()
        self.material = Material()
        if hasattr(self, '_silhouette'):
            self._silhouette.delete()
        self._silhouette = Silhouette()

        # Red, green, blue, opacity, 0-1 range.
        self._background_rgba = (0, 0, 0, 0)
        self._highlight_color = (0, 1, 0, 1)
        self._highlight_width = 1	# pixels

        # Set silhouette and highlight thickness for retina displays
        r = self._render
        if r and r.opengl_context:
            pscale = r.opengl_context.pixel_scale()
            self.silhouette.thickness = pscale
            self.highlight_thickness = pscale

        # Create camera
        from .camera import MonoCamera
        self._camera = MonoCamera()
        from chimerax.geometry import Place
        self._view_matrix = Place()		# Temporary used during rendering

        # Clip planes
        from .clipping import ClipPlanes
        self.clip_planes = ClipPlanes()
        self._near_far_pad = 0.01		# Extra near-far clip plane spacing.
        self._min_near_fraction = 0.001		# Minimum near distance, fraction of depth

        # Center of rotation
        from numpy import array, float32
        self._center_of_rotation = array((0, 0, 0), float32)
        self._update_center_of_rotation = False
        self._center_of_rotation_method = 'front center'

    def delete(self):
        r = self._render
        if r:
            r.delete()
            self._render = None

    @property
    def render(self):
        return self._render
    
    def initialize_rendering(self, opengl_context):
        r = self._render
        if r is None:
            from .opengl import Render
            self._render = r = Render(opengl_context)
            r.lighting = self._lighting
            r.material = self._material
            r.silhouette = self._silhouette
            pscale = opengl_context.pixel_scale()
            self.silhouette.thickness = pscale
            self.highlight_thickness = pscale
        elif opengl_context is r.opengl_context:
            # OpenGL context switched between stereo and mono mode
            self._opengl_initialized = False
        else:
            raise ValueError("OpenGL context is already set")

    def _use_opengl(self):
        if self._render is None:
            raise RuntimeError("running without graphics")
        if not self._render.make_current():
            return False
        self._initialize_opengl()
        return True

    def _initialize_opengl(self):

        # Delay making OpenGL calls until drawing is attempted.
        if self._opengl_initialized:
            return
        self._opengl_initialized = True
        
        r = self._render
        r.check_opengl_version()
        r.set_background_color(self.background_color)

        w, h = self.window_size
        r.initialize_opengl(w, h)

    def _get_camera(self):
        return self._camera
    def _set_camera(self, camera):
        c = self._camera
        c.clear_special_render_modes(self._render)
        self._camera = camera
        camera.set_special_render_modes(self._render)
        self.redraw_needed = True
    camera = property(_get_camera, _set_camera)
    '''The Camera controlling the vantage shown in the graphics window.'''

    def draw(self, camera = None, drawings = None,
             check_for_changes = True, swap_buffers = True):
        '''
        Draw the scene.
        '''
        if not self._use_opengl():
            return	# OpenGL not available

        # OpenGL call list code is experimental study for running opengl in separate thread.
        use_calllist = False
        from . import gllist
        if use_calllist and gllist.replay_opengl(self, drawings, camera, swap_buffers):
            return
        
        if check_for_changes:
            self.check_for_drawing_change()

        if camera is None:
            camera = self.camera

        r = self._render
        r.set_frame_number(self.frame_number)
        r.set_background_color(self.background_color)
        r.update_viewport()	# Need this when window resized.

        if self.update_lighting:
            self.update_lighting = False
            r.set_lighting_shader_capabilities()
            r.update_lighting_parameters()
        r.activate_lighting()

        self._draw_scene(camera, drawings)

        camera.combine_rendered_camera_views(r)

        if self._overlays:
            odrawings = sum([o.all_drawings(displayed_only = True) for o in self._overlays], [])
            from .drawing import draw_overlays
            draw_overlays(odrawings, r)

        if use_calllist:
            gllist.call_opengl_list(self, trace=True)
        
        if swap_buffers:
            if camera.do_swap_buffers():
                r.swap_buffers()
            self.redraw_needed = False
            r.done_current()

    def _draw_scene(self, camera, drawings):

        r = self._render
        self.clip_planes.enable_clip_plane_graphics(r, camera.position)
        mdraw = [self.drawing] if drawings is None else drawings
        (opaque_drawings, transparent_drawings,
         highlight_drawings, on_top_drawings) = self._drawings_by_pass(mdraw)
        no_drawings = (len(opaque_drawings) == 0 and
                       len(transparent_drawings) == 0 and
                       len(highlight_drawings) == 0 and
                       len(on_top_drawings) == 0)

        offscreen = r.offscreen if r.offscreen.enabled else None
        if highlight_drawings and r.outline.offscreen_outline_needed:
            offscreen = r.offscreen
        if offscreen and r.current_framebuffer() is not r.default_framebuffer():
            offscreen = None  # Already using an offscreen framebuffer
            
        silhouette = self.silhouette

        shadow, multishadow = self._compute_shadowmaps(opaque_drawings, transparent_drawings, camera)
            
        from .drawing import draw_depth, draw_opaque, draw_transparent, draw_highlight_outline, draw_on_top
        for vnum in range(camera.number_of_views()):
            camera.set_render_target(vnum, r)
            if no_drawings:
                camera.draw_background(vnum, r)
                continue
            if offscreen:
                offscreen.start(r)
            if silhouette.enabled:
                silhouette.start_silhouette_drawing(r)
            camera.draw_background(vnum, r)
            self._update_projection(camera, vnum)
            if r.recording_opengl:
                from . import gllist
                cp = gllist.ViewMatrixFunc(self, vnum)
            else:
                cp = camera.get_position(vnum)
            self._view_matrix = vm = cp.inverse(is_orthonormal = True)
            r.set_view_matrix(vm)
            if shadow:
                r.shadow.set_shadow_view(cp)
            if multishadow:
                r.multishadow.set_multishadow_view(cp)
                # Initial depth pass optimization to avoid lighting
                # calculation on hidden geometry
                if opaque_drawings:
                    draw_depth(r, opaque_drawings)
                    r.allow_equal_depth(True)
            self._start_timing()
            if opaque_drawings:
                draw_opaque(r, opaque_drawings)
            if highlight_drawings:
                r.outline.set_outline_mask()       # copy depth to outline framebuffer
            if transparent_drawings:
                if silhouette.enabled:
                    # Draw opaque object silhouettes behind transparent surfaces
                    silhouette.draw_silhouette(r)
                draw_transparent(r, transparent_drawings)
            self._finish_timing()
            if multishadow:
                r.allow_equal_depth(False)
            if silhouette.enabled:
                silhouette.finish_silhouette_drawing(r)
            if highlight_drawings:
                draw_highlight_outline(r, highlight_drawings, color = self._highlight_color,
                                       pixel_width = self._highlight_width)
            if on_top_drawings:
                draw_on_top(r, on_top_drawings)
            if offscreen:
                offscreen.finish(r)

                
    def _drawings_by_pass(self, drawings):
        pass_drawings = {}
        for d in drawings:
            d.drawings_for_each_pass(pass_drawings)
        from .drawing import Drawing
        passes = (Drawing.OPAQUE_DRAW_PASS,
                  Drawing.TRANSPARENT_DRAW_PASS,
                  Drawing.HIGHLIGHT_DRAW_PASS,
                  Drawing.LAST_DRAW_PASS)
        return [pass_drawings.get(draw_pass, []) for draw_pass in passes]

    def check_for_drawing_change(self):
        trig = self.triggers
        if trig:
            trig.activate_trigger('graphics update', self)

        c = self.camera
        cp = self.clip_planes
        dm = self._drawing_manager
        draw = self.redraw_needed or c.redraw_needed or cp.changed or dm.redraw_needed
        self._cam_only_change = c.redraw_needed and not (self.redraw_needed or cp.changed or dm.redraw_needed or self.update_lighting)
        if not draw:
            return False

        if dm.shape_changed:
            if trig:
                trig.activate_trigger('shape changed', self)	# Used for updating pseudobond graphics

        corm = self.center_of_rotation_method
        if corm == 'front center':
            if dm.shape_changed or cp.changed:
                self._update_center_of_rotation = True
        elif corm == 'center of view' and cp.changed:
            self._update_center_of_rotation = True

        if dm.shadows_changed() or cp.changed:
            r = self.render
            if r:
                r.multishadow.multishadow_update_needed = True

        c.redraw_needed = False
        dm.clear_changes()
        cp.changed = False

        self.redraw_needed = True

        return True

    def draw_xor_rectangle(self, x1, y1, x2, y2, color):
        if not self._use_opengl():
            return	# OpenGL not available
        if not self.render.front_buffer_valid:
            self.draw(check_for_changes = False)
            self._use_opengl()
        d = getattr(self, '_rectangle_drawing', None)
        from .drawing import draw_xor_rectangle
        self._rectangle_drawing = draw_xor_rectangle(self._render, x1, y1, x2, y2, color, d)

    @property
    def shape_changed(self):
        return self._drawing_manager.shape_changed

    def clear_drawing_changes(self):
        return self._drawing_manager.clear_changes()

    def get_background_color(self):
        return self._background_rgba

    def set_background_color(self, rgba):
        import numpy
        color = numpy.array(rgba, dtype=numpy.float32)
        color[3] = 0	# For transparent background images.
        lp = self._lighting
        if tuple(lp.depth_cue_color) == tuple(self._background_rgba[:3]):
            # Make depth cue color follow background color if they are the same.
            lp.depth_cue_color = tuple(color[:3])
            self.update_lighting = True
        self._background_rgba = color
        self.redraw_needed = True
        if self.triggers:
            from chimerax.core.core_settings import settings
            from chimerax.core.colors import Color
            settings.background_color = Color(rgba=color)
    background_color = property(get_background_color, set_background_color)
    '''Background color as R, G, B, A values in 0-1 range.'''

    def _get_highlight_color(self):
        return self._highlight_color
    def _set_highlight_color(self, rgba):
        self._highlight_color = rgba
        self.redraw_needed = True
    highlight_color = property(_get_highlight_color, _set_highlight_color)
    '''Highlight outline color as R, G, B, A values in 0-1 range.'''

    def _get_highlight_thickness(self):
        return self._highlight_width
    def _set_highlight_thickness(self, thickness):
        self._highlight_width = thickness
        self.redraw_needed = True
    highlight_thickness = property(_get_highlight_thickness, _set_highlight_thickness)
    '''Highlight outline thickness in pixels.'''

    def _get_lighting(self):
        return self._lighting

    def _set_lighting(self, lighting):
        self._lighting = lighting
        r = self._render
        if r:
            r.lighting = lighting
        self.update_lighting = True
        self.redraw_needed = True

    lighting = property(_get_lighting, _set_lighting)
    '''Lighting parameters.'''

    def _get_material(self):
        return self._material

    def _set_material(self, material):
        self._material = material
        r = self._render
        if r:
            r.material = material
        self.update_lighting = True
        self.redraw_needed = True

    material = property(_get_material, _set_material)
    '''Material reflectivity parameters.'''

    @property
    def silhouette(self):
        '''Silhouette parameters.'''
        return self._silhouette

    def add_overlay(self, overlay):
        '''
        Overlays are Drawings rendered after the normal scene is shown.
        They are used for effects such as motion blur or cross fade that
        blend the current rendered scene with a previous rendered scene.
        '''
        overlay.set_redraw_callback(self._drawing_manager)
        self._overlays.append(overlay)
        self.redraw_needed = True

    def overlays(self):
        '''The current list of overlay Drawings.'''
        return self._overlays

    def remove_overlays(self, overlays=None, delete = True):
        '''Remove the specified overlay Drawings.'''
        if overlays is None:
            overlays = self._overlays
        if delete:
            for o in overlays:
                o.delete()
        oset = set(overlays)
        self._overlays = [o for o in self._overlays if o not in oset]
        self.redraw_needed = True

    def image(self, width=None, height=None, supersample=None,
              transparent_background=False, camera=None, drawings=None):
        '''Capture an image of the current scene. A PIL image is returned.'''

        rgba = self.image_rgba(width=width, height=height, supersample=supersample,
                               transparent_background=transparent_background,
                               camera=camera, drawings=drawings)
        ncomp = 4 if transparent_background else 3
        from PIL import Image
        # Flip y-axis since PIL image has row 0 at top,
        # opengl has row 0 at bottom.
        pi = Image.fromarray(rgba[::-1, :, :ncomp])
        return pi

    def image_rgba(self, width=None, height=None, supersample=None,
                   transparent_background=False, camera=None, drawings=None):
        '''
        Capture an image of the current scene.
        A numpy uint8 rgba array is returned.
        '''

        if not self._use_opengl():
            return	# OpenGL not available

        w, h = self._window_size_matching_aspect(width, height)

        # TODO: For recording videos would be useful not to recreate framebuffer on every frame.
        from .opengl import Framebuffer
        fb = Framebuffer('image capture', self.render.opengl_context, w, h,
                         alpha = transparent_background)
        if not fb.activate():
            fb.delete()
            return None         # Image size exceeds framebuffer limits

        r = self._render
        r.push_framebuffer(fb)

        if camera is None:
            if drawings:
                from .camera import camera_framing_drawings
                c = camera_framing_drawings(drawings)
                if c is None:
                    c = self.camera	# Drawings not showing anything, any camera will do
            else:
                c = self.camera
        else:
            c = camera

        # Set flag that image save is in progress for Drawing.draw() routines
        # to adjust sizes of rendered objects with sizes in pixels to preserve
        # the on-screen sizes.  This is used for 2d label sizing.
        r.image_save = True
            
        if supersample is None:
            self.draw(c, drawings, swap_buffers = False)
            rgba = r.frame_buffer_image(w, h)
        else:
            from numpy import zeros, float32, uint8
            srgba = zeros((h, w, 4), float32)
            n = supersample
            s = 1.0 / n
            s0 = -0.5 + 0.5 * s
            for i in range(n):
                for j in range(n):
                    c.set_fixed_pixel_shift((s0 + i * s, s0 + j * s))
                    self.draw(c, drawings, swap_buffers = False)
                    srgba += r.frame_buffer_image(w, h)
            c.set_fixed_pixel_shift((0, 0))
            srgba /= n * n
            # third index 0, 1, 2, 3 is r, g, b, a
            rgba = srgba.astype(uint8)
        r.pop_framebuffer()
        fb.delete()

        delattr(r, 'image_save')

        return rgba

    def frame_buffer_rgba(self):
        '''
        Return a numpy array of R, G, B, A values of the currently
        rendered scene.  This is used for blending effects such as motion
        blur and cross fades.
        '''
        r = self._render
        w, h = r.render_size()
        rgba = self._render.frame_buffer_image(w, h, front_buffer = True)
        return rgba

    def resize(self, width, height):
        '''
        This is called when the graphics window was resized by the
        user and causes the OpenGL rendering to use the specified new
        window size.
        '''
        new_size = (width, height)
        if self.window_size == new_size:
            return
        self.window_size = new_size
        r = self._render
        if r:
            r.set_default_framebuffer_size(width, height)
            self.redraw_needed = True

    def _window_size_matching_aspect(self, width, height):
        w, h = width, height
        vw, vh = self.render.render_size()	# Match display resolution on retina screens
        if w is not None and h is not None:
            return (w, h)
        elif w is not None:
            # Choose height to match window aspect ratio.
            return (w, (vh * w) // vw)
        elif h is not None:
            # Choose width to match window aspect ratio.
            return ((vw * h) // vh, h)
        return (vw, vh)

    def report_framerate(self, report_rate, monitor_period=1.0, _minimum_render_time=None):
        '''
        Report a status message giving the current rendering rate in
        frames per second.  This is computed without the vertical sync
        which normally limits the frame rate to typically 60 frames
        per second.  The minimum drawing time used over a one second
        interval is used. The report_rate function is called with
        the frame rate in frames per second.
        '''
        if _minimum_render_time is None:
            self._framerate_callback = report_rate
            from time import time
            self._time_graphics = time() + monitor_period
            self.minimum_render_time = None
            self.render_start_time = None
            self.redraw_needed = True
        else:
            self._time_graphics = 0
            self._framerate_callback(1.0/_minimum_render_time)

    def _start_timing(self):
        if self._time_graphics:
            self.finish_rendering()
            from time import time
            self.render_start_time = time()

    def _finish_timing(self):
        if self._time_graphics:
            self.finish_rendering()
            from time import time
            t = time()
            rt = t - self.render_start_time
            mint = self.minimum_render_time
            if mint is None or rt < mint:
                self.minimum_render_time = mint = rt
            if t > self._time_graphics:
                self.report_framerate(None, _minimum_render_time = mint)
            else:
                self.redraw_needed = True

    def finish_rendering(self):
        '''
        Force the graphics pipeline to complete all requested drawing.
        This can slow down rendering but is used by display devices
        such as Oculus Rift goggles to reduce latency time between head
        tracking and graphics update.
        '''
        self._render.finish_rendering()

    def _compute_shadowmaps(self, opaque_drawings, transparent_drawings, camera):
        '''
        Compute shadow map textures for specified drawings.
        Does not include child drawings.
        '''
        r = self._render
        lp = r.lighting
        if not lp.shadows and lp.multishadow == 0:
            return False, False
        
        shadow_drawings = opaque_drawings
        mp = r.material
        if mp.transparent_cast_shadows:
            shadow_drawings += transparent_drawings
        if not mp.meshes_cast_shadows:
            shadow_drawings = [d for d in shadow_drawings if d.display_style != d.Mesh]

        shadow_enabled = r.shadow.use_shadow_map(camera, shadow_drawings)
        r.enable_shader_shadows(shadow_enabled)

        multishadow_enabled = r.multishadow.use_multishadow_map(shadow_drawings)
        r.enable_shader_multishadows(multishadow_enabled)
        
        return shadow_enabled, multishadow_enabled

    def max_multishadow(self):
        if not self._use_opengl():
            return 0	# OpenGL not available
        return self._render.multishadow.max_multishadows()

    def drawing_bounds(self, clip=False, cached_only=False, allow_drawing_changes=True):
        '''Return bounds of drawing, displayed part only.'''
        dm = self._drawing_manager
        if cached_only:
            return dm.cached_drawing_bounds
        if allow_drawing_changes:
            # Cause graphics update so bounds include changes in models.
            # TODO: This has been the source of many hard to debug problems.
            #  Some code calls drawing_bounds() and it has all kinds of side effects
            #  because of this check_for_drawing_changes() firing the "graphics update"
            #  trigger.  It would be a good idea to make drawing_bounds() never
            #  call this, or at least make allow_drawing_changes default False.
            #  It will require some close study to find the problems that change causes.
            self.check_for_drawing_change()
        b = dm.cached_drawing_bounds
        if b is None:
            dm.cached_drawing_bounds = b = self.drawing.bounds()
        if clip:
            planes = self.clip_planes.planes()
            if planes:
                # Clipping the bounding box does a poor giving tight bounds
                # or even bounds centered on the visible objects.  But handling
                # clip planes in bounds computations within models is more complex.
                from chimerax.geometry import clip_bounds
                b = clip_bounds(b, [(p.plane_point, p.normal) for p in planes])
        return b

    def _any_drawing_highlighted(self, drawings=None):
        '''Is anything highlighted.'''
        if drawings is None:
            dm = self._drawing_manager
            s = dm.cached_any_part_highlighted
            if s is None:
                dm.cached_any_part_highlighted = s = self.drawing.any_part_highlighted()
            return s
        else:
            for d in drawings:
                if d.any_part_highlighted():
                    return True
            return False

    def initial_camera_view(self, pad = 0.05, set_pivot = True):
        '''Set the camera position to show all displayed drawings,
        looking down the z axis.'''
        b = self.drawing_bounds()
        if b is None:
            return
        c = self.camera
        from chimerax.geometry import identity
        c.position = identity()
        c.view_all(b, window_size = self.window_size, pad = pad)
        if set_pivot:
            self._center_of_rotation = cr = b.center()
            self._update_center_of_rotation = True

    def view_all(self, bounds = None, pad = 0):
        '''Adjust the camera to show all displayed drawings using the
        current view direction.  If bounds is given then view is adjusted
        to show those bounds instead of the current drawing bounds.
        If pad is specified the fit is to a window size reduced by this fraction.
        '''
        if bounds is None:
            bounds = self.drawing_bounds()
            if bounds is None:
                return
        self.camera.view_all(bounds, window_size = self.window_size, pad = pad)
        if self._center_of_rotation_method in ('front center', 'center of view'):
            self._update_center_of_rotation = True

    def _get_cofr(self):
        if self._update_center_of_rotation:
            self._update_center_of_rotation = False
            cofr = self._compute_center_of_rotation()
            if not cofr is None:
                self._center_of_rotation = cofr
        return self._center_of_rotation
    def _set_cofr(self, cofr):
        from numpy import array, float32
        cofr_np = array(cofr, float32)	# TODO: temporary session save fix to handle tinyarray
        self._center_of_rotation = cofr_np
        self._center_of_rotation_method = 'fixed'
        self._update_center_of_rotation = False
    center_of_rotation = property(_get_cofr, _set_cofr)

    def _get_cofr_method(self):
        return self._center_of_rotation_method
    def _set_cofr_method(self, method):
        self._center_of_rotation_method = method
        self._update_center_of_rotation = True
    center_of_rotation_method = property(_get_cofr_method, _set_cofr_method)

    def _compute_center_of_rotation(self):
        '''
        Compute the center of rotation of displayed drawings.
        Use bounding box center if zoomed out, or the front center
        point if zoomed in.
        '''
        m = self._center_of_rotation_method
        if m == 'front center':
            p = self._front_center_cofr()
        elif m == 'fixed':
            p = self._center_of_rotation
        elif m == 'center of view':
            p = self._center_of_view_cofr()
        return p

    def _center_of_view_cofr(self):
        '''
        Keep the center of rotation in the middle of the view at a depth
        midway between near and far clip planes.  If only the near plane
        or only the far plane is enabled use center on that plane.  If neither
        near nor far planes are enabled then the depth is such that the
        previous rotation point and new rotation point are in the same plane
        perpendicular to the new view direction.
        '''
        p = self.clip_planes
        np, fp = p.find_plane('near'), p.find_plane('far')
        if np and fp:
            cr = 0.5 * (np.plane_point + fp.plane_point)
        elif np:
            cr = np.plane_point
        elif fp:
            cr = fp.plane_point
        else:
            # No clip planes.
            # Keep the center of rotation in the middle of the view at a depth
            # such that the new and previous center of rotation are in the same
            # plane perpendicular to the camera view direction.
            cr = self._center_point_matching_depth(self._center_of_rotation)

        return cr

    def _center_point_matching_depth(self, point):
        cam_pos = self.camera.position.origin()
        vd = self.camera.view_direction()
        hyp = point - cam_pos
        from chimerax.geometry import inner_product, norm
        distance = inner_product(hyp, vd)
        cr = cam_pos + distance*vd
        old_cofr = self._center_of_rotation
        if norm(cr - old_cofr) < 1e-6 * distance:
                # Avoid jitter if camera has not moved
                cr = old_cofr
        return cr

    def set_rotation_depth(self, point):
        '''
        Set center of rotation in middle of window at depth matching the
        depth along camera axis of the specified point.
        '''
        self._center_of_rotation = self._center_point_matching_depth(point)
        
    def _front_center_cofr(self):
        '''
        Compute the center of rotation of displayed drawings.
        Use bounding box center if zoomed out, or the front center
        point if zoomed in.
        '''
        b = self.drawing_bounds(allow_drawing_changes = False)
        if b is None:
            return
        vw = self.camera.view_width(b.center())
        if vw is None or vw >= b.width():
            # Use center of drawings for zoomed out views
            cr = b.center()
        else:
            # Use front center point for zoomed in views
            cr = self._front_center_point()	# Can be None
            if cr is None:
                # No objects in center of view, so keep the depth the same
                # but move center to keep it in the center of view.
                cr = self._center_point_matching_depth(self._center_of_rotation)
        return cr

    def _front_center_point(self):
        w, h = self.window_size
        p = self.picked_object(0.5 * w, 0.5 * h, max_transparent_layers = 0, exclude=View.unpickable)
        return p.position if p else None

    unpickable = lambda drawing: not drawing.pickable

    def picked_object(self, win_x, win_y, exclude=unpickable, beyond=None,
                      max_transparent_layers=3):
        '''
        Return a Pick object for the frontmost object below the given
        screen window position (specified in pixels).  This Pick object will
        have an attribute position giving the point where the intercept occurs.
        This is used when hovering the mouse over an object (e.g. an atom)
        to get a description of that object.  Beyond is minimum distance
        as fraction from front to rear clip plane.
        '''
        xyz1, xyz2 = self.clip_plane_points(win_x, win_y)
        if xyz1 is None or xyz2 is None:
            p = None
        else:
            p = self.picked_object_on_segment(xyz1, xyz2, exclude = exclude, beyond = beyond,
                                              max_transparent_layers = max_transparent_layers)

        # If scene clipping and some models disable clipping, try picking those.
        if self.clip_planes.have_scene_plane() and not self.drawing.all_allow_clipping():
            ucxyz1, ucxyz2 = self.clip_plane_points(win_x, win_y, include_scene_clipping = False)
            if ucxyz1 is not None and ucxyz2 is not None:
                def exclude_clipped(d, exclude=exclude):
                    return exclude(d) or d.allow_clipping
                ucp = self.picked_object_on_segment(ucxyz1, ucxyz2,
                                                    max_transparent_layers = max_transparent_layers,
                                                    exclude = exclude_clipped)
                if ucp:
                    from chimerax.geometry import distance
                    if p is None or ucp.distance * distance(ucxyz1, ucxyz2) < distance(ucxyz1, xyz1):
                        p = ucp
            
        return p

    def picked_object_on_segment(self, xyz1, xyz2, exclude=unpickable, beyond=None,
                                 max_transparent_layers=3):
        '''
        Return a Pick object for the first object along line segment from xyz1
        to xyz2 in specified in scene coordinates. This Pick object will
        have an attribute position giving the point where the intercept occurs.
        Beyond is minimum distance as fraction (0-1) along the segment.
        '''
    
        if beyond is not None:
            fb = beyond + 1e-5
            xyz1 = (1-fb)*xyz1 + fb*xyz2

        p = self.drawing.first_intercept(xyz1, xyz2, exclude=exclude)
        if p is None:
            return None
        
        if max_transparent_layers > 0:
            if getattr(p, 'pick_through', False) and p.distance is not None:
                p2 = self.picked_object_on_segment(xyz1, xyz2, exclude=exclude, beyond=p.distance,
                                                   max_transparent_layers = max_transparent_layers-1)
                if p2:
                    p = p2
            
        f = p.distance
        p.position = (1.0 - f) * xyz1 + f * xyz2

        if beyond:
            # Correct distance fraction to refer to clip planes.
            p.distance = fb + f*(1-fb)
            
        return p

    def rectangle_pick(self, win_x1, win_y1, win_x2, win_y2, exclude=unpickable):
        '''
        Return a Pick object for the objects in the rectangle having
        corners at the given screen window position (specified in pixels).
        '''
        # Compute planes bounding view through rectangle.
        planes = self.camera.rectangle_bounding_planes((win_x1, win_y1), (win_x2, win_y2),
                                                       self.window_size)
        if len(planes) == 0:
            return []	# Camera does not support computation of bounding planes.

        # Use clip planes.
        cplanes = self.clip_planes.planes()
        if cplanes:
            from numpy import concatenate, array, float32
            all_planes = concatenate((planes, array([cp.opengl_vec4() for cp in cplanes], float32)))
        else:
            all_planes = planes

        # If scene clipping and some models disable clipping, try picking those.
        if self.clip_planes.have_scene_plane() and not self.drawing.all_allow_clipping():
            def exclude_unclipped(d, exclude=exclude):
                return exclude(d) or not d.allow_clipping
            cpicks = self.drawing.planes_pick(all_planes, exclude=exclude_unclipped)
            def exclude_clipped(d, exclude=exclude):
                return exclude(d) or d.allow_clipping
            upicks = self.drawing.planes_pick(planes, exclude=exclude_clipped)
            picks = cpicks + upicks
        else:
            picks = self.drawing.planes_pick(all_planes, exclude=exclude)
            
        return picks

    def _update_projection(self, camera, view_num):

        r = self._render
        ww, wh = r.render_size()
        if ww == 0 or wh == 0:
            return

        if r.recording_opengl:
            from .gllist import ProjectionCalc
            nfp = ProjectionCalc(self, view_num, (ww,wh))
            near_far, pm = nfp.near_far, nfp.projection_matrix
            n,f = near_far()
            pnf = 1 if camera.name == 'orthographic' else (n / f)
        else:
            near_far = self.near_far_distances(camera, view_num)
            # TODO: Different camera views need to use same near/far if they are part of
            # a cube map, otherwise depth cue dimming is not continuous across cube faces.
            pm = camera.projection_matrix(near_far, view_num, (ww, wh))
            pnf = 1 if camera.name == 'orthographic' else (near_far[0] / near_far[1])

        self.silhouette.perspective_near_far_ratio = pnf

        r.set_projection_matrix(pm)
        r.set_near_far_clip(near_far)	# Used by depth cue

    def near_far_distances(self, camera, view_num, include_clipping = True):
        '''Near and far scene bounds as distances from camera.'''
        cp = camera.get_position(view_num).origin()
        vd = camera.view_direction(view_num)
        near, far = self._near_far_bounds(cp, vd)
        if include_clipping:
            p = self.clip_planes
            np, fp = p.find_plane('near'), p.find_plane('far')
            from chimerax.geometry import inner_product
            if np:
                near = max(near, inner_product(vd, (np.plane_point - cp)))
            if fp:
                far = min(far, inner_product(vd, (fp.plane_point - cp)))
        cnear, cfar = self._clamp_near_far(near, far)
        return cnear, cfar

    def _near_far_bounds(self, camera_pos, view_dir):
        b = self.drawing_bounds(allow_drawing_changes = False)
        if b is None:
            return self._min_near_fraction, 1  # Nothing shown
        from chimerax.geometry import inner_product
        d = inner_product(b.center() - camera_pos, view_dir)         # camera to center of drawings
        r = (1 + self._near_far_pad) * b.radius()
        return (d-r, d+r)

    def _clamp_near_far(self, near, far):
        # Clamp near clip > 0.
        near_min = self._min_near_fraction * (far - near) if far > near else 1
        near = max(near, near_min)
        if far <= near:
            far = 2 * near
        return (near, far)

    def clip_plane_points(self, window_x, window_y, camera=None, view_num=None,
                          include_scene_clipping = True):
        '''
        Return two scene points at the near and far clip planes at
        the specified window pixel position.  The points are in scene
        coordinates.  '''
        c = camera if camera else self.camera
        origin, direction = c.ray(window_x, window_y, self.window_size)	# Scene coords
        if origin is None:
            return (None, None)

        near, far = self.near_far_distances(c, view_num, include_clipping = False)
        cplanes = [(origin + near*direction, direction), 
                   (origin + far*direction, -direction)]
        if include_scene_clipping:
            cplanes.extend((p.plane_point, p.normal) for p in self.clip_planes.planes())
        from chimerax.geometry import ray_segment
        f0, f1 = ray_segment(origin, direction, cplanes)
        if f1 is None or f0 > f1:
            return (None, None)
        scene_pts = (origin + f0*direction, origin + f1*direction)
        return scene_pts

    def win_coord(self, pt, camera=None, view_num=None):
        """Convert world coordinate to window coordinate"""
        # TODO: extend to handle numpy array of points
        c = self.camera if camera is None else camera
        near_far = self.near_far_distances(c, view_num)
        pm = c.projection_matrix(near_far, view_num, self.window_size)
        inv_position = c.position.inverse().opengl_matrix()
        from numpy import array, float32, concatenate
        xpt = concatenate((pt, [1])) @ inv_position @ pm
        width, height = self.window_size
        win_pt = array([
            (xpt[0] + 1) * width / 2,
            (xpt[1] + 1) * height / 2,
            (xpt[2] + 1) / 2
        ], dtype=float32)
        return win_pt

    def rotate(self, axis, angle, drawings=None):
        '''
        Move camera to simulate a rotation of drawings about current
        rotation center.  Axis is in scene coordinates and angle is
        in degrees.
        '''
        if drawings:
            from chimerax.geometry import bounds
            b = bounds.union_bounds(d.bounds() for d in drawings)
            if b is None:
                return
            center = b.center()
        else:
            center = self.center_of_rotation
        from chimerax.geometry import rotation
        r = rotation(axis, angle, center)
        self.move(r, drawings)

    def translate(self, shift, drawings=None, move_near_far_clip_planes = False):
        '''Move camera to simulate a translation of drawings.  Translation
        is in scene coordinates.'''
        if shift[0] == 0 and shift[1] == 0 and shift[2] == 0:
            return
        if self._center_of_rotation_method in ('front center', 'center of view'):
            self._update_center_of_rotation = True
        if not move_near_far_clip_planes:
            # Near and far clip planes are fixed to camera.
            # Move them so they stay fixed relative to models.
            self._shift_near_far_clip_planes(shift)
        from chimerax.geometry import translation
        t = translation(shift)
        self.move(t, drawings)

    def _shift_near_far_clip_planes(self, shift):
        p = self.clip_planes
        np, fp = p.find_plane('near'), p.find_plane('far')
        if np or fp:
            vd = self.camera.view_direction()
            from chimerax.geometry import inner_product
            plane_shift = inner_product(shift,vd)*vd
            if np:
                np.plane_point += plane_shift
            if fp:
                fp.plane_point += plane_shift
        
    def move(self, tf, drawings=None):
        '''
        Move camera to simulate a motion of drawings.
        Transform is in scene coordinates.
        '''
        if drawings is None:
            c = self.camera
            c.position = tf.inverse() * c.position
        else:
            for d in drawings:
                d.scene_position = tf * d.scene_position

    def pixel_size(self, p=None):
        "Return the pixel size in scene length units at point p in the scene."
        if p is None:
            # Don't recompute center of rotation as that can be slow.
            p = self._center_of_rotation
            if p is None:
                p = self.center_of_rotation	# Compute center of rotation
        return self.camera.view_width(p) / self.window_size[0]

    def stereo_scaling(self, delta_z):
        '''
        If in stereo camera mode change eye separation so that
        when models moved towards camera by delta_z, their center
        of bounding box appears to stay at the same depth, giving
        the appearance that the models were simply scaled in size.
        Another way to understand this is the models are scaled
        when measured as a multiple of stereo eye separation.
        '''
        c = self.camera
        if not hasattr(c, 'eye_separation_scene'):
            return
        b = self.drawing_bounds()
        if b is None:
            return
        from chimerax.geometry import distance
        d = distance(b.center(), c.position.origin())
        if d == 0 and delta_z > 0.5*d:
            return
        f = 1 - delta_z / d
        from math import exp
        c.eye_separation_scene *= f
        c.redraw_needed = True

class _RedrawNeeded:

    def __init__(self):
        self.redraw_needed = False
        self.shape_changed = True
        self.shadow_shape_change = False
        self.transparency_changed = False
        self.cached_drawing_bounds = None
        self.cached_any_part_highlighted = None

    def __call__(self, drawing, shape_changed=False, highlight_changed=False, transparency_changed=False):
        self.redraw_needed = True
        if shape_changed:
            self.shape_changed = True
            if drawing.casts_shadows:
                self.shadow_shape_change = True
            if not getattr(drawing, 'skip_bounds', False):
                self.cached_drawing_bounds = None
        if transparency_changed:
            self.transparency_changed = True
        if highlight_changed:
            self.cached_any_part_highlighted = None

    def shadows_changed(self):
        if self.transparency_changed:
            return True
        return self.shadow_shape_change

    def clear_changes(self):
        self.redraw_needed = False
        self.shape_changed = False
        self.shadow_shape_change = False
        self.transparency_changed = False
