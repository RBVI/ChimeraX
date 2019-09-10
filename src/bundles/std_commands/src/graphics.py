# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def graphics(session, background_color=None):
    '''Set graphics parameters.  With no options reports the current settings.

    Parameters
    ----------
    background_color : Color
        Set the graphics window background color.
    '''
    had_arg = False
    view = session.main_view
    if background_color is not None:
        had_arg = True
        view.background_color = background_color.rgba
        view.redraw_needed = True

    if not had_arg:
        msg = 'Background color: %d,%d,%d' % tuple(100*r for r in view.background_color[:3])
        session.logger.info(msg)

def graphics_rate(session, report_frame_rate = None,
                  max_frame_rate = None, wait_for_vsync = None):
    '''
    Set graphics rendering rate parameters.

    Parameters
    ----------
    report_frame_rate : bool
        Whether to show status message with average frame rate each second.
    max_frame_rate : float
        Set maximum graphics frame rate (default 60).
    wait_for_vsync : bool
        Whether drawing is synchronized to the display vertical refresh rate,
        typically 60 Hz.  Disabling wait allows frame rates faster than vsync
        but can exhibit image tearing.  Currently only supported on Windows.
    '''

    change = False
    if max_frame_rate is not None:
        msec = 1000.0 / max_frame_rate
        session.update_loop.set_redraw_interval(msec)
        change = True
    if report_frame_rate is not None:
        show_frame_rate(session, report_frame_rate)
        change = True
    if wait_for_vsync is not None:
        r = session.main_view.render
        r.make_current()
        if not r.wait_for_vsync(wait_for_vsync):
            session.logger.warning('Changing wait for vsync is only supported on Windows by some drivers')
        change = True

    if not change and session.ui.is_gui:
        msec = session.update_loop.redraw_interval
        rate = 1000.0 / msec if msec > 0 else 1000.0
        msg = ('Maximum framerate %.3g' % rate)
        session.logger.status(msg, log = True)

def graphics_quality(session, subdivision = None,
                     atom_triangles = None, bond_triangles = None,
                     total_atom_triangles = None, total_bond_triangles = None,
                     ribbon_divisions = None, ribbon_sides = None, color_depth = None):
    '''
    Set graphics quality parameters.

    Parameters
    ----------
    subdivision : float
        Controls the rendering quality of spheres and cylinders for drawing atoms and bonds.
        Default value is 1, higher values give smoother spheres and cylinders.
    atom_triangles : integer
        Number of triangles for drawing an atom sphere, minimum 4.
        If 0, then automatically compute number of triangles.
    bond_triangles : integer
        Number of triangles for drawing an atom sphere, minimum 12.
        If 0, then automatically compute number of triangles.
    total_atom_triangles : integer
        Target number of triangles for all shown atoms when automatically
        triangles per atom.
    total_bond_triangles : integer
        Target number of triangles for all shown bonds when automatically
        triangles per bond.
    ribbon_divisions : integer
        Number of segments to use for one residue of a ribbon, minimum 2 (default 20).
    ribbon_sides : integer
        Number of segments to use around circumference of ribbon, minimum 4 (default 12).
    color_depth : 8 or 16
        Number of bits per color channel (red, green, blue, alpha) in framebuffer.
        If 16 is specified then offscreen rendering is used since it is not easy or
        possible to switch on-screen framebuffer depth.
    '''
    from chimerax.atomic.structure import structure_graphics_updater
    gu = structure_graphics_updater(session)
    lod = gu.level_of_detail
    change = False
    from chimerax.core.errors import UserError
    if subdivision is not None:
        from chimerax import atomic
        atomic.structure_graphics_updater(session).set_subdivision(subdivision)
        change = True
    if atom_triangles is not None:
        if atom_triangles != 0 and atom_triangles < 4:
            raise UserError('Minimum number of atom triangles is 4')
        lod.atom_fixed_triangles = atom_triangles if atom_triangles > 0 else None
        change = True
    if bond_triangles is not None:
        if bond_triangles != 0 and bond_triangles < 12:
            raise UserError('Minimum number of bond triangles is 12')
        lod.bond_fixed_triangles = bond_triangles if bond_triangles > 0 else None
        change = True
    if total_atom_triangles is not None:
        lod.total_atom_triangles = total_atom_triangles
        change = True
    if total_bond_triangles is not None:
        lod.total_bond_triangles = total_bond_triangles
        change = True
    if ribbon_divisions is not None:
        if ribbon_divisions < 2:
            raise UserError('Minimum number of ribbon divisions is 2')
        lod.ribbon_divisions = ribbon_divisions
        change = True
    if ribbon_sides is not None:
        if ribbon_sides < 4:
            raise UserError('Minimum number of ribbon sides is 4')
        from .cartoon import cartoon_style
        cartoon_style(session, sides = 2*(ribbon_sides//2))
        change = True
    if color_depth is not None:
        if color_depth not in (8, 16):
            from chimerax.core.errors import UserError
            raise UserError('Only color depths 8 or 16 allowed, got %d' % color_depth)
        v = session.main_view
        r = v.render
        r.set_offscreen_color_bits(color_depth)
        r.offscreen.enabled = (color_depth == 16)
        v.redraw_needed = True
        change = True

    if change:
        gu.update_level_of_detail()
    else:
        na = gu.num_atoms_shown
        msg = ('Subdivision %.3g, atom triangles %d, bond triangles %d, ribbon divisions %d' %
               (lod.quality, lod.atom_sphere_triangles(na), lod.bond_cylinder_triangles(na), lod.ribbon_divisions))
        session.logger.status(msg, log = True)

def graphics_silhouettes(session, enable=None, width=None, color=None, depth_jump=None):
    '''Set graphics silhouette parameters.  With no options reports the current settings.

    Parameters
    ----------
    enable : bool
        Enable or disable silhouette edges (black lines drawn at boundaries of objects
        where the depth of the scene jumps.
    width : float
        Width in pixels of silhouette edges. Minimum width is 1 pixel.
    color : Color
        Color of silhouette edges.
    depth_jump : float
        Fraction of scene depth giving minimum depth change to draw a silhouette edge. Default 0.03.
    '''
    had_arg = False
    view = session.main_view
    silhouette = view.silhouette
    if enable is not None:
        had_arg = True
        silhouette.enabled = enable
        view.redraw_needed = True
    if width is not None:
        had_arg = True
        silhouette.thickness = width
        view.redraw_needed = True
    if color is not None:
        had_arg = True
        silhouette.color = color.rgba
        view.redraw_needed = True
    if depth_jump is not None:
        had_arg = True
        silhouette.depth_jump = depth_jump
        view.redraw_needed = True

    if not had_arg:
        msg = '\n'.join(('Current silhouette settings:',
                         '  enabled: ' + str(silhouette.enabled),
                         '  width: %.3g' % silhouette.thickness,
                         '  color: %d,%d,%d' % tuple(100*r for r in silhouette.color[:3]),
                         '  depth jump: %.3g' % silhouette.depth_jump))
        session.logger.info(msg)


def graphics_selection(session, color=None, width=None):
    '''Set selection outline parameters.  With no options reports the current settings.

    Parameters
    ----------
    color : Color
        Color to use when outlining selected objects.  Initially green.
    width : float
        Width in pixels of the selection outline.  Initially 1 (or 2 for high DPI screens).
    '''
    had_arg = False
    view = session.main_view
    if color is not None:
        had_arg = True
        view.highlight_color = color.rgba
    if width is not None:
        had_arg = True
        view.highlight_thickness = width

    if not had_arg:
        msg = '\n'.join(('Current selection outline settings:',
                         '  color: %d,%d,%d' % tuple(100*r for r in view.highlight_color[:3]),
                         '  width: %.3g' % view.highlight_thickness))
        session.logger.info(msg)
        
def graphics_restart(session):
    '''
    Restart graphics rendering after it has been stopped due to an error.
    Usually the errors are graphics driver bugs, and the graphics is stopped
    to avoid a continuous stream of errors. Restarting does not fix the underlying
    problem and it is usually necessary to turn off whatever graphics features
    (e.g. ambient shadoows) that led to the graphics error before restarting.
    '''
    session.update_loop.unblock_redraw()

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, IntArg, FloatArg, BoolArg, ColorArg, TopModelsArg

    desc = CmdDesc(
        keyword=[('background_color', ColorArg)],
        synopsis="set graphics parameters"
    )
    register('graphics', desc, graphics, logger=logger)

    desc = CmdDesc(
        optional = [('report_frame_rate', BoolArg)],
        keyword=[('max_frame_rate', FloatArg),
                 ('wait_for_vsync', BoolArg),
                 ],
        synopsis='Set graphics rendering rate parameters'
    )
    register('graphics rate', desc, graphics_rate, logger=logger)

    desc = CmdDesc(
        keyword=[('subdivision', FloatArg),
                 ('atom_triangles', IntArg),
                 ('bond_triangles', IntArg),
                 ('total_atom_triangles', IntArg),
                 ('total_bond_triangles', IntArg),
                 ('ribbon_divisions', IntArg),
                 ('ribbon_sides', IntArg),
                 ],
        synopsis='Set graphics quality parameters'
    )
    register('graphics quality', desc, graphics_quality, logger=logger)

    desc = CmdDesc(
        optional=[('enable', BoolArg)],
        keyword=[('width', FloatArg),
                 ('color', ColorArg),
                 ('depth_jump', FloatArg)],
        synopsis="set silhouette parameters"
    )
    register('graphics silhouettes', desc, graphics_silhouettes, logger=logger)

    desc = CmdDesc(
        keyword=[('width', FloatArg),
                 ('color', ColorArg)],
        synopsis="set selewction outline parameters"
    )
    register('graphics selection', desc, graphics_selection, logger=logger)

    desc = CmdDesc(synopsis='Restart graphics drawing after an error')
    register('graphics restart', desc, graphics_restart, logger=logger)

    desc = CmdDesc(optional=[('models', TopModelsArg)],
                   synopsis='Report triangles in graphics scene')
    register('graphics triangles', desc, graphics_triangles, logger=logger)

    desc = CmdDesc(synopsis='Report graphics driver version info',
                   keyword=[('verbose', BoolArg)])
    register('graphics driver', desc, graphics_driver, logger=logger)
    
def graphics_triangles(session, models = None):
    '''
    Report shown triangles in graphics scene.  This is for analyzing graphics performance.
    '''
    if models is None:
        models = [m for m in session.models.list() if len(m.id) == 1 and m.display]

    lines = []
    tri = _drawing_triangles(models, lines)

    tot = '%d total triangles in %d models and %d drawings' % (tri, len(models), len(lines))
    session.logger.status(tot)
    lines.insert(0, tot)
    msg = '<pre style="font-family: serif">\n%s\n</pre>' % '\n'.join(lines)
    session.logger.info(msg, is_html = True)

def _drawing_triangles(drawings, lines, indent = ''):
    tri = 0
    from chimerax.core.models import Model
    for d in drawings:
        dtri = d.number_of_triangles(displayed_only = True)
        tri += dtri
        ninst = d.number_of_positions(displayed_only = True)
        name = '#%s %s' % (d.id_string, d.name) if isinstance(d, Model) else d.name
        line = '%s%s %d' % (indent, name, dtri)
        if ninst > 1:
            line += ' in %d instances, %d each' % (ninst, dtri//ninst)
        lines.append(line)
        _drawing_triangles(d.child_drawings(), lines, indent + '  ')
    return tri
    
def show_frame_rate(session, show):
    frr = getattr(session, '_frame_rate_reporter', None)
    if show:
        if frr is None:
            session._frame_rate_reporter = frr = FrameRateReporter(session)
        frr.report(True)
    elif frr:
        frr.report(False)

# Report frame rate.
class FrameRateReporter:
    def __init__(self, session):
        self.session = session
        self.report_interval = 1.0	# seconds
        self._new_frame_handler = None
        self._frame_drawn_handler = None
        self._num_frames = 0
        self._last_time = None
        self._cumulative_times = {'draw_time': 0,
                                  'new_frame_time': 0,
                                  'atomic_check_for_changes_time': 0,
                                  'drawing_change_time': 0,
                                  'clip_time': 0}
    def report(self, enable):
        t = self.session.triggers
        if enable:
            if self._new_frame_handler is None:
                self._new_frame_handler = t.add_handler('new frame', self.new_frame_cb)
                self._frame_drawn_handler = t.add_handler('frame drawn', self.frame_drawn_cb)
        else:
            nfh, fdh = self._new_frame_handler, self._frame_drawn_handler
            if nfh:
                t.remove_handler(nfh)
            if fdh:
                t.remove_handler(fdh)
            self._new_frame_handler = self._frame_drawn_handler = None
            
    def new_frame_cb(self, *ignore):
        # Make frames draw continuously even if scene does not change
        self.session.main_view.redraw_needed = True

    def frame_drawn_cb(self, *ignore):
        from time import time
        t = time()
        lt = self._last_time
        if lt is None:
            self._last_time = t
            self._num_frames = 0
            return
        self.record_times()
        self._num_frames += 1
        nf = self._num_frames
        dt = t - lt
        if dt > self.report_interval:
            fps = nf / dt
            msg = '%.1f frames per second' % fps
            ct = self._cumulative_times
            msg += ', ' + ', '.join(['%s %.0f%%' % (k[:-5], 100*v/dt) for k,v in ct.items()])
            self.session.logger.status(msg)
            self._last_time = t
            self._num_frames = 0
            ct = self._cumulative_times
            for k in ct.keys():
                ct[k] = 0

    def record_times(self):
        u = self.session.update_loop
        ct = self._cumulative_times
        for k in ct.keys():
            ct[k] += getattr(u, 'last_'+k)
    
def graphics_driver(session, verbose = False):
    '''
    Report opengl graphics driver info.
    '''
    info = session.logger.info
    if session.ui.is_gui:
        r = session.main_view.render
        r.make_current()
        if verbose:
            info(r.opengl_info())
        else:
            lines = ['OpenGL version: ' + r.opengl_version(),
                     'OpenGL renderer: ' + r.opengl_renderer(),
                     'OpenGL vendor: ' + r.opengl_vendor()]
            info('\n'.join(lines))
    else:
        info('OpenGL info not available in nogui mode.')
