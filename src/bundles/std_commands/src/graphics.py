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

def graphics(session, bg_color=None, background_color=None):
    '''Set graphics parameters.  With no options reports the current settings.

    Parameters
    ----------
    background_color : Color
        Set the graphics window background color.
    bg_color : Color
        Synonym for background_color.
    '''
    had_arg = False
    view = session.main_view
    bgc = (background_color or bg_color)
    if bgc is not None:
        had_arg = True
        view.background_color = bgc.rgba
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

def graphics_quality(session, quality = None, subdivision = None,
                     atom_triangles = None, bond_triangles = None,
                     total_atom_triangles = None, total_bond_triangles = None,
                     bond_sides = None, pseudobond_sides = None,
                     ribbon_divisions = None, ribbon_sides = None, color_depth = None):
    '''
    Set graphics quality parameters.

    Parameters
    ----------
    quality : float
        Controls the rendering quality of spheres and cylinders for drawing atoms and bonds.
        Default value is 1, higher values give smoother spheres and cylinders by increasing
        the maximum number of triangles rendered by the quality factor.
    subdivision : float
        Deprecated.  Same as quality.
    atom_triangles : integer or "default"
        Number of triangles for drawing an atom sphere, minimum 4.
        If 0, then automatically compute number of triangles.
    bond_triangles : integer or "default"
        Number of triangles for drawing a bond cylinder, minimum 12.
        If 0, then automatically compute number of triangles.
    total_atom_triangles : integer
        Target number of triangles for all shown atoms when automatically
        triangles per atom.
    total_bond_triangles : integer
        Target number of triangles for all shown bonds when automatically
        triangles per bond.
    bond_sides : integer
        Number of sides for a bond cylinder, minimum 3.
    pseudobond_sides : integer
        Number of sides for a pseudobond cylinder, minimum 3.
    ribbon_divisions : integer or "default"
        Number of segments to use for one residue of a ribbon, minimum 2.
        If default then automatically determine value (20 for less than 20,000 residues).
    ribbon_sides : integer
        Number of segments to use around circumference of ribbon, minimum 4 (default 12).
    color_depth : 8 or 16
        Number of bits per color channel (red, green, blue, alpha) in framebuffer.
        If 16 is specified then offscreen rendering is used since it is not easy or
        possible to switch on-screen framebuffer depth.
    '''
    from chimerax.atomic import structure_graphics_updater
    gu = structure_graphics_updater(session)
    lod = gu.level_of_detail
    change = False
    from chimerax.core.errors import UserError
    if subdivision is not None:
        quality = subdivision
    if quality is not None:
        gu.set_quality(quality)
        change = True
    if atom_triangles is not None:
        if isinstance(atom_triangles, int) and atom_triangles != 0 and atom_triangles < 4:
            raise UserError('Minimum number of atom triangles is 4')
        lod.atom_fixed_triangles =  None if atom_triangles in ('default', 0) else atom_triangles
        change = True
    if bond_triangles is not None:
        if isinstance(bond_triangles, int) and bond_triangles != 0 and bond_triangles < 12:
            raise UserError('Minimum number of bond triangles is 12')
        lod.bond_fixed_triangles = None if bond_triangles in ('default', 0) else bond_triangles
        change = True
    if bond_sides is not None:
        if isinstance(bond_sides, int) and bond_sides != 0 and bond_sides < 3:
            raise UserError('Minimum number of bond sides is 3')
        lod.bond_fixed_triangles = None if bond_sides in ('default', 0) else 4*bond_sides
        change = True
    if total_atom_triangles is not None:
        lod.total_atom_triangles = total_atom_triangles
        change = True
    if total_bond_triangles is not None:
        lod.total_bond_triangles = total_bond_triangles
        change = True
    if pseudobond_sides is not None:
        if pseudobond_sides < 3:
            raise UserError('Minimum number of pseudobond sides is 3')
        lod.pseudobond_sides = pseudobond_sides
        from chimerax.atomic import all_pseudobond_groups
        for pbg in all_pseudobond_groups(session):
            pbg.update_cylinder_sides()
        change = True
    if ribbon_divisions is not None:
        if isinstance(ribbon_divisions, int) and ribbon_divisions != 0 and ribbon_divisions < 2:
            raise UserError('Minimum number of ribbon divisions is 2')
        div = None if ribbon_divisions in (0, 'default') else ribbon_divisions
        gu.set_ribbon_divisions(div)
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
        msg = ('Quality %.3g, atom triangles %d, bond triangles %d, pseudobond sides %d' %
               (lod.quality, lod.atom_sphere_triangles(na), lod.bond_cylinder_triangles(na), lod.pseudobond_sides))
        div = [lod.ribbon_divisions(s.num_ribbon_residues) for s in gu.structures]
        if div:
            dmin, dmax = min(div), max(div)
            drange = '%d-%d' % (dmin, dmax) if dmin < dmax else '%d' % dmin
            msg += ', ribbon divisions %s' % drange
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
    from chimerax.core.commands import CmdDesc, register, IntArg, FloatArg, BoolArg, ColorArg, TopModelsArg, Or, EnumOf

    desc = CmdDesc(
        keyword=[('background_color', ColorArg),
                 ('bg_color', ColorArg)],
        hidden = ['background_color'],	# Deprecated in favor of bg_color
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

    IntOrDefaultArg = Or(EnumOf(['default']), IntArg)
    desc = CmdDesc(
        optional = [('quality', FloatArg)],
        keyword=[('subdivision', FloatArg),
                 ('atom_triangles', IntOrDefaultArg),
                 ('bond_triangles', IntOrDefaultArg),
                 ('total_atom_triangles', IntArg),
                 ('total_bond_triangles', IntArg),
                 ('bond_sides', IntOrDefaultArg),
                 ('pseudobond_sides', IntArg),
                 ('ribbon_divisions', IntOrDefaultArg),
                 ('ribbon_sides', IntArg),
                 ('color_depth', IntArg),
                 ],
        hidden = ['subdivision'], # Deprecated in favor of quality
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
        synopsis="set selection outline parameters"
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
