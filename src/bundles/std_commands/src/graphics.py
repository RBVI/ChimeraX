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

def graphics(session, atom_triangles = None, bond_triangles = None,
             total_atom_triangles = None, total_bond_triangles = None,
             ribbon_divisions = None, ribbon_sides = None, max_frame_rate = None,
             frame_rate = None, wait_for_vsync = None):
    '''
    Set graphics rendering parameters.

    Parameters
    ----------
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
    max_frame_rate : float
        Set maximum graphics frame rate (default 60).
    frame_rate : bool
        Whether to show status message with average frame rate each second.
    wait_for_vsync : bool
        Whether drawing is synchronized to the display vertical refresh rate,
        typically 60 Hz.  Disabling wait allows frame rates faster than vsync
        but can exhibit image tearing.  Currently only supported on Windows.
    '''
    from chimerax.atomic.structure import structure_graphics_updater
    gu = structure_graphics_updater(session)
    lod = gu.level_of_detail
    change = False
    from chimerax.core.errors import UserError
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
    if max_frame_rate is not None:
        msec = 1000.0 / max_frame_rate
        session.update_loop.set_redraw_interval(msec)
        change = True
    if frame_rate is not None:
        show_frame_rate(session, frame_rate)
        change = True
    if wait_for_vsync is not None:
        r = session.main_view.render
        r.make_current()
        if not r.wait_for_vsync(wait_for_vsync):
            session.logger.warning('Changing wait for vsync is only supported on Windows by some drivers')
        change = True

    if change:
        gu.update_level_of_detail()
    else:
        na = gu.num_atoms_shown
        if session.ui.is_gui:
            msec = session.update_loop.redraw_interval
            rate = 1000.0 / msec if msec > 0 else 1000.0
        else:
            rate = 0
        msg = ('Atom triangles %d, bond triangles %d, ribbon divisions %d, max framerate %.3g' %
               (lod.atom_sphere_triangles(na), lod.bond_cylinder_triangles(na), lod.ribbon_divisions,
                rate))
        session.logger.status(msg, log = True)
    
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
    from chimerax.core.commands import CmdDesc, register, IntArg, FloatArg, BoolArg, TopModelsArg
    desc = CmdDesc(
        keyword=[('atom_triangles', IntArg),
                 ('bond_triangles', IntArg),
                 ('total_atom_triangles', IntArg),
                 ('total_bond_triangles', IntArg),
                 ('ribbon_divisions', IntArg),
                 ('ribbon_sides', IntArg),
                 ('max_frame_rate', FloatArg),
                 ('frame_rate', BoolArg),
                 ('wait_for_vsync', BoolArg),
                 ],
        synopsis='Set graphics rendering parameters'
    )
    register('graphics', desc, graphics, logger=logger)

    desc = CmdDesc(synopsis='Restart graphics drawing after an error')
    register('graphics restart', desc, graphics_restart, logger=logger)

    desc = CmdDesc(optional=[('models', TopModelsArg)],
                   synopsis='Report triangles in graphics scene')
    register('graphics triangles', desc, graphics_triangles, logger=logger)

    desc = CmdDesc(synopsis='Report graphics driver version info')
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
    
def graphics_driver(session):
    '''
    Report opengl graphics driver info.
    '''
    info = session.logger.info
    if session.ui.is_gui:
        r = session.main_view.render
        r.make_current()
        info('OpenGL version: ' + r.opengl_version())
        info('OpenGL renderer: ' + r.opengl_renderer())
        info('OpenGL vendor: ' + r.opengl_vendor())
    else:
        info('OpenGL info not available in nogui mode.')
