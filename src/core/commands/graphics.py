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
             ribbon_divisions = None, ribbon_sides = None, max_frame_rate = None,
             frame_rate = None):
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
    ribbon_divisions : integer
        Number of segments to use for one residue of a ribbon, minimum 2 (default 20).
    ribbon_sides : integer
        Number of segments to use around circumference of ribbon, minimum 4 (default 12).
    max_frame_rate : float
        Set maximum graphics frame rate (default 60).
    frame_rate : bool
        Whether to show status message with average frame rate each second.
    '''
    from ..atomic.structure import structure_graphics_updater
    gu = structure_graphics_updater(session)
    lod = gu.level_of_detail
    change = False
    from ..errors import UserError
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
        session.ui.main_window.graphics_window.set_redraw_interval(msec)
        change = True
    if frame_rate is not None:
        show_frame_rate(session, frame_rate)
        change = True

    if change:
        gu.update_level_of_detail()
    else:
        na = gu.num_atoms_shown
        msec = session.ui.main_window.graphics_window.redraw_interval
        rate = 1000.0 / msec if msec > 0 else 1000.0
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

def register_command(session):
    from .cli import CmdDesc, register, IntArg, FloatArg, BoolArg
    desc = CmdDesc(
        keyword=[('atom_triangles', IntArg),
                 ('bond_triangles', IntArg),
                 ('ribbon_divisions', IntArg),
                 ('ribbon_sides', IntArg),
                 ('max_frame_rate', FloatArg),
                 ('frame_rate', BoolArg)],
        synopsis='Set graphics rendering parameters'
    )
    register('graphics', desc, graphics, logger=session.logger)
    desc = CmdDesc(
        synopsis='restart graphics drawing after an error'
    )
    register('graphics restart', desc, graphics_restart, logger=session.logger)

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
