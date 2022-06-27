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

from chimerax.ui.widgets.slider import Slider
class CoordinateSetSlider(Slider):

    def __init__(self, session, structure, pause_frames = 1, movie_framerate = 25,
                 steady_atoms = None, compute_ss = False):

        self.structure = structure

        title = 'Coordinate sets %s (%d)' % (structure.name, structure.num_coordsets)
        csids = structure.coordset_ids
        id_start, id_end = min(csids), max(csids)
        self.coordset_ids = set(csids)
        Slider.__init__(self, session, 'Model Series', 'Model', title, value_range = (id_start, id_end),
                        pause_frames = pause_frames, pause_when_recording = True,
                        movie_framerate = movie_framerate)

        from .coordset import CoordinateSetPlayer
        self._player = CoordinateSetPlayer(structure, id_start, id_end, istep = 1,
                                           pause_frames = pause_frames, loop = 1,
                                           compute_ss = compute_ss, steady_atoms = steady_atoms)
        self.set_slider(structure.active_coordset_id)

        from chimerax import atomic
        t = atomic.get_triggers(session)
        self._coordset_change_handler = t.add_handler('changes', self.coordset_change_cb)
        
        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

        if not hasattr(session, '_coord_set_sliders'):
            session._coord_set_sliders = set()
        session._coord_set_sliders.add(self)

    def change_value(self, i, playing = False):
      self._player.change_coordset(i)

    def valid_value(self, i):
        return i in self.coordset_ids

    def coordset_change_cb(self, name, changes):
        # If coordset changed by command, update slider
        s = self.structure
        if ('active_coordset changed' in changes.structure_reasons() and
            s in changes.modified_structures()):
            self.set_slider(s.active_coordset_id)
            
    def models_closed_cb(self, name, models):
      if self.structure in models:
        self.delete()

    # Override ToolInstance method
    def delete(self):
        from chimerax import atomic
        t = atomic.get_triggers(self.session)
        t.remove_handler(self._coordset_change_handler)
        self._coordset_change_handler = None

        self.session._coord_set_sliders.remove(self)

        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None

        super().delete()
        self.structure = None

# -----------------------------------------------------------------------------
#
from chimerax.mouse_modes import MouseMode
class PlayCoordinatesMouseMode(MouseMode):

    name = 'play coordinates'
    icon_file = 'coordset.png'
    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._wrap = False
        self._vr_full_range = 0.5	# Meters.  Motion to play full coordset.

    def mouse_drag(self, event):

        dx,dy = self.mouse_motion(event)
        w,h = self.session.main_view.window_size
        delta = (-dy/h) if abs(dy) >= abs(dx) else (dx/w)
        self._take_step(fraction = delta)
    
    def wheel(self, event):
        d = event.wheel_value()
        self._take_step(step = d)

    def _take_step(self, fraction = None, step = None):
        if fraction is None and step is None:
            return
        from chimerax.atomic import Structure
        mlist = [m for m in self.session.models.list(type = Structure)
                 if m.num_coordsets > 1 and m.visible]
        for m in mlist:
            ids = m.coordset_ids
            nc = len(ids)
            if fraction is not None:
                step = fraction * nc
            s = step + getattr(m, '_play_coordinates_accum_step', 0)
            from math import floor, ceil
            si = int(floor(s) if s >= 0 else ceil(s))
            if s != si:
                m._play_coordinates_accum_step = s-si  # Remember fractional step.
            if si != 0:
                id = m.active_coordset_id
                p = _sequence_position(id, ids)
                np = p + si
                if self._wrap:
                    np = np % nc
                elif np >= nc:
                    np = nc-1
                elif np < 0:
                    np = 0
                nid = ids[np]
                m.active_coordset_id = nid

    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        f = event.room_vertical_motion / self._vr_full_range
        self._take_step(fraction = f)

    def vr_thumbstick(self, event):
        # Virtual reality hand controller thumbstick tilt.
        step = event.thumbstick_step()
        if step != 0:
            self._take_step(step = step)

# -----------------------------------------------------------------------------
#
def _sequence_position(value, seq):
    if isinstance(seq, (tuple, list)):
        return seq.index(value)
    else:
        from numpy import ndarray, where
        if isinstance(seq, ndarray) and seq.ndim == 1:
            return where(seq == value)[0][0]
    raise RuntimeError('_sequence_position() called with non-sequence %s' % str(type(seq)))
        
# -----------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(PlayCoordinatesMouseMode(session))
