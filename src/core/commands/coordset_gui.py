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

from ..ui.widgets.slider import Slider
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
        self.update_value(structure.active_coordset_id)

        from .. import atomic
        t = atomic.get_triggers(session)
        self._coordset_change_handler = t.add_handler('changes', self.coordset_change_cb)
        
        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

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
        from .. import atomic
        t = atomic.get_triggers(self.session)
        t.remove_handler(self._coordset_change_handler)
        self._coordset_change_handler = None

        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None

        super().delete()
        self.structure = None
