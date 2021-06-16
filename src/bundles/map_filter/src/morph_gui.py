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
class MorphMapSlider(Slider):

    def __init__(self, session, interpolated_map, pause_frames = 1, movie_framerate = 25):
        self._interpolated_map = im = interpolated_map   # Interpolated_Map instance
        num_steps = int(.5 + (im.fmax - im.fmin)/im.fstep) if im.fstep != 0 else 25
        self._step_range = (0, num_steps)
        self._block_change = False  # For updating slider without updating morph

        id_sequence = ' to '.join('#%s' % v.id_string for v in im.volumes[:5])
        if len(im.volumes) > 5:
            id_sequence += ' to ...'
        title = 'Map morph #%s (%s)' % (im.result.id_string, id_sequence)
        Slider.__init__(self, session, 'Map Morph', 'Map morph', title,
                        value_range = self._step_range,
                        pause_frames = pause_frames, pause_when_recording = True,
                        movie_framerate = movie_framerate)
        
        im.f_changed_cb = self._morph_changed

        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS,
                                                                 self._models_closed_cb)

    def change_value(self, i, playing = False):
        if self._block_change:
            return
        im = self._interpolated_map
        f = im.fmin + i * im.fstep
        im.stop_playing()
        im.interpolate(f)

    def valid_value(self, i):
        return i >= self._step_range[0] and i <= self._step_range[1]

    def _morph_changed(self, f):
        im = self._interpolated_map
        i = int(.5 + (f - im.fmin) / im.fstep)
        self._block_change = True
        self.set_slider(i)
        self._block_change = False
            
    def _models_closed_cb(self, name, models):
        im = self._interpolated_map
        close = (im.result in models)
        for v in im.volumes:
            if v in models:
                close = True
        if close:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None

        super().delete()
        self._interpolated_map = None
