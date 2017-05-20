# vim: set expandtab ts=4 sw=4:

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

def mseries(session, models, pause_frames = 10, loop = 1, step = 1):
    '''Display series of models one after another.

    Parameters
    ----------
    models : list of Model
      Sequence of models to show.
    pause_frames : int
      Number of frames to show each model. Default 10.
    loop : integer
      How many times to repeat playing through the models.  Default 1.
    step : integer
      Show every Nth model with N given by step, default 1.
    '''
    mlist = models[0].child_models() if len(models) == 1 else models
    if mlist:
        msp = ModelSequencePlayer(mlist, pause_frames = pause_frames,
                                  loop = loop, step = step)
        msp.start()

# -----------------------------------------------------------------------------
#
def mseries_slider(session, models, pause_frames = 10, step = 1, movie_framerate = 25):
    '''Show slider to play through (i.e. display) a series of models.

    Parameters
    ----------
    models : list of Model
      Sequence of models to show.
    pause_frames : int
      Number of frames to show each model. Default 10.
    step : integer
      Show every Nth model with N given by step, default 1.
    movie_framerate : float
      Frames per second used when playing back a movie recorded with the record button.
    '''
    mlist = models[0].child_models() if len(models) == 1 else models
    if mlist:
        ModelSequenceSlider(session, mlist, pause_frames = pause_frames,
                            step = step, movie_framerate = movie_framerate)

# ------------------------------------------------------------------------------
#
def register_mseries_command(logger):
    from chimerax.core.commands import CmdDesc, register, TopModelsArg, IntArg
    desc = CmdDesc(required = [('models', TopModelsArg)],
                   keyword = [('pause_frames', IntArg),
                              ('loop', IntArg),
                              ('step', IntArg)],
                   synopsis = 'Show sequence of models in turn.')
    register('mseries', desc, mseries, logger=logger)

    desc = CmdDesc(required = [('models', TopModelsArg)],
                   keyword = [('pause_frames', IntArg),
                              ('step', IntArg),
                              ('movie_framerate', IntArg)],
                   synopsis = 'Show slider to play through sequence of models.')
    register('mseries slider', desc, mseries_slider, logger=logger)
    
# -----------------------------------------------------------------------------
#
class ModelSequencePlayer:

  def __init__(self, models, pause_frames = 1, loop = 1, step = 1):

    self.models = models
    self.inext = None
    self.pause_frames = pause_frames
    self._pause_count = 0
    self.loop = loop
    self.step = step
    self._handler = None

  def start(self):

    self.inext = 0 if self.step > 0 else len(models)-1
    t = self.models[0].session.triggers
    self._handler = t.add_handler('new frame', self.frame_cb)

  def stop(self):

    if self._handler is None:
      return
    t = self.models[0].session.triggers
    t.remove_handler(self._handler)
    self._handler = None
    self.inext = None

  def frame_cb(self, *_):

    pc = self._pause_count
    self._pause_count = (pc + 1) % self.pause_frames
    if pc > 0:
      return

    i = self.inext
    for mi,m in enumerate(self.models):
        if not m.deleted:
            m.display = (mi == i)
        
    s = self.step
    self.inext += s
    iend = len(self.models)-1
    if ((s > 0 and self.inext > iend) or
        (s < 0 and self.inext < 0)):
      if self.loop <= 1:
        self.stop()
      else:
        self.inext = 0 if s > 0 else iend
        self.loop -= 1

# ------------------------------------------------------------------------------
#
from chimerax.core.ui.widgets.slider import Slider

class ModelSequenceSlider(Slider):

    SESSION_SKIP = True

    def __init__(self, session, models, pause_frames = 1, step = 1, movie_framerate = 25):

        self.models = models[::step]
        n = len(self.models)

        title = 'Model series (%d) %s ...' % (len(models), models[0].name,)
        Slider.__init__(self, session, 'Model Series', 'Model', title, value_range = (1,n),
                        pause_frames = pause_frames, pause_when_recording = True,
                        movie_framerate = movie_framerate)
        dm = [i for i,m in enumerate(self.models) if m.display]
        self.update_value(dm[0]+1 if dm else 1)

        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

    def change_value(self, i, playing = False):
        for mi,m in enumerate(self.models):
            if not m.deleted:
                m.display = (mi == i-1)

    def valid_value(self, i):
        return i-1 < len(self.models) and not self.models[i-1].deleted
            
    def models_closed_cb(self, name, models):
        mset = set(models)
        self.models = [m for m in self.models if m not in mset and not m.deleted]
        if len(self.models) == 0:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None
        super().delete()
        self.models = []
