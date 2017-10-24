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

from chimerax.markers import MarkerSet
class Signals(MarkerSet):
    def __init__(self, session, name, coords, signals):
        MarkerSet.__init__(self, session, name)
        self.coords = coords
        self.signals = signals
        self.num_times = len(signals)
        self.current_time = 0

        self.signal_range = r = (signals.min(), signals.max())
        self.color_range = r

        from chimerax.core.colors import BuiltinColormaps
        self.colormap = BuiltinColormaps['blue-white-red']
#        self.colormap = BuiltinColormaps['spectral']
#        self.colormap = BuiltinColormaps['ylgnbu-8'].rescale_range(0,.4)
#        self.colormap = BuiltinColormaps['ylorrd']

        xyz_min, xyz_max = coords.min(axis=0), coords.max(axis=0)
        box_size = (xyz_max-xyz_min).max()
        from math import sqrt
        r = box_size / sqrt(len(coords))
        self.radius_range = (r,3*r)

        self.create_markers(r)
        self.set_time(0)

    def create_markers(self, radius):
        rgba = (255,255,255,255)
        radius = self.radius_range[0]
        for i, xyz in enumerate(self.coords):
            self.create_marker(xyz, rgba, radius, i)

    def set_time(self, time):
        self.current_time = time
        tsig = self.signals[time,:]
        csmin,csmax = self.color_range
        cfrac = (tsig - csmin) / (csmax-csmin)
        markers = self.atoms
        markers.colors = self.colormap.interpolated_rgba8(cfrac)
        smin,smax = self.signal_range
        rmin,rmax = self.radius_range
        frac = (tsig - smin) / (smax-smin)
        r = rmin + frac*(rmax-rmin)
        from numpy import float32
        markers.radii = r.astype(float32)

    def update(self):
        self.set_time(self.current_time)
        
    def play(self):
        def next_time(session, time, s=self):
            s.set_time(time)
        from chimerax.core.commands import motion
        motion.CallForNFrames(next_time, self.num_times, self.session)

# -----------------------------------------------------------------------------------------
#
from chimerax.core.ui.widgets.slider import Slider
class TimeSlider(Slider):

    def __init__(self, session, signals, pause_frames = 1, movie_framerate = 25,
                 steady_atoms = None, compute_ss = False):

        s = signals[0]
        nc = len(s.coords)
        self.num_times = nt = len(s.signals)
        title = 'Signals for %d points at %d times' % (nc, nt)
        Slider.__init__(self, session, 'Signal', 'Time', title, value_range = (0, nt-1),
                        pause_frames = pause_frames, pause_when_recording = True,
                        movie_framerate = movie_framerate)

        self.signals = signals
        self.set_slider(s.current_time)
        
        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

    def change_value(self, t, playing = False):
        for s in self.signals:
            s.set_time(t)

    def valid_value(self, i):
        return i >= 0 and i < self.num_times
            
    def models_closed_cb(self, name, models):
        sdel = set(s for s in self.signals if s in models)
        if sdel:
            self.signals = [s for s in self.signals if not s in sdel]
            if len(self.signals) == 0:
                self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None
        super().delete()
        self.signals = None

# -----------------------------------------------------------------------------------------
# NPZ file containing two arrays
# "coord" (Nx3 float64 (x,y,z) coordinates) and
# "signal" (NxT float64), N = 1898 cells, T = 1985 time points.
#
def read_signals(session, path):
    import numpy
    f = numpy.load(path)	# NpzFile object
    coord = f['coord']
    sig = f['signal']
    f.close()
    from os.path import basename
    name = basename(path)
    s = Signals(session, name, coord, sig)
    TimeSlider(session, [s])

    smin, smax = s.signal_range
    msg = ('Opened %s with %d positions and %d signal times, signal range %.3g - %.3g'
           % (name, len(coord), len(sig), smin, smax))
    return [s], msg

# -----------------------------------------------------------------------------------------
#
def signal(session, signals, palette = None, color_range = None, radius_range = None,
           play = False, slider = False):
    '''
    Set coloring and marker radius for signal time series models.

    Parameters
    ----------
    signals : list of Signals models
        Models to change
    palette : Colormap
        Colormap mapping signal intensity to color of markers.
    color_range : (float,float)
        Minimum and maximum signal values used with the color palette.
    radius_range : (float,float)
        Minimum and maximum marker radius, minimum used for smallest signal,
        and maximum used for largest signal.
    play : bool
      Whether to play through the series, changing marker colors and radii.
    slider : bool
      Whether to show a slider control for playing the series.
    '''
    for s in signals:
        if palette is not None:
            s.colormap = palette
        if color_range is not None:
            s.color_range = color_range
        if radius_range is not None:
            s.radius_range = radius_range
        s.update()

    if slider:
        TimeSlider(session, signals)

    if play:
        for s in signals:
            s.play()
        
# -----------------------------------------------------------------------------------------
#
from chimerax.core.commands import ModelsArg
class SignalsArg(ModelsArg):
    """Parse Signals markers set specifier"""
    name = "a signals specifier"
    @classmethod
    def parse(cls, text, session):
        models, text, rest = super().parse(text, session)
        sig = [m for m in models if isinstance(m, Signals)]
        return sig, text, rest
    
# -----------------------------------------------------------------------------------------
#
def register_signal_command(logger):
    from chimerax.core.commands import CmdDesc, register, ColormapArg, Float2Arg, NoArg
    desc = CmdDesc(
        required = [('signals', SignalsArg)],
        keyword = [('palette', ColormapArg),
                   ('color_range', Float2Arg),
                   ('radius_range', Float2Arg),
                   ('play', NoArg),
                   ('slider', NoArg),
        ],
        synopsis = 'set colors and radius ranges for signal markers'
    )
    register('signal', desc, signal, logger=logger)
