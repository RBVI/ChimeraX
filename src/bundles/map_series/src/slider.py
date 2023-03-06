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

from chimerax.ui.widgets.slider import Slider
class MapSeriesSlider(Slider):

    def __init__(self, session, *, series=[], pause_frames = 1, movie_framerate = 25):

        self.series = list(series)

        title = "Map series %s" % ', '.join(s.name for s in series)
        n = max((s.number_of_times() for s in series), default = 0)
        self._num_times = n
        Slider.__init__(self, session, 'Map Series', 'Time', title,
                        value_range = (0, n-1),
                        pause_frames = pause_frames, pause_when_recording = True,
                        movie_framerate = movie_framerate)
        self._add_subsample_button()
        
        from .play import Play_Series
        self._player = Play_Series(self.series, session=session,
                                   rendering_cache_size=self._map_count)
        
        if series:
            times = series[0].shown_times
            if times:
                t = next(iter(times))
                self.set_slider(t)
                
        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(
            REMOVE_MODELS, self.models_closed_cb)

        if hasattr(session, '_map_series_sliders'):
            session._map_series_sliders.append(self)
        else:
            session._map_series_sliders = [self]

    def add_series(self, series):
        self.series.append(series)
        from .play import Play_Series
        self._player = Play_Series(self.series, self.session,
                                   rendering_cache_size=self._map_count)
        sname = ', '.join(s.name for s in self.series)
        if len(sname) > 50:
            sname = sname[:50] + '...'
        self.display_name = "Map series %s" % sname
        self.tool_window.title = self.display_name
        t = self.slider.value()
        if t >= 0 and t < series.number_of_times():
            self.change_value(t)

    @property
    def _map_count(self):
        return sum([s.number_of_times() for s in self.series])
    def size(self):
        if not self.series:
            return (0, (0,0,0))
        s0 = self.series[0]
        return s0.number_of_times(), s0.grid_size()

    def change_value(self, t, playing = False):
      self._player.change_time(t)

    def valid_value(self, t):
        for s in self.series:
            if t >= 0 and t < s.number_of_times() and not s.is_volume_closed(t):
                return True
        return False

    def _add_subsample_button(self):
        from Qt.QtWidgets import QPushButton
        self._subsample_button = x2 = QPushButton()
        x2.setCheckable(True)
        from chimerax.ui.icons import get_qt_icon
        x2i = get_qt_icon('half')
        x2.setIcon(x2i)
        x2.clicked.connect(self.subsample_cb)
        layout = self.tool_window.ui_area.layout()
        layout.addWidget(x2)

    def subsample_cb(self, event):
        subsamp = self._subsample_button.isChecked()
        step = (2, 2, 2) if subsamp else (1, 1, 1)
        for s in self.series:
            lt = s.last_shown_time
            if lt is not None:
                s.maps[lt].new_region(ijk_step=step, adjust_step=False)

    def models_closed_cb(self, name, models):
        closed = [s for s in self.series if s in models]
        if closed:
            self.delete()

    # Override ToolInstance method
    def delete(self):
        s = self.session
        s.triggers.remove_handler(self._model_close_handler)
        super().delete()
        s._map_series_sliders.remove(self)


def show_slider_on_open(session):
    # Register callback to show slider when a map series is opened
    if hasattr(session, '_map_series_slider_handler'):
        return
    from chimerax.core.models import ADD_MODELS
    session._map_series_slider_handler = session.triggers.add_handler(
        ADD_MODELS, lambda name, m, s=session: models_added_cb(m, s))


def remove_slider_on_open(session):
    # Remove callback to show slider when a map series is opened
    if not hasattr(session, '_map_series_slider_handler'):
        return
    handler = session._map_series_slider_handler
    del session._map_series_slider_handler
    session.triggers.remove_handler(handler)


def models_added_cb(models, session):
    # Show slider when a map series is opened.
    from .series import MapSeries
    ms = [m for m in models if isinstance(m, MapSeries)]
    if ms:
        msstable = {mss.size():mss for mss in getattr(session, '_map_series_sliders', [])}
        for m in ms:
            mss = msstable.get((m.number_of_times(), m.grid_size()), None)
            if mss:
                mss.add_series(m)
            else:
                mss = MapSeriesSlider(session, series = [m])
                mss.show()
                msstable[mss.size()] = mss
