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

from chimerax.core.tools import ToolInstance


# ------------------------------------------------------------------------------
#
class MapSeries(ToolInstance):

    SESSION_SKIP = True

    def __init__(self, session, bundle_info, *, series=[]):
        ToolInstance.__init__(self, session, bundle_info)

        self.series = list(series)
        self.playing = False
        self._block_time_update = False
        
        self.display_name = "Map series %s" % ', '.join(s.name for s in series)
        from chimerax.core.ui.gui import MainToolWindow

        class MapSeriesWindow(MainToolWindow):
            close_destroys = False

        tw = MapSeriesWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        self._slider_max = n = max((ser.number_of_times() for ser in series), default = 0)
        from os.path import dirname, join
        from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QSlider, QPushButton
        from PyQt5.QtGui import QPixmap, QIcon
        from PyQt5.QtCore import Qt
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(4)
        tl = QLabel('Time')
        layout.addWidget(tl)
        self.time = tv = QSpinBox()
        tv.setMaximum(n-1)
        tv.valueChanged.connect(self.time_changed_cb)
        layout.addWidget(tv)
        self.slider = ts = QSlider(Qt.Horizontal)
        ts.setRange(0,n-1)
        ts.valueChanged.connect(self.slider_moved_cb)
        layout.addWidget(ts)
        self.play_button = pb = QPushButton()
# The QPushButton and QSpinBox add extra vertical space (about 5 pixels above and below)
# on Mac OS 10.11.5 with Qt 5.6.  Didn't find any way to reduce that space.  Some of the
# space is taken by the QFocusFrame around that spin box that highlights when it gets focus.
# The space around the push buttons can be made smaller with an uglier non-native button.
#            pb.setStyleSheet('max-width: 20px; max-height: 20px')
        self.set_play_button_icon(play=True)
        pb.setCheckable(True)
        pb.clicked.connect(self.play_cb)
        layout.addWidget(pb)
        self.subsample_button = x2 = QPushButton()
        x2.setCheckable(True)
        x2icon = join(dirname(__file__), 'half.png')
        x2pix = QPixmap(x2icon)
        x2i = QIcon(x2pix)
        x2.setIcon(x2i)
        x2.clicked.connect(self.subsample_cb)
        layout.addWidget(x2)
        parent.setLayout(layout)

        tw.manage(placement="right")

        from chimerax.core.models import REMOVE_MODELS
        self.model_close_handler = session.triggers.add_handler(
            REMOVE_MODELS, self.models_closed_cb)

        if hasattr(session, '_map_series_sliders'):
            session._map_series_sliders.append(self)
        else:
            session._map_series_sliders = [self]
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def add_series(self, series):
        self.series.append(series)
        sname = ', '.join(s.name for s in self.series)
        if len(sname) > 50:
            sname = sname[:50] + '...'
        self.display_name = "Map series %s" % sname
        self.tool_window.title = self.display_name
        t = self.slider.value()
        self.update_time(t)

    def size(self):
        if not self.series:
            return (0, (0,0,0))
        s0 = self.series[0]
        return s0.number_of_times(), s0.grid_size()
    
    def time_changed_cb(self, event):
        self.update_slider_range()
        t = self.time.value()
        self.slider.setValue(t)

    def slider_moved_cb(self, event):
        self.update_slider_range()
        t = self.slider.value()
        self.time.setValue(t)
        self.update_time(t)

    def update_time(self, t):
        if self._block_time_update:
            return
        for s in self.series:
            lt = s.last_shown_time
            if lt is not None and lt != t:
                s.copy_display_parameters(lt, t)
            s.show_time(t)

    def play_cb(self, event):
        if self.playing:
            self.stop()
        else:
            self.play()

    def play(self):
        if self.series is None:
            return
        s0 = self.series[0]
        t = s0.last_shown_time
        n = s0.number_of_times()
        if t >= n - 1:
            t = 0
        from chimerax.core.map.series.vseries_command import vseries_play
        p = vseries_play(self.session, self.series, start=t, loop=True, cache_frames=n)

        def update_slider(t, self=self):
            self.update_slider_range()
            self._block_time_update = True
            self.slider.setValue(t)
            self._block_time_update = False

        p.time_step_cb = update_slider
        self.playing = True
        self.set_play_button_icon(play=False)

    def stop(self):
        if self.series is None:
            return
        from chimerax.core.map.series.vseries_command import vseries_stop
        vseries_stop(self.session, self.series)
        self.playing = False
        self.set_play_button_icon(play=True)

    def set_play_button_icon(self, play):
        from os.path import dirname, join
        bitmap_path = (join(dirname(__file__),
                       'play.png' if play else 'pause.png'))
        pb = self.play_button
        from PyQt5.QtGui import QPixmap, QIcon
        ppix = QPixmap(bitmap_path)
        pi = QIcon(ppix)
        pb.setIcon(pi)

    def subsample_cb(self, event):
        subsamp = self.subsample_button.isChecked()
        step = (2, 2, 2) if subsamp else (1, 1, 1)
        for s in self.series:
            lt = s.last_shown_time
            if lt is not None:
                s.maps[lt].new_region(ijk_step=step, adjust_step=False)

    def models_closed_cb(self, name, models):
        closed = [s for s in self.series if s in models]
        if closed:
            self.delete()

    def set_series(self, series):
        # TODO: use this in session restore
        self.stop()
        self.display_name = "Map series %s" % ', '.join(s.name for s in series)
        self.series = series
        self.update_slider_range()

    def update_slider_range(self):
        n = max((ser.number_of_times() for ser in self.series), default = 0)
        if n == self._slider_max:
            return
        self._slider_max = n
        maxt = max(0,n-1)
        self.time.setRange(0, maxt)
        self.slider.setMaximum(maxt)

    # Override ToolInstance method
    def delete(self):
        s = self.session
        s.triggers.remove_handler(self.model_close_handler)
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
    from chimerax.core.map.series.series import Map_Series
    ms = [m for m in models if isinstance(m, Map_Series)]
    if ms:
        msstable = {mss.size():mss for mss in getattr(session, '_map_series_sliders', [])}
        for m in ms:
            mss = msstable.get((m.number_of_times(), m.grid_size()), None)
            if mss:
                mss.add_series(m)
            else:
                bundle_info = session.toolshed.find_bundle('map_series_gui')
                MapSeries(session, bundle_info, series = [m]).show()
