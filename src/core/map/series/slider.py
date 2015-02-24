# vi: set expandtab shiftwidth=4 softtabstop=4:
# ------------------------------------------------------------------------------
#
class Volume_Series_Slider:

    def __init__(self, series, session):

        self.session = session
        self.series = series
        self.delay_time_msec = 30

        n = max(ser.number_of_times() for ser in series)
        sname = ', '.join('#%d' % ser.id for ser in series) + (' length %d' % n)

        from ...ui.qt.qt import QtWidgets, Qt, QtCore
        self.dock_widget = dw = QtWidgets.QDockWidget('Image Series %s' % sname, session.main_window)
        dw.destroyed.connect(self.widget_destroyed_cb)
        dw.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        w = QtWidgets.QWidget(dw)               # Frame
        hbox = QtWidgets.QHBoxLayout(w)
        hbox.setContentsMargins(0,0,0,0)

        lab = QtWidgets.QLabel(' Time', w)
        hbox.addWidget(lab)

        sb = QtWidgets.QSpinBox(w)
        sb.setMinimum(0)
        sb.setMaximum(n-1)
        hbox.addWidget(sb)

        self.slider = s = QtWidgets.QSlider(dw)
        s.setMinimum(0)
        s.setMaximum(n-1)
        s.setTickInterval(10)
        s.setTickPosition(s.TicksBelow)
        s.setOrientation(Qt.Horizontal)
#        s.setTracking(False)            # Don't call value changed during drag.
        s.valueChanged.connect(lambda t, se=self: se.slider_moved_cb(t))

        sb.valueChanged.connect(s.setValue)
        s.valueChanged.connect(sb.setValue)

        hbox.addWidget(s)
        
        self.playing = False
        self.play_button = play = QtWidgets.QToolButton(w)
        play.setIcon(w.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        play.clicked.connect(self.play_cb)
        hbox.addWidget(play)
        
        self.subsample_button = samp = QtWidgets.QToolButton(w)
        from ...ui.qt.gui import icon
        samp.setContentsMargins(0,0,0,0)
        samp.setIcon(icon('x2.png'))
        samp.setCheckable(True)
        samp.clicked.connect(self.subsample_cb)
        hbox.addWidget(samp)

        dw.setWidget(w)

        session.close_model_callbacks.append(self.closed_series_cb)

        if not hasattr(session, '_volume_series_sliders'):
            session._volume_series_sliders = []
        session._volume_series_sliders.append(self)

    def show(self):
        from ...ui.qt.qt import QtCore
        dw = self.dock_widget
        self.session.main_window.addDockWidget(QtCore.Qt.TopDockWidgetArea, dw)
        dw.setVisible(True)

    def hide(self):
        self.session.main_window.removeDockWidget(self.dock_widget)

    def slider_moved_cb(self, t):
        # Don't update display in callback or mouse release is missed and slider
        # gets dragged without button held down.  Bug in Mac Qt 5.
        from ...ui.qt.qt import QtCore
        QtCore.QTimer.singleShot(self.delay_time_msec, self.show_slider_time)

    def show_slider_time(self):
        t = self.slider.value()
        for s in self.series:
            lt = s.last_shown_time
            if not lt is None and lt != t:
                s.copy_display_parameters(lt, t)
            s.show_time(t)

    def play_cb(self):
        if self.playing:
            self.stop()
        else:
            self.play()

    def play(self):
        s0 = self.series[0]
        t = s0.last_shown_time
        n = s0.number_of_times()
        if t >= n-1:
            t = 0
        from .vseries_command import play_op
        p = play_op(self.series, session = self.session, start = t, loop = True,
                    cacheFrames = n * len(self.series))
        def update_slider(t, self=self):
            self.slider.setValue(t)
        p.time_step_cb = update_slider
        self.playing = True
        self.set_play_button_icon(play = False)

    def stop(self):
        from .vseries_command import stop_op
        stop_op(self.series)
        self.playing = False
        self.set_play_button_icon(play = True)

    def set_play_button_icon(self, play):
        pb = self.play_button
        from ...ui.qt.qt import QtWidgets
        icon = QtWidgets.QStyle.SP_MediaPlay if play else QtWidgets.QStyle.SP_MediaPause
        pb.setIcon(pb.style().standardIcon(icon))

    def subsample_cb(self):
        subsamp = self.subsample_button.isChecked()
        step = (2,2,2) if subsamp else (1,1,1)
        for s in self.series:
            lt = s.last_shown_time
            if not lt is None:
                s.maps[lt].new_region(ijk_step = step, adjust_step = False)

    def closed_series_cb(self, models):
        closed = [s for s in self.series if s in models]
        if closed:
            self.series = [s for s in self.series if not s in closed]
            if len(self.series) == 0:
                self.dock_widget.close()

    def widget_destroyed_cb(self):
        print('destroyed dock widget')
        self.dock_widget = None

    def closed(self):
        return self.dock_widget is None
        
def show_slider_on_open(session):
    # Register callback to show slider when a map series is opened
    if not hasattr(session, '_registered_map_series_slider'):
        session._registered_map_series_slider = True
        session.add_model_callbacks.append(lambda m,s=session: models_added_cb(m,s))

def models_added_cb(models, session):
    # Show slider when a map series is opened.
    from .series import Map_Series
    ms = [m for m in models if isinstance(m, Map_Series)]
    if ms:
        Volume_Series_Slider(ms, session).show()

def sliders(session):
    s = getattr(session, '_volume_series_sliders', [])
    if s:
        s = [sl for sl in s if not sl.closed()] # Remove closed sliders
    return tuple(s)
