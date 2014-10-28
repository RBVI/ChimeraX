# ------------------------------------------------------------------------------
#
class Volume_Series_Slider:

    def __init__(self, series, session):

        self.session = session
        self.series = series
        self.delay_time_msec = 30

        n = max(ser.number_of_times() for ser in series)
        sname = ', '.join('#%d' % ser.id for ser in series) + (' length %d' % n)

        from ...ui.qt.qt import QtWidgets, Qt
        self.dock_widget = dw = QtWidgets.QDockWidget('Image Series %s' % sname, session.main_window)

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
        
        play = QtWidgets.QToolButton(w)
        play.setIcon(w.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        play.clicked.connect(self.play_cb)
        hbox.addWidget(play)

        stop = QtWidgets.QToolButton(w)
        stop.setIcon(w.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        stop.clicked.connect(self.stop_cb)
        hbox.addWidget(stop)

        dw.setWidget(w)

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
        t = self.series[0].last_shown_time
        from .vseries_command import play_op
        p = play_op(self.series, session = self.session, start = t)
        def update_slider(t, self=self):
            self.slider.setValue(t)
        p.time_step_cb = update_slider

    def stop_cb(self):
        from .vseries_command import stop_op
        stop_op(self.series)
