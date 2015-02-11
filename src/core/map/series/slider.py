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
        p = play_op(self.series, session = self.session, start = t, loop = True, cacheFrames = n)
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

# ------------------------------------------------------------------------------
#
class Volume_Series_WX_GUI:

    SIZE = (500, 25)

    def __init__(self, series, session):

        self.session = session
        self.series = series
        self.playing = False

        n = max(ser.number_of_times() for ser in series)
        sname = ', '.join('#%d' % ser.id for ser in series) + (' length %d' % n)

        from ...ui.tool_api import ToolWindow
        tw = ToolWindow("Volume Series", "General", session,
                        size=self.SIZE, destroy_hides=True)
        self.tool_window = tw
        parent = tw.ui_area

        import wx
        label = wx.StaticText(parent, label = "Time")
        self.time = tt = wx.SpinCtrl(parent, max = n-1, size = (50,-1))
        tt.Bind(wx.EVT_SPINCTRL, self.time_changed_cb)
        self.slider = sl = wx.Slider(parent, value = 0, minValue = 0, maxValue = n-1)
        sl.Bind(wx.EVT_SLIDER, self.slider_moved_cb)
        self.play_button = pb = wx.ToggleButton(parent, label = "Play", style=wx.BU_EXACTFIT)
        pb.Bind(wx.EVT_TOGGLEBUTTON, self.play_cb)
        self.subsample_button = x2 = wx.ToggleButton(parent, label = "x2", style=wx.BU_EXACTFIT)
        x2.Bind(wx.EVT_TOGGLEBUTTON, self.subsample_cb)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(label, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(tt, 0, wx.FIXED_MINSIZE)
        sizer.Add(sl, 1, wx.EXPAND)
        sizer.Add(pb, 0, wx.FIXED_MINSIZE)
        sizer.Add(x2, 0, wx.FIXED_MINSIZE)
        parent.SetSizerAndFit(sizer)

        tw.manage(placement="top")

        # session.close_model_callbacks.append(self.closed_series_cb)

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def time_changed_cb(self, event):
        t = self.time.GetValue()
        self.slider.SetValue(t)
        self.update_time(t)

    def slider_moved_cb(self, event):
        t = self.slider.GetValue()
        self.time.SetValue(t)
        self.update_time(t)

    def update_time(self, t):
        for s in self.series:
            lt = s.last_shown_time
            if not lt is None and lt != t:
                s.copy_display_parameters(lt, t)
            s.show_time(t)

    def play_cb(self, event):
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
        p = play_op(self.series, session = self.session, start = t, loop = True, cacheFrames = n)
        def update_slider(t, self=self):
            self.slider.SetValue(t)
            self.time.SetValue(t)
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
        text = 'Play' if play else 'Stop'
        self.play_button.SetLabel(text)
        # Use bitmaps instead.

    def subsample_cb(self, event):
        subsamp = self.subsample_button.GetValue()
        step = (2,2,2) if subsamp else (1,1,1)
        for s in self.series:
            lt = s.last_shown_time
            if not lt is None:
                s.maps[lt].new_region(ijk_step = step, adjust_step = False)

    def closed_series_cb(self, models):
        closed = [s for s in self.series if s in models]
        if closed:
            self.tool_window.destroy()

def show_slider_on_open2(session):
    # Register callback to show slider when a map series is opened
    if not hasattr(session, '_registered_map_series_slider'):
        session._registered_map_series_slider = True
        from ...models import ADD_MODELS
        session.triggers.add_handler(ADD_MODELS, lambda name,m,s=session: models_added2_cb(m,s))

def models_added2_cb(models, session):
    # Show slider when a map series is opened.
    from .series import Map_Series
    ms = [m for m in models if isinstance(m, Map_Series)]
    if ms:
        Volume_Series_WX_GUI(ms, session).show()
