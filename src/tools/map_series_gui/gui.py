# vi: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance


# ------------------------------------------------------------------------------
#
class MapSeries(ToolInstance):

    SIZE = (500, 25)

    def __init__(self, session, tool_info, series=None):

        super().__init__(session, tool_info)

        self.series = series
        self.playing = False

        if series is None:
            n = 1
        else:
            n = max(ser.number_of_times() for ser in series)
            # sname = (', '.join('#%d' % ser.id for ser in series) +
            #          (' length %d' % n))

        self.display_name = "Map series %s" % ', '.join(s.name for s in series)
        tw = session.ui.create_main_tool_window(
            self, size=self.SIZE, destroy_hides=True)
        self.tool_window = tw
        parent = tw.ui_area

        import wx
        label = wx.StaticText(parent, label="Time")
        self.time = tt = wx.SpinCtrl(parent, max=n - 1, size=(50, -1))
        tt.Bind(wx.EVT_SPINCTRL, self.time_changed_cb)
        self.slider = sl = wx.Slider(parent, value=0, minValue=0,
                                     maxValue=n - 1)
        sl.Bind(wx.EVT_SLIDER, self.slider_moved_cb)
        self.play_button = pb = wx.ToggleButton(parent, style=wx.BU_EXACTFIT)
        self.set_play_button_icon(play=True)
        pb.Bind(wx.EVT_TOGGLEBUTTON, self.play_cb)
        self.subsample_button = x2 = wx.ToggleButton(parent,
                                                     style=wx.BU_EXACTFIT)
        from os.path import dirname, join
        hbm = wx.Bitmap(join(dirname(__file__), 'half.png'))
        x2.SetBitmap(hbm)
        x2.Bind(wx.EVT_TOGGLEBUTTON, self.subsample_cb)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(label, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(tt, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(sl, 1, wx.EXPAND)
        sizer.Add(pb, 0, wx.FIXED_MINSIZE)
        sizer.Add(x2, 0, wx.FIXED_MINSIZE)
        parent.SetSizerAndFit(sizer)

        tw.manage(placement="top")

        from chimera.core.models import REMOVE_MODELS
        self.model_close_handler = session.triggers.add_handler(
            REMOVE_MODELS, self.models_closed_cb)

        session.tools.add([self])

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
        from chimera.core.map.series.vseries_command import play_op
        p = play_op(self.session, self.series, start=t, loop=True, cacheFrames=n)

        def update_slider(t, self=self):
            self.slider.SetValue(t)
            self.time.SetValue(t)
        p.time_step_cb = update_slider
        self.playing = True
        self.set_play_button_icon(play=False)

    def stop(self):
        if self.series is None:
            return
        from chimera.core.map.series.vseries_command import stop_op
        stop_op(self.session, self.series)
        self.playing = False
        self.set_play_button_icon(play=True)

    def set_play_button_icon(self, play):
        from os.path import dirname, join
        bitmap_path = (join(dirname(__file__),
                       'play.png' if play else 'pause.png'))
        import wx
        pbm = wx.Bitmap(bitmap_path)
        self.play_button.SetBitmap(pbm)

    def subsample_cb(self, event):
        subsamp = self.subsample_button.GetValue()
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
        if series is None:
            n = 1
        else:
            n = max(ser.number_of_times() for ser in series)
        self.display_name = "Map series %s" % ', '.join(s.name for s in series)
        self.series = series
        self.time.SetRange(0, n - 1)
        self.slider.SetMax(n - 1)

    #
    # Override ToolInstance methods
    #
    def delete(self):
        s = self.session
        s.triggers.delete_handler(self.model_close_handler)
        self.tool_window.shown = False
        self.tool_window.destroy()
        s.tools.remove([self])
        super().delete()

    def display(self, b):
        """Show or hide map series user interface."""
        self.tool_window.shown = b

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, phase, session, flags):
        pass

    def restore_snapshot(self, phase, session, version, data):
        pass

    def reset_state(self):
        pass


def show_slider_on_open(session):
    # Register callback to show slider when a map series is opened
    if not hasattr(session, '_registered_map_series_slider'):
        session._registered_map_series_slider = True
        from chimera.core.models import ADD_MODELS
        session.triggers.add_handler(
            ADD_MODELS, lambda name, m, s=session: models_added_cb(m, s))


def models_added_cb(models, session):
    # Show slider when a map series is opened.
    from chimera.core.map.series.series import Map_Series
    ms = [m for m in models if isinstance(m, Map_Series)]
    if ms:
        tool_info = session.toolshed.find_tool('map_series_gui')
        MapSeries(session, tool_info, ms).show()
