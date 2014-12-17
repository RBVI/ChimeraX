# vim: set expandtab ts=4 sw=4:

class ToolWindow:

    placements = ["right", "left", "top", "bottom"]

    def __init__(self, tool_name, category, session, menus=False,
            icon=None, size=None, destroy_hides=False):
        """ 'ui_area' is the parent to all the tool's widgets;
            Call 'manage' once the widgets are set up to put the
            tool into the main window.
        """
        try:
            self.__toolkit = _Wx(self, tool_name, menus, session, size,
                destroy_hides)
        except ImportError:
            # browser version
            raise NotImplementedError("Browser tool API not implemented")
        self.ui_area = self.__toolkit.ui_area

    def destroy(self):
        self.__toolkit.destroy()
        self.__toolkit = None

    def manage(self, placement):
        """ Tool will be docked into main window on the side indicated by
            'placement' (which should be a value from self.placements or None);
            if 'placement' is None, the tool will be detached from the main
            window.
        """
        self.__toolkit.manage(placement)

    def getShown(self):
        return self.__toolkit.shown

    def setShown(self, shown):
        self.__toolkit.shown = shown

    shown = property(getShown, setShown)

class _Wx:

    def __init__(self, tool_window, tool_name, menus, session, size,
            destroy_hides):
        import wx
        self.tool_window = tool_window
        self.tool_name = tool_name
        self.destroy_hides = destroy_hides
        self.main_window = mw = session.ui.main_window
        if not mw:
            raise RuntimeError("No main window or main window dead")
        if size is None:
            size = wx.DefaultSize
        class WxToolPanel(wx.Panel):
            def __init__(self, parent, destroy_hides=destroy_hides, **kw):
                self._destroy_hides = destroy_hides
                wx.Panel.__init__(self, parent, **kw)
                self.Bind(wx.EVT_CLOSE, self.OnClose)

            def OnClose(self, event):
                if self.destroy_hides and event.CanVeto():
                    self.Shown = False
                else:
                    self.Destroy()

        self.ui_area = WxToolPanel(mw, name=tool_name, size=size)
        self._pane_info = None

    def destroy(self):
        if self.tool_window.ui_panel:
            self.shown = False
            if self.destroy_hides:
                return
            self.ui_area.Destroy()
        self.tool_window = None

    def manage(self, placement):
        import wx
        placements = self.tool_window.placements
        if placement is None:
            side = wx.RIGHT
        else:
            if placement not in placements:
                raise ValueError("placement value must be one of: {}, or None"
                    .format(", ".join(placements)))
            else:
                side = dict(zip(placements, [wx.RIGHT, wx.LEFT,
                    wx.TOP, wx.BOTTOM]))[placement]
        mw = self.main_window
        mw.aui_mgr.AddPane(self.ui_area, side, self.tool_name)
        mw.aui_mgr.Update()
        if placement is None:
           mw.aui_mgr.GetPane(self.ui_area).Float()

    def getShown(self):
        return self.ui_area.Shown

    def setShown(self, shown):
        aui_mgr = self.main_window.aui_mgr
        if shown:
            if self._pane_info:
                # has been hidden at least once
                aui_mgr.AddPane(self.ui_area, self._pane_info)
                self._pane_info = None
        else:
            self._pane_info = aui_mgr.GetPane(self.ui_area)
            aui_mgr.DetachPane(self.ui_area)
        aui_mgr.Update()

        self.ui_area.Shown = shown

    shown = property(getShown, setShown)
