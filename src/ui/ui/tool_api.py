# vim: set expandtab ts=4 sw=4:

class ToolWindow:

    def __init__(self, toolName, category, menus=False, prefer_detached=False,
            icon=None, size=None, placement=None):
        try:
            import wx
            self._wx_init(toolName, menus, prefer_detached, size, placement)
        except ImportError:
            # browser version
            raise NotImplementedError("Browser tool API not implemented")

    def _wx_init(self, toolName, menus, prefer_detached, size, placement):
        import wx
        from .main_win import main_window as mw
        if not mw:
            raise RuntimeError("Main window already dead!")
        if size is None:
            size = wx.DefaultSize
        self.ui_area = wx.Panel(mw, name=toolName, size=size)
        if placement is None:
            placement = wx.RIGHT
        mw.aui_mgr.AddPane(self.ui_area, placement, toolName)
        mw.aui_mgr.Update()
        if prefer_detached:
           mw.aui_mgr.GetPane(self.ui_area).Float()
