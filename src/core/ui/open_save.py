# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
open_save: open/save dialogs
============================

TODO
"""

import wx
class OpenDialog(wx.FileDialog):
    def __init__(self, parent, *args, **kw):
        kw['style'] = kw.get('style', 0) | wx.FD_OPEN
        super(OpenDialog, self).__init__(parent, *args, **kw)

class SaveDialog(wx.FileDialog):
    def __init__(self, parent, *args, **kw):
        kw['style'] = kw.get('style', 0) | wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        super(SaveDialog, self).__init__(parent, *args, **kw)

