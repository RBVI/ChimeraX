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
        super().__init__(parent, *args, **kw)

class SaveDialog(wx.FileDialog):
    def __init__(self, parent, *args, **kw):
        kw['style'] = kw.get('style', 0) | wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        self.__add_ext = kw.pop('add_extension', None)
        super().__init__(parent, *args, **kw)

    def GetPath(self):
        path = super().GetPath()
        return self.__add_extension(path)

    def __add_extension(self, path):
        if self.__add_ext is None:
            return path
        appended = []
        import os.path
        front, ext = os.path.splitext(path)
        if ext:
            return path
        return path + self.__add_ext

