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
        self.__add_suffix = kw.pop('add_suffix', None)
        super().__init__(parent, *args, **kw)

    def GetFilenames(self):
        fns = super().GetFilename()
        return self.__append_suffix(fns)

    def GetPaths(self):
        paths = super().GetPaths()
        return self.__append_suffix(paths)

    def __append_suffix(self, paths):
        if self.__add_suffix is None:
            return paths
        appended = []
        import os.path
        for path in paths:
            front, ext = os.path.splitext(path)
            if ext:
                appended.append(path)
            else:
                appended.append(path + self.__add_suffix)

