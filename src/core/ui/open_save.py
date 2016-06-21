# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
open_save: open/save dialogs
============================

TODO
"""

from .. import window_sys
if window_sys == "wx":
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
            import os.path
            front, ext = os.path.splitext(path)
            if ext:
                return path
            return path + self.__add_ext

    def export_file_filter(category=None, format_name=None, all=False):
        """Return file name filter suitable for Export File dialog for WX"""

        if format_name:
            # since it seems we will be doing the exporting, don't filter
            # formats with no registered export function
            export_kw_val = False
        else:
            export_kw_val = True
        result = []
        from .. import io
        for fmt_name in io.format_names(open=False, export=export_kw_val):
            if format_name and fmt_name != format_name:
                continue
            if category and io.category(fmt_name) != category:
                continue
            exts = ', '.join(io.extensions(fmt_name))
            fmts = ';'.join('*%s' % ext for ext in io.extensions(fmt_name))
            result.append("%s files (%s)|%s" % (fmt_name, exts, fmts))
        if all:
            result.append("All files (*.*)|*.*")
        if not result:
            if not category:
                files = "any"
            else:
                files = "\"%s\"" % category
            raise ValueError("No filters for %s files" % files)
        result.sort(key=str.casefold)
        return '|'.join(result)

    def open_file_filter(all=False):
        """Return file name filter suitable for Open File dialog for WX"""

        combine = {}
        from .. import io
        for fmt_name in io.format_names():
            exts = combine.setdefault(io.category(fmt_name), [])
            exts.extend(io.extensions(fmt_name))
        result = []
        for k in combine:
            exts = ', '.join(combine[k])
            fmts = ';'.join('*%s' % ext for ext in combine[k])
            compression_suffixes = io.compression_suffixes()
            if compression_suffixes:
                for ext in combine[k]:
                    fmts += ';' + ';'.join('*%s%s' % (ext, c) for c in compression_suffixes)
            result.append("%s files (%s)|%s" % (k, exts, fmts))
        result.sort(key=str.casefold)
        if all:
            result.insert(0, "All files (*.*)|*.*")
        return '|'.join(result)
else:
    """Just use PyQt5.QtWidgets.QFileDialog for opening files"""
    from PyQt5.QtWidgets import QFileDialog
    class SaveDialog(QFileDialog):
        def __init__(self, parent, *args, **kw):
            default_suffix = kw.pop('add_extension', None)
            name_filter = kw.pop('name_filter', None)
            super().__init__(parent, *args, **kw)
            self.setFileMode(QFileDialog.AnyFile)
            self.setAcceptMode(QFileDialog.AcceptSave)
            self.setOption(QFileDialog.DontUseNativeDialog)
            if default_suffix:
                self.setDefaultSuffix(default_suffix)
            if name_filter:
                self.setNameFilter(name_filter)
            self._custom_area = None

        @property
        def custom_area(self):
            if self._custom_area is None:
                layout = self.layout()
                row = layout.rowCount()
                from PyQt5.QtWidgets import QFrame
                self._custom_area = QFrame(self)
                from PyQt5.QtCore import Qt
                layout.addWidget(self._custom_area, row, 0, 1, -1, Qt.AlignCenter)
            return self._custom_area

        def get_path(self):
            paths = self.selectedFiles()
            if not paths:
                return None
            path = paths[0]
            return path

    def export_file_filter(category=None, format_name=None, all=False):
        """Return file name filter suitable for Export File dialog for Qt"""

        if format_name:
            # since it seems we will be doing the exporting, don't filter
            # formats with no registered export function
            export_kw_val = False
        else:
            export_kw_val = True
        result = []
        from .. import io
        for fmt_name in io.format_names(open=False, export=export_kw_val):
            if format_name and fmt_name != format_name:
                continue
            if category and io.category(fmt_name) != category:
                continue
            exts = '*' + ' *'.join(io.extensions(fmt_name))
            result.append("%s files (%s)" % (fmt_name, exts))
        if all:
            result.append("All files (*)")
        if not result:
            if not category:
                files = "any"
            else:
                files = "\"%s\"" % category
            raise ValueError("No filters for %s files" % files)
        result.sort(key=str.casefold)
        return ';;'.join(result)

    def open_file_filter(all=False):
        """Return file name filter suitable for Open File dialog for Qt"""

        combine = {}
        from .. import io
        for fmt_name in io.format_names():
            exts = combine.setdefault(io.category(fmt_name), [])
            exts.extend(io.extensions(fmt_name))
        result = []
        for k in combine:
            exts = '*' + ' *'.join(combine[k])
            compression_suffixes = io.compression_suffixes()
            if compression_suffixes:
                for ext in combine[k]:
                    exts += ' ' + ' '.join('*%s%s' % (ext, c) for c in compression_suffixes)
            result.append("%s files (%s)" % (k, exts))
        result.sort(key=str.casefold)
        if all:
            result.insert(0, "All files (*)")
        return ';;'.join(result)

