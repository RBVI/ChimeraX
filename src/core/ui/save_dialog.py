# vi: set expandtab ts=4 sw=4:


class _SaveFormat:

    def __init__(self, name, wildcard, make_ui, save):
        self.name = name
        self._wildcard = wildcard
        self._make_ui = make_ui
        self._save = save
        self._window = None

    @property
    def wildcard(self):
        return self._wildcard()

    def window(self, parent):
        if self._window is None:
            self._window = self._make_ui(parent)
        return self._window

    def save(self, session, filename):
        return self._save(session, filename)


class SaveDialog:

    DEFAULT_FORMAT = "Chimera Session"

    def __init__(self, ui):
        self.file_dialog = None
        self._registered_formats = {}
        self._format_selector = None
        import io
        self.register(self.DEFAULT_FORMAT, _session_wildcard, _session_ui, _session_save)

    def register(self, fmt, wildcard, make_ui, save):
        self._registered_formats[fmt] = _SaveFormat(fmt, wildcard, make_ui, save)
        if self._format_selector:
            self._update_format_selector()

    def display(self, parent, session):
        import wx
        if self.file_dialog is None:
            self.file_dialog = wx.FileDialog(parent, "Save File", "", "", "",
                                             wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            self.file_dialog.SetExtraControlCreator(self._extraCreatorCB)
        if self.file_dialog.ShowModal() == wx.ID_CANCEL:
            return
        self._registered_formats[self._format_selector.GetValue()].save(session, filename)

    def _extraCreatorCB(self, parent):
        import wx
        p = wx.Panel(parent)
        s = wx.BoxSizer(wx.VERTICAL)
        # ComboBox at top for selecting save file type
        self._format_selector = wx.ComboBox(p, style=wx.CB_READONLY)
        self._format_selector.Bind(wx.EVT_COMBOBOX, self._select_format)
        s.Add(self._format_selector, flag=wx.LEFT)
        # Panel for displaying format-specific options goes immediately below
        # Default is a window displaying text of "No user-settable options"
        # For some reason, a Panel does not work but a Window does
        self._no_options_window = wx.Window(p)
        ns = wx.BoxSizer(wx.VERTICAL)
        t = wx.StaticText(self._no_options_window, label="No user-settable options",
                          style=wx.ALIGN_CENTER)
        ns.Add(t, flag=wx.LEFT)
        self._no_options_window.SetSizerAndFit(ns)
        self._known_options = set([self._no_options_window])
        self._current_option = self._no_options_window
        s.Add(self._no_options_window, flag=wx.EXPAND|wx.ALL)
        # Finish up layout
        p.SetSizerAndFit(s)
        self._options_sizer = s
        self._options_window = p
        # Update everything
        self._update_format_selector()
        self._format_selector.SetValue(self.DEFAULT_FORMAT)
        self._select_format(self.DEFAULT_FORMAT)
        return p

    def _update_format_selector(self):
        choices = list(self._registered_formats.keys())
        choices.sort()
        self._format_selector.Items = choices

    def _select_format(self, event=None):
        if self._format_selector is None:
            format_name = self.DEFAULT_FORMAT
        else:
            format_name = self._format_selector.GetValue()
        fmt = self._registered_formats[self._format_selector.GetValue()]
        self.file_dialog.SetWildcard(fmt.wildcard)
        w = fmt.window(self._options_window) or self._no_options_window
        if w not in self._known_options:
            self._options_sizer.Hide(self._current_option)
            self._options_sizer.Add(w, flag=wx.EXPAND|wx.ALL)
        elif w is not self._current_option:
            self._options_sizer.Hide(self._current_option)
            self._options_sizer.Show(w)
        self._options_sizer.Layout()

    def set_wildcard(self, format):
        fmt = self._registered_formats[self._format_selector.GetValue()]
        self.file_dialog.SetWildcard(fmt.wildcard)


def _session_wildcard():
    from .. import io
    return io.wx_export_file_filter(io.SESSION)


def _session_ui(parent):
    return None


def _session_save(self, session, filename):
    import os.path
    ext = os.path.splitext(filename)[1]
    from .. import io
    exts = io.extensions(io.category(io.SESSION))
    if exts and ext not in exts:
        filename += exts[0]
    # TODO: generate text command instead of calling export directly
    # so that command logging happens automatically
    from .. import commands
    commands.export(session, filename)
    session.logger.info("File \"%s\" saved." % filename)
