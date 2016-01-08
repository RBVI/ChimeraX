# vim: set expandtab ts=4 sw=4:


class _SaveFormat:

    def __init__(self, name, wildcard, make_ui, update, save):
        self.name = name
        self._wildcard = wildcard
        self._make_ui = make_ui
        self._update = update
        self._save = save
        self._window = None

    @property
    def wildcard(self):
        return self._wildcard()

    def window(self, parent):
        if self._make_ui is None:
            return None
        if self._window is None:
            self._window = self._make_ui(parent)
        return self._window

    def update(self, session, save_dialog):
        if self._update:
            self._update(session, save_dialog)

    def save(self, session, filename):
        return self._save(session, filename)


class MainSaveDialog:

    DEFAULT_FORMAT = "ChimeraX Session"

    def __init__(self, ui):
        self.file_dialog = None
        self._registered_formats = {}
        self._format_selector = None
        import io
        self.register(self.DEFAULT_FORMAT, _session_wildcard, None, None, _session_save)

    def register(self, format_name, wildcard, make_ui, update, save):
        self._registered_formats[format_name] = _SaveFormat(format_name, wildcard, make_ui,
                                                            update, save)
        if self._format_selector:
            self._update_format_selector()

    def deregister(self, format_name):
        del self._registered_formats[format_name]

    def current_format(self):
        if self._format_selector is None:
            format_name = self.DEFAULT_FORMAT
        else:
            n = self._format_selector.GetSelection()
            format_name = self._format_selector.GetString(n)
        return self._registered_formats[format_name]

    def display(self, parent, session):
        import wx
        if self.file_dialog is None:
            from .open_save import SaveDialog
            self.file_dialog = SaveDialog(parent, "Save File")
            self.file_dialog.SetExtraControlCreator(self._extraCreatorCB)
        else:
            fmt = self.current_format()
            fmt.update(session, self)
        try:
            self.session = session
            if self.file_dialog.ShowModal() == wx.ID_CANCEL:
                return
        finally:
            del self.session
        fmt = self.current_format()
        filename = self.file_dialog.GetPath()
        fmt.save(session, filename)

    def _extraCreatorCB(self, parent):
        import wx
        # TODO: make tab traversal work on extra control panel
        p = wx.Panel(parent)
        try:
            s = wx.BoxSizer(wx.VERTICAL)
            # Choice at top for selecting save file type
            format_selector = wx.Choice(p, style=wx.CB_READONLY)
            format_selector.Bind(wx.EVT_CHOICE, self._select_format)
            s.Add(format_selector, proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
            # Panel for displaying format-specific options goes immediately below
            # Default is a window displaying text of "No user-settable options"
            # For some reason, a Panel does not work but a Window does
            no_options_window = wx.Window(p, style=wx.BORDER_SIMPLE)
            ns = wx.BoxSizer(wx.VERTICAL)
            t = wx.StaticText(no_options_window, label="No user-settable options")
            ns.Add(t, proportion=1, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
            no_options_window.SetSizerAndFit(ns)
            s.Add(no_options_window, proportion=1, flag=wx.EXPAND|wx.ALL|wx.ALIGN_CENTER, border=2)
            # Save references to widgets we need
            self._options_panel = p
            self._options_sizer = s
            self._format_selector = format_selector
            self._no_options_window = no_options_window
            self._known_options = set([self._no_options_window])
            self._current_option = self._no_options_window
            # Update everything (may require attributes set above)
            self._update_format_selector()
            p.SetSizerAndFit(s)
            n = self._format_selector.Items.index(self.DEFAULT_FORMAT)
            self._format_selector.SetSelection(n)
            self._select_format(self.DEFAULT_FORMAT)
            return p
        except:
            import traceback
            print("Error in MainSaveDialog SetExtraControlCreator callback:\n")
            print(traceback.format_exc())
            p.Destroy()
            return None

    def _update_format_selector(self):
        choices = list(self._registered_formats.keys())
        choices.sort()
        self._format_selector.Items = choices

    def _select_format(self, event=None):
        fmt = self.current_format()
        self.file_dialog.SetWildcard(fmt.wildcard)
        w = fmt.window(self._options_panel) or self._no_options_window
        fmt.update(self.session, self)
        if w is self._current_option:
            return
        import wx
        self._current_option.Show(False)
        if w not in self._known_options:
            self._options_sizer.Add(w, proportion=1, flag=wx.EXPAND|wx.ALL|wx.ALIGN_CENTER,
                                    border=2)
            self._known_options.add(w)
        w.Show(True)
        self._options_sizer.Layout()
        self._options_panel.Fit()
        self._current_option = w

    def set_wildcard(self, format):
        fmt = self.current_format()
        self.file_dialog.SetWildcard(fmt.wildcard)


def _session_wildcard():
    from .. import io
    return io.wx_export_file_filter(io.SESSION)


def _session_save(session, filename):
    import os.path
    ext = os.path.splitext(filename)[1]
    from .. import io
    exts = io.extensions("ChimeraX session")
    if exts and ext not in exts:
        filename += exts[0]
    # TODO: generate text command instead of calling function directly
    # so that command logging happens automatically
    from ..commands import export
    export.export(session, filename)
    session.logger.info("File \"%s\" saved." % filename)


class ImageSaver:

    DEFAULT_FORMAT = "PNG"
    DEFAULT_EXT = "png"
    SUPERSAMPLE_OPTIONS = (("None", None),
                           ("2x2", 2),
                           ("3x3", 3),
                           ("4x4", 4))

    def __init__(self, save_dialog):
        import weakref
        self._save_dialog = weakref.ref(save_dialog)

    def wildcard(self):
        from .. import commands
        exts = list(commands.image_formats.keys())
        exts.remove(self.DEFAULT_EXT)
        exts.insert(0, self.DEFAULT_EXT)
        fmts = ';'.join("*.%s" % e for e in exts)
        wildcard = "Image file (%s)|%s" % (fmts, fmts)
        return wildcard

    def make_ui(self, parent):
        import wx
        from .. import commands
        w = wx.Window(parent, style=wx.BORDER_SIMPLE)
        s = wx.FlexGridSizer(rows=3, cols=2, hgap=2, vgap=2)

        selector = wx.Choice(w, choices=list(commands.image_formats.values()),
                             style=wx.CB_READONLY)
        selector.Bind(wx.EVT_CHOICE, self._select_format)
        selector.SetSelection(selector.Items.index(self.DEFAULT_FORMAT))
        s.Add(wx.StaticText(w, label="Format:"), flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        s.Add(selector, proportion=1, flag=wx.ALL|wx.ALIGN_LEFT, border=5)
        self._format_selector = selector

        size_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._width = wx.TextCtrl(w)
        self._height = wx.TextCtrl(w)
        # debugging code:
        # self._width.Bind(wx.EVT_TEXT, self.EvtText)
        # self._width.Bind(wx.EVT_CHAR, self.EvtChar)
        size_sizer.Add(self._width, proportion=1)
        size_sizer.Add(wx.StaticText(w, label="x"), flag=wx.LEFT|wx.RIGHT, border=4)
        size_sizer.Add(self._height, proportion=1)
        s.Add(wx.StaticText(w, label="Size:"), flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        s.Add(size_sizer, proportion=1, flag=wx.ALL|wx.ALIGN_LEFT, border=5)

        ss = wx.Choice(w, choices=[o[0] for o in self.SUPERSAMPLE_OPTIONS], style=wx.CB_READONLY)
        s.Add(wx.StaticText(w, label="Supersample:"), flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL)
        s.Add(ss, proportion=1, flag=wx.ALL|wx.ALIGN_LEFT, border=5)
        self._supersample = ss

        w.SetSizerAndFit(s)
        parent.Fit()
        return w

    def EvtText(self, event):
        import sys
        print("EvtText", event, event.GetString(), file=sys.__stderr__)
        event.Skip()

    def EvtChar(self, event):
        import sys
        print("EvtChar", event, event.GetKeyCode(), file=sys.__stderr__)
        event.Skip()

    def _get_current_extension(self):
        format_name = self._format_selector.GetString(self._format_selector.Selection)
        from .. import commands
        for e, n in commands.image_formats.items():
            if n == format_name:
                return e
        else:
            raise RuntimeError("unsupported graphics format: %s" % format_name)

    def _file_dialog(self):
        d = self._save_dialog()
        if d:
            return d.file_dialog
        else:
            return None

    def _select_format(self, event):
        # TODO: enable options that apply to this graphics format
        pass

    def update(self, session, save_dialog):
        gw = session.ui.main_window.graphics_window
        w, h = gw.GetSize()
        self._width.SetValue(str(w))
        self._height.SetValue(str(h))

    def save(self, session, filename):
        import os.path
        ext = os.path.splitext(filename)[1]
        e = '.' + self._get_current_extension()
        if ext != e:
            filename += e
        try:
            w = int(self._width.GetValue())
            h = int(self._height.GetValue())
        except ValueError:
            from ..errors import UserError
            raise UserError("width/height must be integers")
        if w <= 0 or h <= 0:
            from ..errors import UserError
            raise UserError("width/height must be positive integers")
        ss = self.SUPERSAMPLE_OPTIONS[self._supersample.GetSelection()][1]
        # TODO: generate text command instead of calling function directly
        # so that command logging happens automatically
        from .. import commands
        commands.save_image(session, filename, width=w, height=h, supersample=ss)

    def register(self):
        self._save_dialog().register("Image File", self.wildcard, self.make_ui,
                                     self.update, self.save)
