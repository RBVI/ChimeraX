# vi: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance


class CommandLine(ToolInstance):

    SESSION_ENDURING = True
    SIZE = (500, 25)
    VERSION = 1

    show_history_label = "Command History..."
    compact_label = "Remove duplicate consecutive commands"

    def __init__(self, session, tool_info, **kw):
        super().__init__(session, tool_info, **kw)
        from chimera.core.ui import MainToolWindow
        class CmdWindow(MainToolWindow):
            close_destroys = False
        self.tool_window = CmdWindow(self, size=self.SIZE)
        parent = self.tool_window.ui_area
        import wx
        self.text = wx.ComboBox(parent, size=self.SIZE,
                                style=wx.TE_PROCESS_ENTER | wx.TE_NOHIDESEL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.text, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.history_dialog = _HistoryDialog(self)
        self.text.Bind(wx.EVT_COMBOBOX, self.on_combobox)
        self.text.Bind(wx.EVT_TEXT_ENTER, self.on_enter)
        self.text.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.tool_window.manage(placement="bottom")
        self.history_dialog.populate()
        session.ui.register_for_keystrokes(self)
        session.tools.add([self])
        self._last_thumb = None
        # since only TextCtrls have the EmulateKeyPress method,
        # create a completely hidden TextCtrl so that we can
        # process forwarded keystrokes and copy the result back
        # into the ComboBox!
        self.kludge_text = wx.TextCtrl(parent)
        self.kludge_text.Hide()

    def cmd_clear(self):
        self.text.SetValue("")

    def cmd_replace(self, cmd):
        self.text.SetValue(cmd)
        self.text.SetInsertionPointEnd()

    def delete(self):
        self.session.ui.deregister_for_keystrokes(self)
        super().delete()

    def forwarded_keystroke(self, event):
        if event.KeyCode == 13:          # Return
            self.on_enter(event)
        elif event.KeyCode == 14:        # Ctrl-N
            self.history_dialog.down(event.GetModifiers() & wx.MOD_SHIFT)
        elif event.KeyCode == 16:        # Ctrl-P
            self.history_dialog.up(event.GetModifiers() & wx.MOD_SHIFT)
        elif event.KeyCode == 21:        # Ctrl-U
            self.cmd_clear()
        elif event.KeyCode == 27:         # Escape
            self.session.keyboard_shortcuts.enable_shortcuts()
        elif event.KeyCode == 315:        # Up arrow
            self.session.selection.promote()
        elif event.KeyCode == 317:        # Down arrow
            self.session.selection.demote()
        else:
            # Only TextCtrls handle forwarded keystroke events,
            # so copy the current ComboBox text state into a
            # TextCtrl, apply the forwarded keystroke, and copy
            # the result back (ugh)
            self.kludge_text.Value = self.text.Value
            self.kludge_text.SetInsertionPoint(self.text.InsertionPoint)
            self.kludge_text.SetSelection(*self.text.GetTextSelection())
            self.kludge_text.EmulateKeyPress(event)
            self.text.Value = self.kludge_text.Value
            self.text.SetInsertionPoint(self.kludge_text.InsertionPoint)
            self.text.SetSelection(*self.kludge_text.GetSelection())

    def on_combobox(self, event):
        val = self.text.GetValue()
        if val == self.show_history_label:
            self.cmd_clear()
            self.history_dialog.window.shown = True
        elif val == self.compact_label:
            self.cmd_clear()
            prev_cmd = None
            unique_cmds = []
            for cmd in self.history_dialog.history:
                if cmd != prev_cmd:
                    unique_cmds.append(cmd)
                    prev_cmd = cmd
            self.history_dialog.history.replace(unique_cmds)
            self.history_dialog.populate()
        else:
            event.Skip()

    def on_enter(self, event):
        session = self.session
        logger = session.logger
        text = self.text.Value
        logger.status("")
        from chimera.core import errors
        from chimera.core.commands import Command
        from html import escape
        for cmd_text in text.split("\n"):
            if not cmd_text:
                continue
            try:
                cmd = Command(session, cmd_text, final=True)
                cmd.error_check()
                cmd.execute()
            except SystemExit:
                # TODO: somehow quit application
                raise
            except errors.UserError as err:
                rest = cmd.current_text[cmd.amount_parsed:]
                spaces = len(rest) - len(rest.lstrip())
                error_at = cmd.amount_parsed + spaces
                syntax_error = error_at < len(cmd.current_text)
                # error message in red text
                err_color = 'crimson'
                err_text = '<span style="color:%s;">%s</span>\n' % (
                    err_color, escape(str(err)))
                if syntax_error:
                    err_text = '<p>%s<span style="color:white; background-color:%s;">%s</span><br>\n' % (
                        escape(cmd.current_text[:error_at]), err_color,
                        escape(cmd.current_text[error_at:])) + err_text
                logger.info(err_text, is_html=True)
                logger.status(str(err))
            except:
                import traceback
                session.logger.error(traceback.format_exc())
            else:
                self.history_dialog.add(cmd_text)
                thumb = session.main_view.image(width=100, height=100)
                log_thumb = False
                if thumb.getcolors(1) is None:
                    # image not just a solid background color;
                    # ensure it differs from previous thumbnail
                    thumb_data = thumb.tobytes()
                    if thumb_data != self._last_thumb:
                        self._last_thumb = thumb_data
                        log_thumb = True
                else:
                    self._last_thumb = None
                if log_thumb:
                    session.logger.info("graphics image", image=thumb)
        self.text.SetValue(cmd_text)
        self.text.SelectAll()

    def on_key_down(self, event):
        import wx
        # prevent combobox from responding to up/down arrow key
        # (opening/closing dropdown listbox), and handle it as
        # history forward/back, as well as getting other relevant
        # control-key events before the ComboBox KeyDown handler
        # consumes them
        shifted = event.GetModifiers() & wx.MOD_SHIFT
        if event.KeyCode == 315:  # up arrow
            self.history_dialog.up(shifted)
        elif event.KeyCode == 317:  # down arrow
            self.history_dialog.down(shifted)
        elif event.GetModifiers() & wx.MOD_RAW_CONTROL:
            if event.KeyCode == 78:
                self.history_dialog.down(shifted)
            elif event.KeyCode == 80:
                self.history_dialog.up(shifted)
            elif event.KeyCode == 85:
                self.cmd_clear()
            else:
                event.Skip()
        else:
            event.Skip()

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, phase, session, flags):
        if phase != self.SAVE_PHASE:
            return
        version = self.VERSION
        data = {"shown": self.tool_window.shown}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import RestoreError
        if version != self.VERSION:
            raise RestoreError("unexpected version")
        if phase == self.CREATE_PHASE:
            # All the action is in phase 2 because we do not
            # want to restore until all objects have been resolved
            pass
        else:
            self.display(data["shown"])

    def reset_state(self):
        self.tool_window.shown = True


class _HistoryDialog:

    record_label = "Record..."
    execute_label = "Execute"

    NUM_REMEMBERED = 500

    def __init__(self, controller):
        # make dialog hidden initially
        self.controller = controller
        from chimera.core.ui import ChildToolWindow
        class HistoryWindow(ChildToolWindow):
            close_destroys = False
        self.window = controller.tool_window.create_child_window(
            "Command History", window_class=HistoryWindow)

        parent = self.window.ui_area
        import wx
        self.listbox = wx.ListBox(parent, size=(100, 400),
                                  style=wx.LB_EXTENDED | wx.LB_NEEDED_SB)
        self.listbox.Bind(wx.EVT_LISTBOX, self.on_listbox)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.listbox, 1, wx.EXPAND)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(button_sizer)
        record_button = wx.Button(parent, label=self.record_label)
        button_sizer.Add(record_button)
        button_sizer.Add(wx.Button(parent, label=self.execute_label))
        for stock_id in [wx.ID_DELETE, wx.ID_COPY, wx.ID_HELP]:
            sz = button_sizer.Add(wx.Button(parent, id=stock_id))
        sz.GetWindow().Disable()
        parent.SetSizerAndFit(main_sizer)
        parent.Bind(wx.EVT_BUTTON, self.button_cb)
        self.window.manage(placement=None)
        self.window.shown = False
        from chimera.core.history import FIFOHistory
        self.history = FIFOHistory(self.NUM_REMEMBERED, controller.session, "command line")
        self._record_dialog = None

    def add(self, item):
        self.listbox.Append(item)
        while self.listbox.GetCount() > self.NUM_REMEMBERED:
            self.listbox.Delete(0)
        self.history.enqueue(item)
        for sel in self.listbox.GetSelections():
            self.listbox.Deselect(sel)
        self.listbox.SetSelection(len(self.history) - 1)
        self.update_list()

    def button_cb(self, event):
        label = event.GetEventObject().GetLabelText()
        import wx
        if label == self.record_label:
            from chimera.core.io import extensions
            ext = extensions("Chimera")[0]
            wc = "Chimera commands (*{})|*{}".format(ext, ext)
            from chimera.core.ui.open_save import SaveDialog
            from chimera.core.io import open_filename, extensions
            if self._record_dialog is None:
                self._record_dialg = dlg = SaveDialog(
                    self.window.ui_area, "Record Commands",
                    wildcard=wc, add_extension=extensions("Chimera")[0])
                dlg.SetExtraControlCreator(self._record_customize_cb)
            else:
                dlg = self._record_dialog
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()
            if not path:
                from chimera.core.errors import UserError
                raise UserError("No file specified for saving command history")
            if self.save_amount_Choice.GetStringSelection() == "all":
                cmds = [cmd for cmd in self.history]
            else:
                cmds = [self.listbox.GetString(x) for x in self.listbox.GetSelections()]
            if self.save_append_CheckBox.Value:
                mode = 'a'
            else:
                mode = 'w'
            f = open_filename(path, mode)
            for cmd in cmds:
                print(cmd, file=f)
            f.close()
            return
        if label == self.execute_label:
            for index in self.listbox.GetSelections():
                self.controller.cmd_replace(self.listbox.GetString(index))
                self.controller.on_enter(None)
            return
        stock_id = event.GetEventObject().GetId()
        if stock_id == wx.ID_DELETE:
            self.history.replace([self.history[i]
                                  for i in range(len(self.history))
                                  if i not in self.listbox.GetSelections()])
            self.populate()
            return
        if stock_id == wx.ID_COPY:
            if not wx.TheClipboard.Open():
                self.controller.session.logger.error(
                    "Could not access the system clipboard")
                return
            wx.TheClipboard.SetData(wx.TextDataObject("\n".join(
                [self.listbox.GetString(i)
                 for i in self.listbox.GetSelections()])))
            wx.TheClipboard.Flush()
            wx.TheClipboard.Close()
            return
        if stock_id == wx.ID_HELP:
            # TODO
            return

    def down(self, shifted):
        sels = self.listbox.GetSelections()
        if len(sels) != 1:
            return
        orig_sel = sel = sels[0]
        match_against = None
        if shifted:
            words = self.controller.text.Value.strip().split()
            if words:
                match_against = words[0]
        if match_against:
            last = self.listbox.GetCount() - 1
            while sel < last:
                if self.listbox.GetString(sel+1).startswith(match_against):
                    break
                sel += 1
        if sel == self.listbox.GetCount() - 1:
            return
        self.listbox.Deselect(orig_sel)
        self.listbox.SetSelection(sel + 1)
        self.controller.cmd_replace(self.listbox.GetString(sel + 1))

    def on_append_change(self, event):
        self.overwrite_disclaimer.Show(self.save_append_CheckBox.Value)

    def on_listbox(self, event):
        self.select()

    def populate(self):
        self.listbox.Items = [cmd for cmd in self.history]
        self.listbox.Selection = len(self.history) - 1
        self.update_list()
        self.select()
        self.controller.text.SetFocus()
        self.controller.text.SetSelection(-1, -1)
        cursels = self.listbox.GetSelections()
        if len(cursels) == 1:
            self.listbox.EnsureVisible(cursels[0])

    def select(self):
        sels = self.listbox.GetSelections()
        if len(sels) != 1:
            return
        self.controller.cmd_replace(self.listbox.GetString(sels[0]))

    def up(self, shifted):
        sels = self.listbox.GetSelections()
        if len(sels) != 1:
            return
        orig_sel = sel = sels[0]
        match_against = None
        if shifted:
            words = self.controller.text.Value.strip().split()
            if words:
                match_against = words[0]
        if match_against:
            while sel > 0:
                if self.listbox.GetString(sel-1).startswith(match_against):
                    break
                sel -= 1
        if sel == 0:
            return
        self.listbox.Deselect(orig_sel)
        self.listbox.SetSelection(sel - 1)
        self.controller.cmd_replace(self.listbox.GetString(sel - 1))

    def update_list(self):
        c = self.controller
        last8 = self.history[-8:]
        last8.reverse()
        c.text.Items = last8 + [c.show_history_label, c.compact_label]

    def _record_customize_cb(self, parent):
        import wx
        panel = wx.Panel(parent)
        amount_label1 = wx.StaticText(panel, label="Record")
        self.save_amount_Choice = amount = wx.Choice(panel, choices=["all", "selected"])
        amount.SetSelection(0)
        amount_label2 = wx.StaticText(panel, label="commands")
        amount_sizer = wx.BoxSizer(wx.HORIZONTAL)
        amount_sizer.Add(amount_label1)
        amount_sizer.Add(amount)
        amount_sizer.Add(amount_label2)
        row_sizer = wx.BoxSizer(wx.VERTICAL)
        row_sizer.Add(amount_sizer)
        self.save_append_CheckBox = cb = wx.CheckBox(panel, label="Append to file")
        cb.Bind(wx.EVT_CHECKBOX, self.on_append_change)
        cb.Value = False
        row_sizer.Add(cb, flag=wx.ALIGN_CENTER)
        self.overwrite_disclaimer = disclaimer = wx.StaticText(
            panel, label="(ignore overwrite warning)")
        disclaimer.SetLabelMarkup("<small><i>(ignore overwrite warning)</i></small>")
        disclaimer.Hide()
        row_sizer.Add(disclaimer, flag=wx.ALIGN_CENTER | wx.RESERVE_SPACE_EVEN_IF_HIDDEN)
        panel.SetSizerAndFit(row_sizer)
        return panel
