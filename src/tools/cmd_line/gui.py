# vim: set expandtab ts=4 sw=4:

from chimerax.core.tools import ToolInstance


class CommandLine(ToolInstance):

    SESSION_ENDURING = True

    show_history_label = "Command History..."
    compact_label = "Remove duplicate consecutive commands"
    help = "help:user/tools/cli.html"

    def __init__(self, session, bundle_info, *, restoring=False):
        if not restoring:
            ToolInstance.__init__(self, session, bundle_info)
        from chimerax.core.ui.gui import MainToolWindow

        class CmdWindow(MainToolWindow):
            close_destroys = False
        self.tool_window = CmdWindow(self)
        parent = self.tool_window.ui_area
        self.history_dialog = _HistoryDialog(self)
        from chimerax.core import window_sys
        self.window_sys = window_sys
        if window_sys == "qt":
            from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QLabel
            label = QLabel(parent)
            label.setText("Command:")
            self.text = QComboBox(parent)
            self.text.setEditable(True)
            layout = QHBoxLayout(parent)
            layout.setSpacing(1)
            layout.setContentsMargins(2, 0, 0, 0)
            layout.addWidget(label)
            layout.addWidget(self.text, 1)
            parent.setLayout(layout)
            self.text.lineEdit().returnPressed.connect(self.execute)
            self.text.lineEdit().editingFinished.connect(self.text.lineEdit().selectAll)
            self.text.currentTextChanged.connect(self.text_changed)
            session.ui.register_for_keystrokes(self.text)
        else:
            import wx
            SIZE = (500, 25)
            self.text = wx.ComboBox(parent, size=SIZE,
                                    style=wx.TE_PROCESS_ENTER | wx.TE_NOHIDESEL)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            label = wx.StaticText(parent, label="Command:")
            sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            sizer.Add(self.text, 1, wx.EXPAND)
            parent.SetSizerAndFit(sizer)
            self.text.Bind(wx.EVT_COMBOBOX, self.on_combobox)
            self.text.Bind(wx.EVT_TEXT, self.history_dialog._entry_modified)
            self.text.Bind(wx.EVT_TEXT_ENTER, lambda e: self.execute())
            self.text.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
            session.ui.register_for_keystrokes(self)
            # since only TextCtrls have the EmulateKeyPress method,
            # create a completely hidden TextCtrl so that we can
            # process forwarded keystrokes and copy the result back
            # into the ComboBox!
            self.kludge_text = wx.TextCtrl(parent)
            self.kludge_text.Hide()
        self.history_dialog.populate()
        self.tool_window.manage(placement="bottom")

    def cmd_clear(self):
        from chimerax.core import window_sys
        if window_sys == "qt":
            self.text.lineEdit().clear()
        else:
            self.text.SetValue("")

    def cmd_replace(self, cmd):
        from chimerax.core import window_sys
        if window_sys == "qt":
            line_edit = self.text.lineEdit()
            line_edit.setText(cmd)
            line_edit.setCursorPosition(len(cmd))
        else:
            self.text.SetValue(cmd)
            self.text.SetInsertionPointEnd()

    def delete(self):
        self.session.ui.deregister_for_keystrokes(self.text)
        super().delete()

    def forwarded_keystroke(self, event):
        import wx
        if event.KeyCode == 13:          # Return
            self.execute()
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
            if self.text.Value != self.kludge_text.Value:
                # prevent gratuituous 'text modified' events
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

    def text_changed(self, text):
        if text == self.show_history_label:
            self.cmd_clear()
            self.history_dialog.window.shown = True
        elif text == self.compact_label:
            self.cmd_clear()
            prev_cmd = None
            unique_cmds = []
            for cmd in self.history_dialog.history:
                if cmd != prev_cmd:
                    unique_cmds.append(cmd)
                    prev_cmd = cmd
            self.history_dialog.history.replace(unique_cmds)
            self.history_dialog.populate()

    def execute(self):
        session = self.session
        logger = session.logger
        if self.window_sys == "wx":
            text = self.text.Value
        else:
            text = self.text.lineEdit().text()
        logger.status("")
        from chimerax.core import errors
        from chimerax.core.commands import Command
        from html import escape
        for cmd_text in text.split("\n"):
            if not cmd_text:
                continue
            try:
                cmd = Command(session)
                cmd.run(cmd_text)
            except SystemExit:
                # TODO: somehow quit application
                raise
            except errors.UserError as err:
                rest = cmd.current_text[cmd.amount_parsed:]
                spaces = len(rest) - len(rest.lstrip())
                error_at = cmd.amount_parsed + spaces
                syntax_error = error_at < len(cmd.current_text)
                # error message in red text
                msg = '<div class="cxcmd">%s' % escape(
                        cmd.current_text[cmd.start:error_at])
                err_color = 'crimson'
                if syntax_error:
                    msg += '<span style="color:white; background-color:%s;">%s</span>' % (
                        err_color,
                        escape(cmd.current_text[error_at:]))
                msg += '</div>\n<span style="color:%s;">%s</span>\n' % (
                    err_color, escape(str(err)))
                logger.info(msg, is_html=True)
                logger.status(str(err))
            except:
                import traceback
                session.logger.error(traceback.format_exc())
            else:
                self.history_dialog.add(cmd_text)
        if self.window_sys == "wx":
            self.text.SetValue(cmd_text)
            self.text.SelectAll()
        else:
            self.text.lineEdit().setText(cmd_text)
            self.text.lineEdit().selectAll()

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
    def take_snapshot(self, session, flags):
        data = {"shown": self.tool_window.shown}
        return self.bundle_info.session_write_version, data

    @classmethod
    def restore_snapshot_new(cls, session, bundle_info, version, data):
        return cls.get_singleton(session)

    def restore_snapshot_init(self, session, bundle_info, version, data):
        if version not in bundle_info.session_versions:
            from chimerax.core.state import RestoreError
            raise RestoreError("unexpected version")
        self.display(data["shown"])

    def reset_state(self, session):
        self.tool_window.shown = True

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, CommandLine, 'cmd_line')

class _HistoryDialog:

    record_label = "Record..."
    execute_label = "Execute"

    NUM_REMEMBERED = 500

    def __init__(self, controller):
        # make dialog hidden initially
        self.controller = controller
        from chimerax.core.ui.gui import ChildToolWindow

        class HistoryWindow(ChildToolWindow):
            close_destroys = False
        self.window = controller.tool_window.create_child_window(
            "Command History", window_class=HistoryWindow)

        parent = self.window.ui_area
        from chimerax.core import window_sys
        self.window_sys = window_sys
        if window_sys == "qt":
            self.add = self.qt_add
            self.up = self.qt_up
            self.down = self.qt_down
            self.select = self.qt_select
            self.populate = self.qt_populate
            self.update_list = self.qt_update_list
            from PyQt5.QtWidgets import QListWidget, QVBoxLayout, QFrame, QHBoxLayout, QPushButton
            self.listbox = QListWidget(parent)
            self.listbox.setSelectionMode(QListWidget.ExtendedSelection)
            self.listbox.itemSelectionChanged.connect(self.select)
            main_layout = QVBoxLayout(parent)
            main_layout.addWidget(self.listbox)
            button_frame = QFrame(parent)
            main_layout.addWidget(button_frame)
            button_layout = QHBoxLayout(button_frame)
            for but_name in [self.record_label, self.execute_label, "Delete", "Copy", "Help"]:
                but = QPushButton(but_name, button_frame)
                but.setAutoDefault(False)
                but.clicked.connect(lambda arg, txt=but_name: self.button_clicked(txt))
                if but_name == "Help":
                    but.setDisabled(True)
                button_layout.addWidget(but)
            button_frame.setLayout(button_layout)
        else:
            self.add = self.wx_add
            self.up = self.wx_up
            self.down = self.wx_down
            self.select = self.wx_select
            self.populate = self.wx_populate
            self.update_list = self.wx_update_list
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
        from chimerax.core.history import FIFOHistory
        self.history = FIFOHistory(self.NUM_REMEMBERED, controller.session, "command line")
        self._record_dialog = None
        self._search_cache = None
        self._suspend_handler = False

    def wx_add(self, item):
        self.listbox.Append(item)
        while self.listbox.GetCount() > self.NUM_REMEMBERED:
            self.listbox.Delete(0)
        self.history.enqueue(item)
        for sel in self.listbox.GetSelections():
            self.listbox.Deselect(sel)
        self.listbox.SetSelection(len(self.history) - 1)
        self.update_list()

    def qt_add(self, item):
        self.listbox.addItem(item)
        while self.listbox.count() > self.NUM_REMEMBERED:
            self.listbox.removeItemWidget(self.listbox.item(0))
        self.history.enqueue(item)
        self.listbox.clearSelection()
        self.listbox.setCurrentRow(len(self.history) - 1)
        self.update_list()

    def button_cb(self, event):
        label = event.GetEventObject().GetLabelText()
        import wx
        if label == self.record_label:
            from chimerax.core.io import extensions
            ext = extensions("ChimeraX")[0]
            wc = "ChimeraX commands (*{})|*{}".format(ext, ext)
            from chimerax.core.ui.open_save import SaveDialog
            from chimerax.core.io import open_filename, extensions
            if self._record_dialog is None:
                self._record_dialog = dlg = SaveDialog(
                    self.window.ui_area, "Record Commands",
                    wildcard=wc, add_extension=extensions("ChimeraX")[0])
                dlg.SetExtraControlCreator(self._record_customize_cb)
            else:
                dlg = self._record_dialog
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()
            if not path:
                from chimerax.core.errors import UserError
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

    def button_clicked(self, label):
        if label == self.record_label:
            from chimerax.core.ui.open_save import export_file_filter, SaveDialog
            from chimerax.core.io import open_filename, extensions
            if self._record_dialog is None:
                self._record_dialog = dlg = SaveDialog(self.window.ui_area,
                    "Record Commands", name_filter=export_file_filter(format_name="ChimeraX"),
                    add_extension=extensions("ChimeraX")[0])
                from PyQt5.QtWidgets import QFrame, QLabel, QHBoxLayout, QVBoxLayout, QComboBox
                from PyQt5.QtWidgets import QCheckBox
                from PyQt5.QtCore import Qt
                options_frame = dlg.custom_area
                options_layout = QVBoxLayout(options_frame)
                options_frame.setLayout(options_layout)
                amount_frame = QFrame(options_frame)
                options_layout.addWidget(amount_frame, Qt.AlignCenter)
                amount_layout = QHBoxLayout(amount_frame)
                amount_layout.addWidget(QLabel("Record", amount_frame))
                self.save_amount_widget = saw = QComboBox(amount_frame)
                saw.addItems(["all", "selected"])
                amount_layout.addWidget(saw)
                amount_layout.addWidget(QLabel("commands", amount_frame))
                amount_frame.setLayout(amount_layout)
                self.append_checkbox = QCheckBox("Append to file", options_frame)
                self.append_checkbox.stateChanged.connect(self.append_changed)
                options_layout.addWidget(self.append_checkbox, Qt.AlignCenter)
                self.overwrite_disclaimer = disclaimer = QLabel(
                    "<small><i>(ignore overwrite warning)</i></small>", options_frame)
                options_layout.addWidget(disclaimer, Qt.AlignCenter)
                disclaimer.hide()
            else:
                dlg = self._record_dialog
            if not dlg.exec():
                return
            path = dlg.selectedFiles()[0]
            if not path:
                from chimerax.core.errors import UserError
                raise UserError("No file specified for saving command history")
            if self.save_amount_widget.currentText() == "all":
                cmds = [cmd for cmd in self.history]
            else:
                # listbox.selectedItems() may not be in order, so...
                items = [self.listbox.item(i) for i in range(self.listbox.count())
                    if self.listbox.item(i).isSelected()]
                cmds = [item.text() for item in items]
            if self.append_checkbox.isChecked():
                mode = 'a'
            else:
                mode = 'w'
            f = open_filename(path, mode)
            for cmd in cmds:
                print(cmd, file=f)
            f.close()
            return
        if label == self.execute_label:
            for item in self.listbox.selectedItems():
                self.controller.cmd_replace(item.text())
                self.controller.execute()
            return
        if label == "Delete":
            self.history.replace([self.listbox.item(i).text()
                for i in range(self.listbox.count()) if not self.listbox.item(i).isSelected()])
            self.populate()
            return
        if label == "Copy":
            clipboard = self.controller.session.ui.clipboard()
            clipboard.setText("\n".join([item.text() for item in self.listbox.selectedItems()]))
            return
        if label == "Help":
            # TODO
            return

    def wx_down(self, shifted):
        sels = self.listbox.GetSelections()
        if len(sels) != 1:
            return
        orig_sel = sel = sels[0]
        orig_text = self.controller.text.Value
        match_against = None
        self._suspend_handler = False
        if shifted:
            if self._search_cache is None:
                words = orig_text.strip().split()
                if words:
                    match_against = words[0]
                    self._search_cache = match_against
            else:
                match_against = self._search_cache
            self._suspend_handler = True
        if match_against:
            last = self.listbox.GetCount() - 1
            while sel < last:
                if self.listbox.GetString(sel + 1).startswith(match_against):
                    break
                sel += 1
        if sel == self.listbox.GetCount() - 1:
            return
        self.listbox.Deselect(orig_sel)
        self.listbox.SetSelection(sel + 1)
        new_text = self.listbox.GetString(sel + 1)
        self.controller.cmd_replace(new_text)
        if orig_text == new_text:
            self.down(shifted)
        self._suspend_handler = False

    def qt_down(self, shifted):
        sels = self.listbox.selectedIndexes()
        if len(sels) != 1:
            return
        sel = sels[0]
        orig_text = self.controller.text.text()
        match_against = None
        self._suspend_handler = False
        if shifted:
            if self._search_cache is None:
                words = orig_text.strip().split()
                if words:
                    match_against = words[0]
                    self._search_cache = match_against
            else:
                match_against = self._search_cache
            self._suspend_handler = True
        if match_against:
            last = self.listbox.count() - 1
            while sel < last:
                if self.listbox.item(sel + 1).text().startswith(match_against):
                    break
                sel += 1
        if sel == self.listbox.count() - 1:
            return
        self.listbox.clearSelection()
        self.listbox.setCurrentIndex(sel + 1)
        new_text = self.listbox.item(sel + 1).text()
        self.controller.cmd_replace(new_text)
        if orig_text == new_text:
            self.down(shifted)
        self._suspend_handler = False

    def on_append_change(self, event):
        self.overwrite_disclaimer.Show(self.save_append_CheckBox.Value)

    def append_changed(self, append):
        if append:
            self.overwrite_disclaimer.show()
        else:
            self.overwrite_disclaimer.hide()

    def on_listbox(self, event):
        self.select()

    def wx_populate(self):
        self.listbox.Items = [cmd for cmd in self.history]
        self.listbox.Selection = len(self.history) - 1
        self.update_list()
        self.select()
        self.controller.text.SetFocus()
        self.controller.text.SetSelection(-1, -1)
        cursels = self.listbox.GetSelections()
        if len(cursels) == 1:
            self.listbox.EnsureVisible(cursels[0])

    def qt_populate(self):
        self.listbox.clear()
        self.listbox.addItems([cmd for cmd in self.history])
        self.listbox.setCurrentRow(len(self.history) - 1)
        self.update_list()
        self.select()
        self.controller.text.lineEdit().setFocus()
        self.controller.text.lineEdit().selectAll()
        cursels = self.listbox.scrollToBottom()

    def wx_select(self):
        sels = self.listbox.GetSelections()
        if len(sels) != 1:
            return
        self.controller.cmd_replace(self.listbox.GetString(sels[0]))

    def qt_select(self):
        sels = self.listbox.selectedItems()
        if len(sels) != 1:
            return
        self.controller.cmd_replace(sels[0].text())

    def wx_up(self, shifted):
        sels = self.listbox.GetSelections()
        if len(sels) != 1:
            return
        orig_sel = sel = sels[0]
        orig_text = self.controller.text.Value
        match_against = None
        self._suspend_handler = False
        if shifted:
            if self._search_cache is None:
                words = orig_text.strip().split()
                if words:
                    match_against = words[0]
                    self._search_cache = match_against
            else:
                match_against = self._search_cache
            self._suspend_handler = True
        if match_against:
            while sel > 0:
                if self.listbox.GetString(sel - 1).startswith(match_against):
                    break
                sel -= 1
        if sel == 0:
            return
        self.listbox.Deselect(orig_sel)
        self.listbox.SetSelection(sel - 1)
        new_text = self.listbox.GetString(sel - 1)
        self.controller.cmd_replace(new_text)
        if orig_text == new_text:
            self.up(shifted)
        self._suspend_handler = False

    def qt_up(self, shifted):
        sels = self.listbox.selectedIndexes()
        if len(sels) != 1:
            return
        sel = sels[0]
        orig_text = self.controller.text.text()
        match_against = None
        self._suspend_handler = False
        if shifted:
            if self._search_cache is None:
                words = orig_text.strip().split()
                if words:
                    match_against = words[0]
                    self._search_cache = match_against
            else:
                match_against = self._search_cache
            self._suspend_handler = True
        if match_against:
            while sel > 0:
                if self.listbox.item(sel - 1).text().startswith(match_against):
                    break
                sel -= 1
        if sel == 0:
            return
        self.listbox.clearSelection()
        self.listbox.setCurrentRow(sel - 1)
        new_text = self.listbox.item(sel - 1).text()
        self.controller.cmd_replace(new_text)
        if orig_text == new_text:
            self.up(shifted)
        self._suspend_handler = False

    def wx_update_list(self):
        c = self.controller
        last8 = self.history[-8:]
        last8.reverse()
        c.text.Items = last8 + [c.show_history_label, c.compact_label]

    def qt_update_list(self):
        c = self.controller
        last8 = self.history[-8:]
        last8.reverse()
        c.text.clear()
        c.text.addItems(last8 + [c.show_history_label, c.compact_label])

    def _entry_modified(self, event):
        if not self._suspend_handler:
            self._search_cache = None
        event.Skip()

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
