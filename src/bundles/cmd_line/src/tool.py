# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance


class CommandLine(ToolInstance):

    SESSION_ENDURING = True

    show_history_label = "Command History..."
    compact_label = "Remove duplicate consecutive commands"
    help = "help:user/tools/cli.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self._in_init = True
        from .settings import settings
        self.settings = settings
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, hide_title_bar=True)
        parent = self.tool_window.ui_area
        self.tool_window.fill_context_menu = self.fill_context_menu
        self.history_dialog = _HistoryDialog(self, self.settings.typed_only)
        from Qt.QtWidgets import QComboBox, QHBoxLayout, QLabel
        label = QLabel(parent)
        label.setText("Command:")
        import sys
        class CmdText(QComboBox):
            def __init__(self, parent, tool):
                self.tool = tool
                QComboBox.__init__(self, parent)
                self._processing_key = False
                from Qt.QtCore import Qt
                # defer context menu to parent
                self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
                self.setAcceptDrops(True)
                self._out_selection = None
                # horrible hack needed for Linux...
                self._drop_hack = False

            def dragEnterEvent(self, event):
                if event.mimeData().text():
                    event.acceptProposedAction()
                    if sys.platform == "linux" and not self._drop_hack:
                        if "file://" not in self.lineEdit().text():
                            self._drop_hack = True
                            self.editTextChanged.connect(self._drop_hack_cb)

            # On Linux, not only do you not get the drop event, but
            # dragLeaveEvent is called immediately after dragEnterEvent,
            # before the release for the drop has even happened, so can't
            # depend on dragLeaveEvent to make the hack more reliable
            """
            def dragLeaveEvent(self, event):
                if self._drop_hack:
                    self._drop_hack = False
                    self.editTextChanged.disconnect(self._drop_hack_cb)
            """

            def dropEvent(self, event):
                from urllib.parse import unquote
                text = unquote(event.mimeData().text())
                if text.startswith("file://"):
                    text = text[7:]
                    if sys.platform.startswith("win") and text.startswith('/'):
                        # Windows seems to provide /C:/...
                        text = text[1:]
                from chimerax.core.commands import StringArg
                self.lineEdit().insert(StringArg.unparse(text))
                event.acceptProposedAction()

            def _drop_hack_cb(self, new_text):
                self._drop_hack = False
                self.editTextChanged.disconnect(self._drop_hack_cb)
                if "file://" in new_text:
                    # Since we are getting the text of the entire line
                    # rather than just the dropped text, we have no
                    # ability to quotes spaces properly if needed
                    fixed_up = "".join([c for c in new_text.replace(
                        "file://", "") if c.isprintable()])
                    self.lineEdit().setText(fixed_up)

            def focusInEvent(self, event):
                self._out_selection = None
                QComboBox.focusInEvent(self, event)

            def focusOutEvent(self, event):
                le = self.lineEdit()
                self._out_selection = (sel_start, sel_length, txt) = (le.selectionStart(),
                    len(le.selectedText()), le.text())
                QComboBox.focusOutEvent(self, event)
                if sel_start >= 0:
                    le.setSelection(sel_start, sel_length)

            def keyPressEvent(self, event, forwarded=False):
                self._processing_key = True
                from Qt.QtCore import Qt
                from Qt.QtGui import QKeySequence

                if session.ui.key_intercepted(event.key()):
                    return
                
                want_focus = forwarded and event.key() not in [Qt.Key.Key_Control,
                                                               Qt.Key.Key_Shift,
                                                               Qt.Key.Key_Meta,
                                                               Qt.Key.Key_Alt]
                import sys
                control_key = Qt.KeyboardModifier.MetaModifier if sys.platform == "darwin" else Qt.KeyboardModifier.ControlModifier
                shifted = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                if event.key() == Qt.Key.Key_Up:  # up arrow
                    self.tool.history_dialog.up(shifted)
                elif event.key() == Qt.Key.Key_Down:  # down arrow
                    self.tool.history_dialog.down(shifted)
                elif event.matches(QKeySequence.StandardKey.Undo):
                    want_focus = False
                    session.undo.undo()
                elif event.matches(QKeySequence.StandardKey.Redo):
                    want_focus = False
                    session.undo.redo()
                elif event.modifiers() & control_key:
                    if event.key() == Qt.Key.Key_N:
                        self.tool.history_dialog.down(shifted)
                    elif event.key() == Qt.Key.Key_P:
                        self.tool.history_dialog.up(shifted)
                    elif event.key() == Qt.Key.Key_U:
                        self.tool.cmd_clear()
                        self.tool.history_dialog.search_reset()
                    elif event.key() == Qt.Key.Key_K:
                        self.tool.cmd_clear_to_end_of_line()
                        self.tool.history_dialog.search_reset()
                    else:
                        QComboBox.keyPressEvent(self, event)
                else:
                    QComboBox.keyPressEvent(self, event)
                if want_focus:
                    # Give command line the focus, so that up/down arrow work as
                    # expected rather than changing the selection level
                    self.setFocus()
                self._processing_key = False

            def sizeHint(self):
                # prevent super-long commands from making the whole interface super wide
                return self.minimumSizeHint()

        self.text = CmdText(parent, self)
        self.text.setEditable(True)
        self.text.setCompleter(None)
        def sel_change_correction():
            # don't allow selection to change while focus is out
            if self.text._out_selection is not None:
                start, length, text = self.text._out_selection
                le = self.text.lineEdit()
                if text != le.text():
                    self.text._out_selection = (le.selectionStart(), len(le.selectedText()), le.text())
                    return
                if start >= 0 and (start, length) != (le.selectionStart(), len(le.selectedText())):
                    le.setSelection(start, length)
        self.text.lineEdit().selectionChanged.connect(sel_change_correction)
        # pastes can have a trailing newline, which is problematic when appending to the pasted command...
        def strip_trailing_newlines():
            le = self.text.lineEdit()
            while le.text().endswith('\n'):
                le.setText(le.text()[:-1])
        self.text.lineEdit().textEdited.connect(strip_trailing_newlines)
        self.text.lineEdit().textEdited.connect(self.history_dialog.search_reset)
        def text_change(*args):
            # if text changes while focus is out, remember new selection
            if self.text._out_selection is not None:
                le = self.text.lineEdit()
                self.text._out_selection = (le.selectionStart(), len(le.selectedText()), le.text())
        self.text.lineEdit().selectionChanged.connect(text_change)
        layout = QHBoxLayout(parent)
        layout.setSpacing(1)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(self.text, 1)
        parent.setLayout(layout)
        # lineEdit() seems to be None during entire CmdText constructor, so connect here...
        self.text.lineEdit().returnPressed.connect(self.execute)
        self.text.currentTextChanged.connect(self.text_changed)
        self.text.forwarded_keystroke = lambda e: self.text.keyPressEvent(e, forwarded=True)
        session.ui.register_for_keystrokes(self.text)
        self.history_dialog.populate()
        self._just_typed_command = None
        self._in_open_command = 0
        self._handlers = []
        self._handlers.append(session.triggers.add_handler("command started", self._command_started_cb))
        self._handlers.append(session.triggers.add_handler("command failed", self._command_ended_cb))
        self._handlers.append(session.triggers.add_handler("command finished", self._command_ended_cb))
        self.tool_window.manage(placement="bottom")
        self._in_init = False
        self._processing_command = False
        if self.settings.startup_commands:
            # prevent the startup command output from being summarized into 'startup messages' table
            self._handlers.append(session.ui.triggers.add_handler('ready', self._run_startup_commands))

    def cmd_clear(self):
        self.text.lineEdit().clear()

    def cmd_clear_to_end_of_line(self):
        le = self.text.lineEdit()
        t = le.text()[:le.cursorPosition()]
        le.setText(t)

    def cmd_replace(self, cmd):
        line_edit = self.text.lineEdit()
        line_edit.setText(cmd)
        line_edit.setCursorPosition(len(cmd))

    def delete(self):
        self.session.ui.deregister_for_keystrokes(self.text)
        for handler in self._handlers:
            handler.remove()
        super().delete()

    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from Qt.QtGui import QAction
        filter_action = QAction("Typed Commands Only", menu)
        filter_action.setCheckable(True)
        filter_action.setChecked(self.settings.typed_only)
        filter_action.toggled.connect(lambda arg, f=self._set_typed_only: f(arg))
        menu.addAction(filter_action)
        select_action = QAction("Leave Failed Command Highlighted", menu)
        select_action.setCheckable(True)
        select_action.setChecked(self.settings.select_failed)
        select_action.toggled.connect(lambda arg, f=self._set_select_failed: f(arg))
        menu.addAction(select_action)

    def on_combobox(self, event):
        val = self.text.GetValue()
        if val == self.show_history_label:
            self.cmd_clear()
            self.history_dialog.window.shown = True
        elif val == self.compact_label:
            self.cmd_clear()
            prev_cmd = None
            unique_cmds = []
            for cmd in self.history_dialog._history:
                if cmd != prev_cmd:
                    unique_cmds.append(cmd)
                    prev_cmd = cmd
            self.history_dialog._history.replace(unique_cmds)
            self.history_dialog.populate()
        else:
            event.Skip()

    def text_changed(self, text):
        if text == self.show_history_label:
            self.cmd_clear()
            if not self._in_init:
                self.history_dialog.window.shown = True
        elif text == self.compact_label:
            self.cmd_clear()
            prev_cmd = None
            unique_cmds = []
            for cmd in self.history_dialog._history:
                if cmd != prev_cmd:
                    unique_cmds.append(cmd)
                    prev_cmd = cmd
            self.history_dialog._history.replace(unique_cmds)
            self.history_dialog.populate()

    def execute(self):
        from contextlib import contextmanager
        @contextmanager
        def processing_command(line_edit, cmd_text, command_worked, select_failed):
            line_edit.blockSignals(True)
            self._processing_command = True
            # as per the docs for contextmanager, the yield needs
            # to be in a try/except if the exit code is to execute
            # after errors
            try:
                yield
            finally:
                line_edit.blockSignals(False)
                line_edit.setText(cmd_text)
                if command_worked[0] or select_failed:
                    line_edit.selectAll()
                self._processing_command = False
        session = self.session
        logger = session.logger
        text = self.text.lineEdit().text()
        logger.status("")
        from chimerax.core import errors
        from chimerax.core.commands import Command
        from html import escape
        for cmd_text in text.split("\n"):
            if not cmd_text:
                continue
            # don't select the text if the command failed, so that
            # an accidental keypress won't erase the command, which
            # probably needs to be edited to work
            command_worked = [False]
            with processing_command(self.text.lineEdit(), cmd_text, command_worked,
                    self.settings.select_failed):
                try:
                    self._just_typed_command = cmd_text
                    cmd = Command(session)
                    cmd.run(cmd_text)
                    command_worked[0] = True
                except SystemExit:
                    # TODO: somehow quit application
                    raise
                except errors.UserError as err:
                    logger.status(str(err), color="crimson")
                    from chimerax.core.logger import error_text_format
                    logger.info(error_text_format % escape(str(err)), is_html=True)
                except BaseException:
                    raise
        self.set_focus()

    def set_focus(self):
        from Qt.QtCore import Qt
        self.text.lineEdit().setFocus(Qt.FocusReason.OtherFocusReason)

    @classmethod
    def get_singleton(cls, session, **kw):
        from chimerax.core import tools
        return tools.get_singleton(session, CommandLine, 'Command Line Interface', **kw)

    def _command_started_cb(self, trig_name, cmd_text):
        # the self._processing_command test is necessary when multiple commands
        # separated by semicolons are typed in order to prevent putting the 
        # second and later commands into the command history, since we will get 
        # triggers for each command in the line
        if cmd_text.startswith("open ") and ".cxc" in cmd_text:
            # Kludge to try to just put commands from .cxc scripts into the command history.
            self._in_open_command += 1
        if self._just_typed_command or not self._processing_command or self._in_open_command:
            self.history_dialog.add(self._just_typed_command or cmd_text,
                typed=self._just_typed_command is not None)
            self.text.lineEdit().selectAll()
            self._just_typed_command = None

    def _command_ended_cb(self, trig_name, cmd_text):
        if cmd_text.startswith("open ") and ".cxc" in cmd_text:
            self._in_open_command -= 1

    def _run_startup_commands(self, *args):
        # log the commands; but prevent them from going into command history...
        self._processing_command = True
        from chimerax.core.commands import run
        from chimerax.core.errors import UserError
        try:
            for cmd_text in self.settings.startup_commands:
                run(self.session, cmd_text)
        except UserError as err:
            self.session.logger.status("Error running startup command '%s': %s" % (cmd_text, str(err)),
                color="crimson", log=True)
        except Exception:
            self._processing_command = False
            raise
        self._processing_command = False

    def _set_select_failed(self, select_failed):
        self.settings.select_failed = select_failed

    def _set_typed_only(self, typed_only):
        self.settings.typed_only = typed_only
        self.history_dialog.set_typed_only(typed_only)

class _HistoryDialog:

    record_label = "Save..."
    execute_label = "Execute"

    def __init__(self, controller, typed_only):
        # make dialog hidden initially
        self.controller = controller
        self.typed_only = typed_only

        self.window = controller.tool_window.create_child_window(
            "Command History", close_destroys=False)
        self.window.fill_context_menu = self.fill_context_menu

        parent = self.window.ui_area
        from Qt.QtWidgets import QListWidget, QVBoxLayout, QFrame, QHBoxLayout, QPushButton, QLabel
        self.listbox = QListWidget(parent)
        self.listbox.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.listbox.itemSelectionChanged.connect(self.select)
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(self.listbox)
        num_cmd_frame = QFrame(parent)
        main_layout.addWidget(num_cmd_frame)
        num_cmd_layout = QHBoxLayout(num_cmd_frame)
        num_cmd_layout.setContentsMargins(0,0,0,0)
        remem_label = QLabel("Remember")
        from Qt.QtCore import Qt
        remem_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        num_cmd_layout.addWidget(remem_label, 1)
        from Qt.QtWidgets import QSpinBox, QSizePolicy
        class ShorterQSpinBox(QSpinBox):
            max_val = 1000000
            def textFromValue(self, val):
                # kludge to make the damn entry field shorter
                if val == self.max_val:
                    return "1 mil"
                return str(val)

        spin_box = ShorterQSpinBox()
        spin_box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        spin_box.setRange(100, spin_box.max_val)
        spin_box.setSingleStep(100)
        spin_box.setValue(controller.settings.num_remembered)
        spin_box.valueChanged.connect(self._num_remembered_changed)
        num_cmd_layout.addWidget(spin_box, 0)
        num_cmd_layout.addWidget(QLabel("commands"), 1)
        num_cmd_frame.setLayout(num_cmd_layout)
        button_frame = QFrame(parent)
        main_layout.addWidget(button_frame)
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0,0,0,0)
        for but_name in [self.record_label, self.execute_label, "Delete", "Copy", "Help"]:
            but = QPushButton(but_name, button_frame)
            but.setAutoDefault(False)
            but.clicked.connect(lambda *args, txt=but_name: self.button_clicked(txt))
            button_layout.addWidget(but)
        button_frame.setLayout(button_layout)
        self.window.manage(placement=None, initially_hidden=True)
        from chimerax.core.history import FIFOHistory
        self._history = FIFOHistory(controller.settings.num_remembered, controller.session, "commands")
        self._record_dialog = None
        self._search_cache = (False, None)

    def add(self, item, *, typed=False):
        if len(self._history) >= self.controller.settings.num_remembered:
            if not self.typed_only or self._history[0][1]:
                self.listbox.takeItem(0)
        if typed or not self.typed_only:
            self.listbox.addItem(item)
        self._history.enqueue((item, typed))
        # 'if typed:' to avoid clearing any partially entered command text
        if typed:
            self.listbox.clearSelection()
            self.listbox.setCurrentRow(len(self.history()) - 1)
            self.update_list()

    def button_clicked(self, label):
        session = self.controller.session
        if label == self.record_label:
            from chimerax.ui.open_save import SaveDialog
            if self._record_dialog is None:
                fmt = session.data_formats["ChimeraX commands"]
                self._record_dialog = dlg = SaveDialog(session, self.window.ui_area,
                    "Save Commands", data_formats=[fmt])
                from Qt.QtWidgets import QFrame, QLabel, QHBoxLayout, QVBoxLayout, QComboBox
                from Qt.QtWidgets import QCheckBox
                from Qt.QtCore import Qt
                options_frame = dlg.custom_area
                options_layout = QVBoxLayout(options_frame)
                options_frame.setLayout(options_layout)
                amount_frame = QFrame(options_frame)
                options_layout.addWidget(amount_frame, Qt.AlignCenter)
                amount_layout = QHBoxLayout(amount_frame)
                amount_layout.addWidget(QLabel("Save", amount_frame))
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
                cmds = [cmd for cmd in self.history()]
            else:
                # listbox.selectedItems() may not be in order, so...
                items = [self.listbox.item(i) for i in range(self.listbox.count())
                    if self.listbox.item(i).isSelected()]
                cmds = [item.text() for item in items]
            from chimerax.io import open_output
            f = open_output(path, encoding='utf-8', append=self.append_checkbox.isChecked())
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
            retain = []
            listbox_index = 0
            for h_item in self._history:
                if self.typed_only and not h_item[1]:
                    retain.append(h_item)
                    continue
                if not self.listbox.item(listbox_index).isSelected():
                    # not selected for deletion
                    retain.append(h_item)
                listbox_index += 1
            self._history.replace(retain)
            self.populate()
            return
        if label == "Copy":
            clipboard = session.ui.clipboard()
            clipboard.setText("\n".join([item.text() for item in self.listbox.selectedItems()]))
            return
        if label == "Help":
            from chimerax.core.commands import run
            run(session, 'help help:user/tools/cli.html#history')
            return

    def down(self, shifted):
        sels = self.listbox.selectedIndexes()
        if len(sels) != 1:
            self._search_cache = (False, None)
            return
        sel = sels[0].row()
        orig_text = self.controller.text.currentText()
        match_against = None
        if shifted:
            was_searching, prev_search = self._search_cache
            if was_searching:
                match_against = prev_search
            else:
                words = orig_text.strip().split()
                if words:
                    match_against = words[0]
                    self._search_cache = (True, match_against)
                else:
                    self._search_cache = (False, None)
        else:
            self._search_cache = (False, None)
        if match_against:
            last = self.listbox.count() - 1
            while sel < last:
                if self.listbox.item(sel + 1).text().startswith(match_against):
                    break
                sel += 1
        if sel == self.listbox.count() - 1:
            return
        self.listbox.clearSelection()
        self.listbox.setCurrentRow(sel + 1)
        new_text = self.listbox.item(sel + 1).text()
        self.controller.cmd_replace(new_text)
        if orig_text == new_text:
            self.down(shifted)

    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from Qt.QtGui import QAction
        filter_action = QAction("Typed commands only", menu)
        filter_action.setCheckable(True)
        filter_action.setChecked(self.controller.settings.typed_only)
        filter_action.toggled.connect(lambda arg, f=self.controller._set_typed_only: f(arg))
        menu.addAction(filter_action)

    def on_append_change(self, event):
        self.overwrite_disclaimer.Show(self.save_append_CheckBox.Value)

    def append_changed(self, append):
        if append:
            self.overwrite_disclaimer.show()
        else:
            self.overwrite_disclaimer.hide()

    def on_listbox(self, event):
        self.select()

    def populate(self):
        self.listbox.clear()
        history = self.history()
        self.listbox.addItems([cmd for cmd in history])
        self.listbox.setCurrentRow(len(history) - 1)
        self.update_list()
        self.select()
        self.controller.text.lineEdit().setFocus()
        self.controller.text.lineEdit().selectAll()
        cursels = self.listbox.scrollToBottom()

    def search_reset(self):
        searching, target = self._search_cache
        if searching:
            self._search_cache = (False, None)
            self.listbox.blockSignals(True)
            self.listbox.clearSelection()
            self.listbox.setCurrentRow(self.listbox.count() - 1)
            self.listbox.blockSignals(False)

    def select(self):
        sels = self.listbox.selectedItems()
        if len(sels) != 1:
            return
        self.controller.cmd_replace(sels[0].text())

    def up(self, shifted):
        sels = self.listbox.selectedIndexes()
        if len(sels) != 1:
            self._search_cache = (False, None)
            return
        sel = sels[0].row()
        orig_text = self.controller.text.currentText()
        match_against = None
        if shifted:
            was_searching, prev_search = self._search_cache
            if was_searching:
                match_against = prev_search
            else:
                words = orig_text.strip().split()
                if words:
                    match_against = words[0]
                    self._search_cache = (True, match_against)
                else:
                    self._search_cache = (False, None)
        else:
            self._search_cache = (False, None)
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

    def update_list(self):
        c = self.controller
        last8 = list(reversed(self.history()[-8:]))
        # without blocking signals, if the command list is empty then
        # "Command History" (the first entry) will execute...
        c.text.blockSignals(True)
        c.text.clear()
        c.text.addItems(last8 + [c.show_history_label, c.compact_label])
        if not last8:
            c.text.lineEdit().setText("")
        c.text.blockSignals(False)

    def history(self):
        if self.typed_only:
            return [h[0] for h in self._history if h[1]]
        return [h[0] for h in self._history]

    def set_typed_only(self, typed_only):
        self.typed_only = typed_only
        self.populate()

    def _num_remembered_changed(self, new_hist_len):
        if len(self._history) > new_hist_len:
            self._history.replace(self._history[-new_hist_len:])
            self.populate()
        self.controller.settings.num_remembered = new_hist_len

