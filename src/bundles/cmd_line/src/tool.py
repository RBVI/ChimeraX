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
        from chimerax.core.ui.gui import MainToolWindow

        self._in_init = True
        self.tool_window = MainToolWindow(self, close_destroys=False)
        parent = self.tool_window.ui_area
        self.history_dialog = _HistoryDialog(self)
        from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QLabel
        label = QLabel(parent)
        label.setText("Command:")
        class CmdText(QComboBox):
            def __init__(self, parent, tool):
                self.tool = tool
                QComboBox.__init__(self, parent)
                self._processing_key = False
                from PyQt5.QtCore import Qt
                # defer context menu to parent
                self.setContextMenuPolicy(Qt.NoContextMenu)

            def keyPressEvent(self, event, forwarded=False):
                self._processing_key = True
                if forwarded:
                    # Give command line the focus, so that up/down arrow work as
                    # exepcted rather than changing the selection level
                    self.setFocus()
                from PyQt5.QtCore import Qt
                import sys
                control_key = Qt.MetaModifier if sys.platform == "darwin" else Qt.ControlModifier
                shifted = event.modifiers() & Qt.ShiftModifier
                if event.key() == Qt.Key_Up:  # up arrow
                    self.tool.history_dialog.up(shifted)
                elif event.key() == Qt.Key_Down:  # down arrow
                    self.tool.history_dialog.down(shifted)
                elif event.modifiers() & control_key:
                    if event.key() == Qt.Key_N:
                        self.tool.history_dialog.down(shifted)
                    elif event.key() == Qt.Key_P:
                        self.tool.history_dialog.up(shifted)
                    elif event.key() == Qt.Key_U:
                        self.tool.cmd_clear()
                    elif event.key() == Qt.Key_K:
                        self.tool.cmd_clear_to_end_of_line()
                    else:
                        QComboBox.keyPressEvent(self, event)
                else:
                    QComboBox.keyPressEvent(self, event)
                self._processing_key = False

            def retain_selection_on_focus_out(self):
                # prevent de-selection of text when focus lost
                if self._processing_key:
                    return
                le = self.lineEdit()
                if not le.hasFocus() and not le.selectedText() and le.text():
                    le.selectAll()

        self.text = CmdText(parent, self)
        self.text.setEditable(True)
        self.text.setCompleter(None)
        layout = QHBoxLayout(parent)
        layout.setSpacing(1)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(self.text, 1)
        parent.setLayout(layout)
        self.text.lineEdit().returnPressed.connect(self.execute)
        self.text.lineEdit().editingFinished.connect(self.text.lineEdit().selectAll)
        # lineEdit() seems to be None during entire CmdText constructor, so connect here...
        self.text.lineEdit().selectionChanged.connect(self.text.retain_selection_on_focus_out)
        self.text.currentTextChanged.connect(self.text_changed)
        self.text.forwarded_keystroke = lambda e: self.text.keyPressEvent(e, forwarded=True)
        session.ui.register_for_keystrokes(self.text)
        self.history_dialog.populate()
        self.tool_window.manage(placement="bottom")
        self._in_init = False

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
        super().delete()

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
            if not self._in_init:
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
        from contextlib import contextmanager
        @contextmanager
        def processing_command(line_edit, cmd_text):
            line_edit.blockSignals(True)
            try:
                yield
            finally:
                line_edit.blockSignals(False)
                line_edit.setText(cmd_text)
                line_edit.selectAll()
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
            with processing_command(self.text.lineEdit(), cmd_text):
                try:
                    cmd = Command(session)
                    cmd.run(cmd_text)
                except SystemExit:
                    # TODO: somehow quit application
                    raise
                except errors.UserError as err:
                    logger.status(str(err), color="crimson")
                except:
                    raise
                finally:
                    # done before command execution, will show
                    # oldest known command while command executing
                    self.history_dialog.add(cmd_text)
        self.set_focus()

    def set_focus(self):
        from PyQt5.QtCore import Qt
        self.text.lineEdit().setFocus(Qt.OtherFocusReason)

    @classmethod
    def get_singleton(cls, session, **kw):
        from chimerax.core import tools
        return tools.get_singleton(session, CommandLine, 'Command Line Interface', **kw)

class _HistoryDialog:

    record_label = "Record..."
    execute_label = "Execute"

    NUM_REMEMBERED = 500

    def __init__(self, controller):
        # make dialog hidden initially
        self.controller = controller
        from chimerax.core.ui.gui import ChildToolWindow

        self.window = controller.tool_window.create_child_window(
            "Command History", close_destroys=False)

        parent = self.window.ui_area
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
        self.window.manage(placement=None)
        self.window.shown = False
        from chimerax.core.history import FIFOHistory
        self.history = FIFOHistory(self.NUM_REMEMBERED, controller.session, "command line")
        self._record_dialog = None
        self._search_cache = None
        self._suspend_handler = False

    def add(self, item):
        self.listbox.addItem(item)
        while self.listbox.count() > self.NUM_REMEMBERED:
            self.listbox.takeItem(0)
        self.history.enqueue(item)
        self.listbox.clearSelection()
        self.listbox.setCurrentRow(len(self.history) - 1)
        self.update_list()

    def button_clicked(self, label):
        if label == self.record_label:
            from chimerax.core.ui.open_save import export_file_filter, SaveDialog
            from chimerax.core.io import open_filename, format_from_name
            if self._record_dialog is None:
                fmt = format_from_name("ChimeraX commands")
                ext = fmt.extensions[0]
                self._record_dialog = dlg = SaveDialog(self.window.ui_area,
                    "Record Commands", name_filter=export_file_filter(format_name="ChimeraX commands"),
                                                       add_extension=ext)
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

    def down(self, shifted):
        sels = self.listbox.selectedIndexes()
        if len(sels) != 1:
            return
        sel = sels[0].row()
        orig_text = self.controller.text.currentText()
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
        self.listbox.setCurrentRow(sel + 1)
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

    def populate(self):
        self.listbox.clear()
        self.listbox.addItems([cmd for cmd in self.history])
        self.listbox.setCurrentRow(len(self.history) - 1)
        self.update_list()
        self.select()
        self.controller.text.lineEdit().setFocus()
        self.controller.text.lineEdit().selectAll()
        cursels = self.listbox.scrollToBottom()

    def select(self):
        sels = self.listbox.selectedItems()
        if len(sels) != 1:
            return
        self.controller.cmd_replace(sels[0].text())

    def up(self, shifted):
        sels = self.listbox.selectedIndexes()
        if len(sels) != 1:
            return
        sel = sels[0].row()
        orig_text = self.controller.text.currentText()
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

    def update_list(self):
        c = self.controller
        last8 = self.history[-8:]
        last8.reverse()
        c.text.clear()
        c.text.addItems(last8 + [c.show_history_label, c.compact_label])

    def _entry_modified(self, event):
        if not self._suspend_handler:
            self._search_cache = None
        event.Skip()
