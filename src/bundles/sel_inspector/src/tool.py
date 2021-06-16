# vim: set expandtab shiftwidth=4 softtabstop=4:

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

class SelInspector(ToolInstance):

    help = "help:user/tools/inspector.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QMenu, QHBoxLayout
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        layout.setContentsMargins(3,3,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        self.text_description = QLabel()
        layout.addWidget(self.text_description)

        container = QWidget()
        layout.addWidget(container, alignment=Qt.AlignLeft)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0,0,0,0)
        button_layout.setSpacing(0)
        container.setLayout(button_layout)
        button_layout.addWidget(QLabel("Inspect"), alignment=Qt.AlignRight)
        self.chooser = QPushButton()
        self.item_menu = QMenu()
        self.item_menu.triggered.connect(self._menu_cb)
        self.chooser.setMenu(self.item_menu)
        button_layout.addWidget(self.chooser, alignment=Qt.AlignLeft)

        self.options_layout = QVBoxLayout()
        layout.addLayout(self.options_layout)
        self.options_container = None
        class SizedLabel(QLabel):
            def sizeHint(self):
                from Qt.QtCore import QSize
                return QSize(300, 200)
        self.no_sel_label = SizedLabel()
        self.options_layout.addWidget(self.no_sel_label)
        self.no_sel_label.hide()

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        from chimerax.dist_monitor.cmd import group_triggers
        self.handlers = [
            session.items_inspection.triggers.add_handler("inspection items changed", self._new_items),
            session.triggers.add_handler("selection changed", self._sel_changed)
        ]
        self.current_item_handlers = []
        self._new_items()
        self._sel_changed()
        tw.manage(placement=None)

    def delete(self):
        for handler in self.handlers + self.current_item_handlers:
            handler.remove()
        super().delete()

    def _menu_cb(self, action=None):
        if action:
            cur_text = action.text()
            self.chooser.setText(cur_text)
        else:
            cur_text = self.chooser.text()
        if self.options_container:
            self.options_layout.removeWidget(self.options_container)
            self.options_container.hide()
            self.options_container.destroy()
        for handler in self.current_item_handlers:
            handler.remove()
        self.current_item_handlers = []
        from chimerax.ui.options import OptionsPanel
        class SizedOptionsPanel(OptionsPanel):
            def sizeHint(self):
                from Qt.QtCore import QSize
                return QSize(300, 200)
        container = self.options_container = SizedOptionsPanel()
        self.options_layout.addWidget(container)
        for option, trigger_info in self.session.items_inspection.item_info(self.button_mapping[cur_text]):
            opt = option(None, None, self._option_cb)
            container.add_option(opt)
            trigger_set, trigger_name, check_func = trigger_info
            def cb(trig_name, trig_data, *, refresh=self.refresh, check_func=check_func, opt=opt):
                if check_func(trig_data):
                    refresh(opt)
            self.current_item_handlers.append(trigger_set.add_handler(trigger_name, cb))
        self.refresh()

    def _new_items(self, *args, **kw):
        cur_text = self.chooser.text()
        self.button_mapping = {}
        self.item_menu.clear()
        self.item_types = self.session.items_inspection.item_types
        self.item_types.sort(key=lambda x: x.lower())
        for item_type in self.item_types:
            menu_text = item_type.capitalize() if item_type.islower() else item_type
            self.item_menu.addAction(menu_text)
            self.button_mapping[menu_text] = item_type
        if not cur_text and not self.item_menu.isEmpty():
            first_action = self.item_menu.actions()[0]
            self.chooser.setText(first_action.text())
            self._menu_cb()

    def _option_cb(self, opt):
        from chimerax.core.commands import run
        run(self.session, opt.command_format % "sel")

    def _sel_changed(self, *args, **kw):
        sel_strings = []
        for item_type in self.item_types:
            sel_items = self.session.selection.items(item_type)
            if not sel_items:
                continue
            num = sum([len(x) for x in sel_items])
            sel_strings.append("%d %s"
                % (num, item_type if num != 1 or item_type[-1] != 's' else item_type[:-1]))
        if sel_strings:
            description = "\n".join(sel_strings)
        elif self.session.selection.empty():
            description = "Nothing selected"
        else:
            description = "Nothing inspectable selected"
        self.text_description.setText(description)
        self.refresh()

    def refresh(self, opt=None):
        button_text = self.chooser.text()
        cur_item_type = self.button_mapping[button_text] if button_text else None
        sel_items = None
        for item_type in self.item_types:
            if item_type == cur_item_type:
                sel_items = self.session.selection.items(item_type)
                break
        if not sel_items:
            self.options_container.hide()
            self.no_sel_label.setText("(no %s selected)" % button_text.lower())
            self.no_sel_label.show()
            return
        self.no_sel_label.hide()
        self.options_container.show()
        from chimerax.atomic import Collection
        if [isinstance(x, Collection) for x in sel_items].count(True) == len(sel_items):
            from  chimerax.atomic import concatenate
            items = concatenate(sel_items)
        else:
            items = []
            for si in sel_items:
                items.extend(si)
        if opt is None:
            for option in self.options_container.options():
                option.display_for_items(items)
        else:
            opt.display_for_items(items)

