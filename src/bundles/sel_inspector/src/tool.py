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

    #help = "help:user/tools/distances.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QMenu
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        self.text_description = QLabel()
        layout.addWidget(self.text_description)

        self.chooser = QPushButton()
        self.item_menu = QMenu()
        self.item_menu.triggered.connect(self._menu_cb)
        self.chooser.setMenu(self.item_menu)
        layout.addWidget(self.chooser)

        self.options_layout = QVBoxLayout()
        layout.addLayout(self.options_layout)
        self.options_container = None

        from chimerax.dist_monitor.cmd import group_triggers
        self.handlers = [
            session.items_inspection.triggers.add_handler("inspection items changed", self._new_items),
            session.triggers.add_handler("selection changed", self._sel_changed)
        ]
        self._new_items()
        self._sel_changed()
        tw.manage(placement=None)

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def _menu_cb(self, action):
        if self.options_container:
            self.options_layout.removeWidget(self.options_container)
        from chimerax.ui.options import OptionsPanel
        container = self.options_container = OptionsPanel()
        self.options_layout.addWidget(container)
        for option in self.session.items_inspection.item_options(self.menu_mapping[action]):
            container.add_option(option(None, None, self._option_cb))

    def _new_items(self, *args, **kw):
        cur_text = None if self.item_menu.isEmpty() else self.item_menu.menuAction().text()
        self.menu_mapping = {}
        self.item_menu.clear()
        cur_action = None
        self.item_types = self.session.items_inspection.item_types
        self.item_types.sort(key=lambda x: x.lower())
        for item_type in self.item_types:
            menu_text = item_type.capitalize() if item_type.islower() else item_type
            action = self.item_menu.addAction(menu_text)
            self.menu_mapping[action] = item_type
            if cur_text == menu_text:
                cur_action = action
        if cur_action:
            self.item_menu.setActiveAction(cur_action)
        elif not self.item_menu.isEmpty():
            first_action = self.item_menu.actions()[0]
            self.item_menu.setActiveAction(first_action)
            self.chooser.setText(first_action.text())
            self._menu_cb(first_action)

    def _option_cb(self, opt):
        from chimerax.core.commands import run
        run(self.session, opt.command_format % "sel")

    def _sel_changed(self, *args, **kw):
        cur_item_type = self.menu_mapping[self.item_menu.activeAction()]
        sel_strings = []
        for item_type in self.item_types:
            sel_items = self.session.selection.items(item_type)
            if item_type == cur_item_type:
                cur_items = sel_items
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
        if not cur_items:
            return
        from chimerax.atomic import Collection
        if [isinstance(x, Collection) for x in cur_items].count(True) == len(cur_items):
            from  chimerax.atomic import concatenate
            items = concatenate(cur_items)
        else:
            items = []
            for ci in cur_items:
                items.extend(ci)
        for option in self.options_container.options():
            option.display_for_items(items)

