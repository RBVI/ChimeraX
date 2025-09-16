# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.core.errors import UserError
from chimerax.core.commands import run, concise_model_spec
from chimerax.core.settings import Settings

class ModelPanel(ToolInstance):

    SESSION_ENDURING = True
    # if SESSION_ENDURING is True, tool instance not deleted at session closure
    help = "help:user/tools/modelpanel.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "Models"
        self.settings = ModelPanelSettings(session, "ModelPanel")
        last = self.settings.last_use
        from time import time
        now = self.settings.last_use = time()
        short_titles = last != None and now - last < 777700 # about 3 months
        if not short_titles:
            session.logger.status("You can double click a model's Name or ID in the model panel"
                " to edit those fields", log=True, color="forest green")

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        tw.fill_context_menu = self.fill_context_menu
        parent = tw.ui_area
        from Qt.QtWidgets import QTreeWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, \
            QFrame, QPushButton, QSizePolicy, QScrollArea, QWidget
        class SizedTreeWidget(QTreeWidget):
            def sizeHint(self):
                from Qt.QtCore import QSize
                # side buttons will keep the vertical size reasonable
                if getattr(self, '_first_size_hint_call', True):
                    self._first_size_hint_call = False
                    width = 0
                else:
                    width = self.header().length()
                return QSize(width, 200)
        self.tree = SizedTreeWidget()
        self.tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tree.expanded.connect(self._ensure_id_width)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.tree)
        layout.setStretchFactor(self.tree, 1)
        parent.setLayout(layout)
        shown_title = "" if short_titles else "Shown"
        sel_title = "" if short_titles else "Select"
        self.tree.setHeaderLabels(["Name", "ID", " ", shown_title, sel_title, "Skip"])
        from chimerax.ui.icons import get_qt_icon
        self.tree.headerItem().setIcon(3, get_qt_icon("shown"))
        self.tree.headerItem().setToolTip(3, "Shown")
        self.tree.headerItem().setIcon(4, get_qt_icon("select"))
        self.tree.headerItem().setToolTip(4, "Selected")
        self.tree.setColumnWidth(self.NAME_COLUMN, 200)
        self.tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setAnimated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.itemChanged.connect(self._tree_change_cb)
        self.tree.itemDoubleClicked.connect(self._item_double_clicked)
        if not self.showing_sequence_controls:
            self.tree.hideColumn(self.SKIP_COLUMN)
        scrolled_button_area = QScrollArea()
        layout.addWidget(scrolled_button_area)
        button_area = QWidget()
        self.buttons_layout = buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(0,0,0,0)
        buttons_layout.setSpacing(0)
        button_area.setLayout(buttons_layout)
        self._items = []
        for model_func in [close, hide, show, view, info]:
            button = QPushButton(model_func.__name__.capitalize())
            buttons_layout.addWidget(button)
            button.clicked.connect(lambda *, self=self, mf=model_func, ses=session:
                mf([self.models[row] for row in [self._items.index(i)
                    for i in self.tree.selectedItems()]] or self.models, ses))
        self._seq_buttons = []
        for model_func in [self._next_model, self._previous_model]:
            button = QPushButton(model_func.__name__[1:-6].capitalize())
            self._seq_buttons.append(button)
            buttons_layout.addWidget(button)
            if not self.showing_sequence_controls:
                button.setHidden(True)
            button.clicked.connect(model_func)
        scrolled_button_area.setWidget(button_area)
        self.simply_changed_models = set()
        self.check_model_list = True
        self.countdown = 1
        self.self_initiated = False
        import weakref
        self.skip_models = weakref.WeakKeyDictionary()
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, \
            MODEL_COLOR_CHANGED, MODEL_DISPLAY_CHANGED, MODEL_ID_CHANGED, MODEL_NAME_CHANGED
        from chimerax.core.selection import SELECTION_CHANGED
        session.triggers.add_handler(SELECTION_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, countdown=3))
        session.triggers.add_handler(MODEL_COLOR_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, simple_change=True, countdown=(0,3)))
        session.triggers.add_handler(MODEL_DISPLAY_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, simple_change=True, countdown=(0,3)))
        session.triggers.add_handler(ADD_MODELS,
            lambda *args: self._initiate_fill_tree(*args, always_rebuild=True, countdown=(3,10)))
        session.triggers.add_handler(REMOVE_MODELS,
            lambda *args: self._initiate_fill_tree(*args, always_rebuild=True))
        session.triggers.add_handler(MODEL_ID_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, always_rebuild=True, countdown=3))
        session.triggers.add_handler(MODEL_NAME_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, simple_change=True, countdown=3))
        from chimerax import atomic
        atomic.get_triggers().add_handler("changes", self._changes_cb)
        self._frame_drawn_handler = None
        tw.manage(placement="side")
        tw.shown_changed = self._shown_changed

    NAME_COLUMN = 0
    ID_COLUMN = 1
    COLOR_COLUMN = 2
    SHOWN_COLUMN = 3
    SELECT_COLUMN = 4
    SKIP_COLUMN = 5

    def fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction

        action = QAction("Group Models...", menu)
        action.triggered.connect(lambda *args, s=self: show_group_models_tool(s.session))
        menu.addAction(action)

        action = QAction("Show Sequential Display Controls", menu)
        action.setCheckable(True)
        action.setChecked(self.showing_sequence_controls)
        action.triggered.connect(lambda *args, s=self:
            setattr(s, 'showing_sequence_controls', not s.showing_sequence_controls))
        menu.addAction(action)

    @classmethod
    def get_singleton(self, session):
        from chimerax.core import tools
        return tools.get_singleton(session, ModelPanel, 'Model Panel', create=False)

    @property
    def showing_sequence_controls(self):
        return self.settings.show_sequential_controls

    @showing_sequence_controls.setter
    def showing_sequence_controls(self, show):
        if show == self.settings.show_sequential_controls:
            return
        self.settings.show_sequential_controls = show
        from Qt.QtCore import Qt
        if show:
            self.tree.showColumn(self.SKIP_COLUMN)
            self.tree.keyPressEvent = self._seq_key_press
        else:
            self.tree.hideColumn(self.SKIP_COLUMN)
            self.tree.keyPressEvent = self.session.ui.forward_keystroke
        for button in self._seq_buttons:
            button.setHidden(not show)

    def _changes_cb(self, trigger_name, changes):
        reasons = changes.atom_reasons()
        if "color changed" in reasons or 'display changed' in reasons:
            self._initiate_fill_tree(countdown=3)

    def _ensure_id_width(self, *args):
        # ensure that the newly visible model id isn't just "..."
        self.tree.resizeColumnToContents(self.ID_COLUMN)

    def _initiate_fill_tree(self, *args, always_rebuild=False,
            simple_change=False, countdown=1):
        # in order to allow models to be drawn as quickly as possible,
        # delay the update of the tree until the 'new frame' trigger fires
        # a number of times ('countdown' -- if a tuple then it varies based on
        # the number of models, from (lowest, highest)).  However, if no models
        # are open, we update immediately because Rapid Access will come up and
        # 'new frame' may not fire for awhile.
        # We used to work off the 'frame drawn' trigger with no delay, but
        # that is problematic if some other code is trying to do something
        # quickly every frame (e.g. volume series playback) that causes the
        # triggers we react to to fire.
        if self.self_initiated:
            self.countdown = 1
            self.self_initiated = False
        else:
            if isinstance(countdown, tuple):
                # base countdown on how expensive updating the table is
                least, most = countdown
                target = int(len(self.session.models) / 50)
                countdown = max(least, min(target, most))
            self.countdown = max(countdown, self.countdown)
        if self.simply_changed_models is not None:
            if simple_change and args:
                self.simply_changed_models.add(args[-1])
            else:
                self.simply_changed_models = None
        if len(self.session.models) == 0:
            if self._frame_drawn_handler is not None:
                self.session.triggers.remove_handler(self._frame_drawn_handler)
            self._fill_tree(always_rebuild=True)
        elif self._frame_drawn_handler is None:
            self._frame_drawn_handler = self.session.triggers.add_handler("new frame",
                lambda *args, ft=self._fill_tree, ar=always_rebuild: ft(always_rebuild=ar))
        elif always_rebuild:
            self.session.triggers.remove_handler(self._frame_drawn_handler)
            self._frame_drawn_handler = self.session.triggers.add_handler("new frame",
                lambda *args, ft=self._fill_tree: ft(always_rebuild=True))

    def _fill_tree(self, *, always_rebuild=False):
        if not self.displayed():
            # Don't update panel when it is hidden.
            self._frame_drawn_handler = None
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER

        self.countdown -= 1
        if self.countdown > 0:
            return

        # cell editing could have disabled key forwarding
        # (to block the Return key getting to the command line)
        self.tree.keyPressEvent = self._seq_key_press \
            if self.showing_sequence_controls else self.session.ui.forward_keystroke
        self.tree.blockSignals(True) # particularly itemChanged
        if self.check_model_list or always_rebuild:
            update = self._process_models() and not always_rebuild
            self.check_model_list = False
        else:
            update = not always_rebuild
        if not update:
            expanded_models = { i._model : i.isExpanded()
                                for i in self._items if hasattr(i, '_model')}
            scroll_position = self.tree.verticalScrollBar().sliderPosition()
            self.tree.clear()
            self._items = []
        all_selected_models = self.session.selection.models(all_selected=True)
        part_selected_models = self.session.selection.models()
        from Qt.QtWidgets import QTreeWidgetItem, QPushButton
        from Qt.QtCore import Qt
        from Qt.QtGui import QColor
        item_stack = [self.tree.invisibleRootItem()]
        for model in self.models:
            model_id, model_id_string, bg_color, display, name, selected, part_selected, skip = \
                self._get_info(model, all_selected_models, part_selected_models)
            if model_id is None:
                continue
            len_id = len(model_id)
            if update:
                if len_id == len(item_stack):
                    # first child
                    item = item_stack[-1].child(0)
                    item_stack.append(item)
                else:
                    # sibling
                    parent, previous_child = item_stack[len_id-1:len_id+1]
                    item = parent.child(parent.indexOfChild(previous_child)+1)
                    item_stack[len_id:] = [item]
            else:
                parent = item_stack[0] if len(item_stack) == 1 else item_stack[len_id-1]
                item = QTreeWidgetItem(parent)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item._model = model
                item_stack[len_id:] = [item]
                self._items.append(item)
                if bg_color is not False:
                    from chimerax.ui.widgets import MultiColorButton
                    but = MultiColorButton(has_alpha_channel=True, max_size=(16,16), pause_delay=0.5)
                    def set_model_color(rgba, m=model, ses=self.session, but=but):
                        from chimerax.core.models import Surface
                        from chimerax.atomic import Structure
                        if isinstance(m, (Structure, Surface)):
                            target_string = ""
                        else:
                            target_string = " models"
                        from chimerax.core.commands import run
                        from chimerax.core.colors import color_name
                        cmd = "color #%s %s%s" % (m.id_string,
                            color_name(rgba, always_include_hex_alpha=True), target_string)
                        run(ses, cmd, log=False)
                        but.delayed_cmd_text = cmd
                    but.color_changed.connect(set_model_color)
                    def log_delayed_cmd(*args, but=but, ses=self.session):
                        from chimerax.core.commands import Command
                        Command(ses).run(but.delayed_cmd_text, log_only=True)
                    but.color_pause.connect(log_delayed_cmd)
                    but.set_color(bg_color)
                    self.tree.setItemWidget(item, self.COLOR_COLUMN, but)
            if len(name) > 30:
                item.setToolTip(self.NAME_COLUMN, name)
                
                    
                
            if self.simply_changed_models and model not in self.simply_changed_models:
                continue
            item.setText(self.ID_COLUMN, model_id_string)
            bg = item.background(self.ID_COLUMN)
            if bg_color is False:
                bg.setStyle(Qt.NoBrush)
            else:
                but = self.tree.itemWidget(item, self.COLOR_COLUMN)
                if but is not None:
                    but.set_color(bg_color)
            item.setBackground(self.COLOR_COLUMN, bg)
            if display is not None:
                item.setCheckState(self.SHOWN_COLUMN,
                    Qt.CheckState.Checked if display else Qt.CheckState.Unchecked)
            if selected:
                item.setCheckState(self.SELECT_COLUMN, Qt.CheckState.Checked)
            elif part_selected:
                item.setCheckState(self.SELECT_COLUMN, Qt.CheckState.PartiallyChecked)
            else:
                item.setCheckState(self.SELECT_COLUMN, Qt.CheckState.Unchecked)

            if skip is not None:
                item.setCheckState(self.SKIP_COLUMN,
                    Qt.CheckState.Checked if skip else Qt.CheckState.Unchecked)
            item.setText(self.NAME_COLUMN, name)
            if not update:
                # Expand new top-level displayed models, or if previously expanded
                if hasattr(model, 'model_panel_show_expanded'):
                    expand_default = model.model_panel_show_expanded
                else:
                    expand_default = (model.display
                                      and len(model.id) <= 1
                                      and len(model.child_models()) <= 10)
                expand = expanded_models.get(model, expand_default)
                if expand:
                    self.tree.expandItem(item)
        if not update:
            self.tree.verticalScrollBar().setSliderPosition(scroll_position)
        for i in range(1,self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)
        self.tree.blockSignals(False)
        self.simply_changed_models = set()
        name_width = self.tree.sizeHintForColumn(self.NAME_COLUMN)
        self.tree.setColumnWidth(self.NAME_COLUMN, min(max(200, name_width), 400))

        self._frame_drawn_handler = None
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _get_info(self, obj, all_selected_models, part_selected_models):
        model_id = obj.id
        model_id_string = obj.id_string
        bg_color = self._model_color(obj)
        display = obj.display
        name = getattr(obj, "name", "(unnamed)")
        selected = obj in all_selected_models
        part_selected = selected or obj in part_selected_models
        # skip is only True/False for top-level models that aren't overlays
        # or grouped models whose group is top level, None otherwise.
        # Unfortunately obj.empty_drawing() is True for AtomicStructures, so can't use that test
        from chimerax.core.models import Model
        if obj in obj.session.main_view.overlays() or obj.__class__ == Model:
            skip = None
        elif obj.parent == obj.session.models.scene_root_model \
        or obj.parent.__class__ == Model and obj.parent.parent == obj.session.models.scene_root_model:
            skip = self.skip_models.setdefault(obj, False)
        else:
            skip = None
        return model_id, model_id_string, bg_color, display, name, selected, part_selected, skip

    def _header_click_cb(self, index):
        if index == 0:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()

    def _item_double_clicked(self, item, column):
        if column == self.NAME_COLUMN or column == self.ID_COLUMN:
            # prevent the Return key from reaching the command line
            self.tree.keyPressEvent = lambda event: event.setAccepted(True)
            self.tree.editItem(item, column)
            # too lazy to do the delegation rewrite to catch the editing
            # finishing when the editing ends with no change

    def _label_click(self, event):
        if event.Col == self.ID_COLUMN:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()
        event.Skip()

    def _next_model(self):
        self._show_next_model(1)

    def _model_color(self, model):
        return model.overall_color

    def _previous_model(self):
        self._show_next_model(-1)

    def _process_models(self):
        models = self.session.models.list()
        sorted_models = sorted(models, key=lambda m: m.id)
        final_models = list(sorted_models)
        update = True if hasattr(self, 'models') and final_models == self.models else False
        self.models = final_models
        return update

    def _seq_key_press(self, event):
        from Qt.QtCore import Qt
        if event.key() == Qt.Key_Up:
            self._previous_model()
        elif event.key() == Qt.Key_Down:
            self._next_model()
        else:
            self.session.ui.forward_keystroke(event)

    def _show_next_model(self, direction):
        if self._frame_drawn_handler is not None:
            # self.models is not up to date, typically happens when arrow key held down
            return
        cur_shown = [m for m in self.models
            if m in self.skip_models and not self.skip_models[m] and m.display]
        if len(cur_shown) > 1:
            hide(cur_shown[1:], self.session)
            self.session.logger.status("Showing %s" % cur_shown[0])
            return
        if not cur_shown:
            for m in self.models:
                if m in self.skip_models and not self.skip_models[m]:
                    show([m], self.session)
                    self.session.logger.status("Showing %s" % m)
                    break
            else:
                self.session.logger.warning("No models in display sequence")
            return
        m = cur_shown[0]
        index = self.models.index(m)
        while True:
            index = (index + direction) % len(self.models)
            next_m = self.models[index]
            if next_m == m:
                break
            if next_m not in self.skip_models:
                continue
            if self.skip_models[next_m]:
                continue
            show([next_m], self.session)
            hide([m], self.session)
            self.session.logger.status("Showing %s" % next_m)
            break

    def _shown_changed(self, shown):
        if shown:
            # Update panel when it is shown.
            self.check_model_list = True
            self._initiate_fill_tree()

    def _tree_change_cb(self, item, column):
        from Qt.QtCore import Qt
        model = self.models[self._items.index(item)]
        if column == self.SHOWN_COLUMN:
            self.self_initiated = True
            command_name = "show" if item.checkState(self.SHOWN_COLUMN) == Qt.CheckState.Checked else "hide"
            run(self.session, "%s #%s%s models" % (command_name,
                "!" if len(model.all_models()) > 1 else "", model.id_string))
        elif column == self.SELECT_COLUMN:
            self.self_initiated = True
            mode = "add" if item.checkState(self.SELECT_COLUMN) == Qt.CheckState.Checked else "subtract"
            run(self.session, "select " + mode +  " #" + model.id_string)
        elif column == self.ID_COLUMN:
            id_text = item.text(self.ID_COLUMN)
            try:
                ids = [int(x) for x in id_text.split('.')]
            except Exception:
                self._initiate_fill_tree()
                raise UserError("ID must be one or more integers separated by '.' characters")
            self.self_initiated = True
            run(self.session, "rename %s id #%s" % (item._model.atomspec, id_text))
        elif column == self.NAME_COLUMN:
            new_name = item.text(self.NAME_COLUMN)
            if not new_name or new_name.isspace():
                from chimerax.ui.ask import ask
                if ask(self.session, "Really use blank model name?", default="no") == "no":
                    self._initiate_fill_tree()
                    return
            self.self_initiated = True
            from chimerax.core.commands import StringArg
            run(self.session, "rename %s %s" % (item._model.atomspec, StringArg.unparse(new_name)))
        elif column == self.SKIP_COLUMN:
            self.skip_models[model] = item.checkState(self.SKIP_COLUMN) == Qt.CheckState.Checked


class ModelPanelSettings(Settings):
    AUTO_SAVE = {
        'last_use': None,
        'show_sequential_controls': False
    }

def close(models, session):
    # ask for confirmation if multiple top-level models being closed without explicitly selecting them
    if len([m for m in models if '.' not in m.id_string]) > 1 and not _mp.tree.selectedItems():
        from chimerax.ui.ask import ask
        if ask(session, "Really close all models?", title="Confirm close models") == "no":
            return
    _mp.self_initiated = True
    from chimerax.core.models import Model
    # The 'close' command explicitly avoids closing grouping models where not
    # all the child models are being closed so that "close ~#1.1" doesn't 
    # close the #1 grouping model.  Therefore we need to change the '#!'s
    # generated by concise_model_spec to just '#'s
    run(session, "close %s" %
        concise_model_spec(session, [m for m in models if isinstance(m, Model)]).replace('#!', '#'))

def hide(models, session):
    _mp.self_initiated = True
    run(session, "hide %s target m" % concise_model_spec(session, models))

def info(models, session):
    if not models:
        raise UserError("No selection made")
    for m in models:
        m.show_info()

_mp = None
def model_panel(session, tool_name):
    global _mp
    if _mp is None:
        _mp = ModelPanel(session, tool_name)
    return _mp

def show(models, session):
    _mp.self_initiated = True
    run(session, "show %s target m" % concise_model_spec(session, models))

def view(objs, session):
    from chimerax.core.models import Model
    models = [o for o in objs if isinstance(o, Model)]
    # 'View' should include submodels, even if not explicitly selected in the panel.
    # In particular a "volume model" is a grouping model with nothing directly in it, but
    # one or more surfaces as submodels.  So change any '#!'s to just '#'.
    run(session, "view %s clip false" % concise_model_spec(session, models).replace('#!', '#'))

_gmt = None
def show_group_models_tool(session):
    global _gmt
    if _gmt is None:
        _gmt = GroupModelsTool(session)
    _gmt.display(True)

class GroupModelsTool(ToolInstance):

    #help = "help:user/tools/modelpanel.html"

    def __init__(self, session):
        ToolInstance.__init__(self, session, "Group Models")

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QLineEdit
        from Qt.QtCore import Qt

        layout = QVBoxLayout()
        parent.setLayout(layout)

        from chimerax.ui.widgets import ModelListWidget
        self.model_list = ModelListWidget(session)
        layout.addWidget(self.model_list, stretch=1)

        layout.addWidget(QLabel("Choose models from above list to group"), alignment=Qt.AlignCenter)

        info_layout = QHBoxLayout()
        info_layout.addStretch(1)
        info_layout.addWidget(QLabel("New group ID: #"))
        self.id_entry = QLineEdit()
        info_layout.addWidget(self.id_entry)
        info_layout.addStretch(1)
        info_layout.addWidget(QLabel("New group name:"))
        self.name_entry = QLineEdit()
        self.name_entry.setText("group")
        info_layout.addWidget(self.name_entry)
        info_layout.addStretch(1)
        layout.addLayout(info_layout)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        # Don't also connect accepted to self.delete, since input error should reshow dialog
        bbox.accepted.connect(self.group_models)
        bbox.rejected.connect(self.delete)
        if getattr(self, 'help', None) is None:
            bbox.button(qbbox.Help).setEnabled(False)
        else:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)


        tw.manage(placement=None)

    def group_models(self):
        from chimerax.ui import tool_user_error
        id_components = []
        id_err_msg = "Group ID must be one or more positive integers separated by period ('.') characters"
        for comp in self.id_entry.text().strip().split('.'):
            try:
                comp = int(comp.strip())
            except ValueError:
                return tool_user_error(id_err_msg)
            if comp <= 0:
                return tool_user_error(id_err_msg)
            id_components.append(comp)
        target_id = tuple(id_components)
        target_id_string = '.'.join([str(comp) for comp in target_id])
        if len(id_components) > 1:
            parent = self.session.models.list(model_id=target_id[:-1])
            if not parent:
                return tool_user_error("Cannot create group with ID %s because the parent model (%s)"
                    " does not exist" % (target_id_string, '.'.join([str(comp) for comp in target_id[:-1]])))
            parent_model = parent[0]
        else:
            parent_model = None

        existing_models = self.session.models.list(model_id=target_id)
        if existing_models:
            return tool_user_error("There is already an existing model with the requested ID (%s)"
                % existing_models[0])

        grouped_members = self.model_list.value
        if not grouped_members:
            return tool_user_error("Must select at least one model to group")
        for member in grouped_members:
            if member.id == target_id[:len(member.id)]:
                return tool_user_error("Cannot make a model (%s) a child of itself (%s)"
                    % (member, target_id_string))

        group_name = self.name_entry.text().strip()
        if not group_name:
            return tool_user_error("Group name entry field is blank")

        from chimerax.core.commands import concise_model_spec, run, StringArg
        spec = concise_model_spec(self.session, grouped_members, allow_empty_spec=False)
        run(self.session, f"rename {spec} id {target_id_string}")
        run(self.session, f"rename #{target_id_string} {StringArg.unparse(group_name)}")

        self.display(False)
