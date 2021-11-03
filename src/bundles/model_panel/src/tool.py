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

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
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
        self.tree.keyPressEvent = session.ui.forward_keystroke
        self.tree.expanded.connect(self._ensure_id_width)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(self.tree)
        layout.setStretchFactor(self.tree, 1)
        parent.setLayout(layout)
        shown_title = "" if short_titles else "Shown"
        sel_title = "" if short_titles else "Select"
        self.tree.setHeaderLabels(["Name", "ID", " ", shown_title, sel_title])
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
        scrolled_button_area = QScrollArea()
        layout.addWidget(scrolled_button_area)
        button_area = QWidget()
        buttons_layout = QVBoxLayout()
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
        scrolled_button_area.setWidget(button_area)
        self.simply_changed_models = set()
        self.check_model_list = True
        self.countdown = 1
        self.self_initiated = False
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, \
            MODEL_DISPLAY_CHANGED, MODEL_ID_CHANGED, MODEL_NAME_CHANGED
        from chimerax.core.selection import SELECTION_CHANGED
        session.triggers.add_handler(SELECTION_CHANGED,
            lambda *args: self._initiate_fill_tree(*args, countdown=3))
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
    
    def _shown_changed(self, shown):
        if shown:
            # Update panel when it is shown.
            self.check_model_list = True
            self._initiate_fill_tree()

    @classmethod
    def get_singleton(self, session):
        from chimerax.core import tools
        return tools.get_singleton(session, ModelPanel, 'Model Panel', create=False)

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
        # a time of times ('countdown' -- if a tuple then it varies based on
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

        self.tree.blockSignals(True) # particularly itemChanged
        if self.check_model_list or always_rebuild:
            update = self._process_models() and not always_rebuild
            self.check_model_list = False
        else:
            update = not always_rebuild
        if not update:
            expanded_models = { i._model : i.isExpanded()
                                for i in self._items if hasattr(i, '_model')}
            self.tree.clear()
            self._items = []
        all_selected_models = self.session.selection.models(all_selected=True)
        part_selected_models = self.session.selection.models()
        from Qt.QtWidgets import QTreeWidgetItem, QPushButton
        from Qt.QtCore import Qt
        from Qt.QtGui import QColor
        item_stack = [self.tree.invisibleRootItem()]
        for model in self.models:
            model_id, model_id_string, bg_color, display, name, selected, part_selected = \
                self._get_info(model, all_selected_models, part_selected_models)
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
                        c_name = color_name(rgba)
                        need_transparency = (not c_name[0] == '#') or len(c_name) == 7
                        cmd = "color #%s %s%s%s" % (m.id_string, color_name(rgba), target_string,
                            " transparency 0" if need_transparency else "")
                        run(ses, cmd, log=False)
                        but.delayed_cmd_text = cmd
                    but.color_changed.connect(set_model_color)
                    def log_delayed_cmd(*args, but=but, ses=self.session):
                        from chimerax.core.commands import Command
                        Command(ses).run(but.delayed_cmd_text, log_only=True)
                    but.color_pause.connect(log_delayed_cmd)
                    but.set_color(bg_color)
                    self.tree.setItemWidget(item, self.COLOR_COLUMN, but)
                
                    
                
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
                item.setCheckState(self.SHOWN_COLUMN, Qt.CheckState.Checked if display else Qt.CheckState.Unchecked)
            if selected:
                item.setCheckState(self.SELECT_COLUMN, Qt.CheckState.Checked)
            elif part_selected:
                item.setCheckState(self.SELECT_COLUMN, Qt.CheckState.PartiallyChecked)
            else:
                item.setCheckState(self.SELECT_COLUMN, Qt.CheckState.Unchecked)
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
        for i in range(1,self.tree.columnCount()):
            self.tree.resizeColumnToContents(i)
        self.tree.blockSignals(False)
        self.simply_changed_models = set()

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
        return model_id, model_id_string, bg_color, display, name, selected, part_selected

    def _header_click_cb(self, index):
        if index == 0:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()

    def _label_click(self, event):
        if event.Col == self.ID_COLUMN:
            # ID label clicked.
            # Toggle sort order.
            self._sort_breadth_first = not self._sort_breadth_first
            self._fill_tree()
        event.Skip()

    def _model_color(self, model):
        return model.model_color

    def _process_models(self):
        models = self.session.models.list()
        sorted_models = sorted(models, key=lambda m: m.id)
        final_models = list(sorted_models)
        update = True if hasattr(self, 'models') and final_models == self.models else False
        self.models = final_models
        return update

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
            prefix = "" if item.checkState(self.SELECT_COLUMN) == Qt.CheckState.Checked else "~"
            run(self.session, prefix + "select #" + model.id_string)

from chimerax.core.settings import Settings
class ModelPanelSettings(Settings):
    AUTO_SAVE = {
        'last_use': None
    }

from chimerax.core.commands import run, concise_model_spec
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
    from chimerax.atomic import AtomicStructure
    structures = [m for m in models if isinstance(m, AtomicStructure)]
    if not structures:
        from chimerax.core.errors import UserError
        raise UserError("No atomic structure models chosen")
    spec = concise_model_spec(session, structures, allow_empty_spec=False, relevant_types=AtomicStructure)
    from chimerax.atomic.structure import assembly_html_table
    for s in structures:
        if assembly_html_table(s):
            base_cmd = "sym %s; " % spec
            break
    else:
        base_cmd = ""
    run(session, base_cmd + "log metadata %s; log chains %s" % (spec, spec))

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

