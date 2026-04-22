# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance

from chimerax.core.tools import ToolInstance
class MutationScoresList(ToolInstance):
    help = 'https://www.rbvi.ucsf.edu/chimerax/data/mutation-scores-oct2024/mutation_scores.html'

    def __init__(self, session, tool_name = 'Mutation Scores'):
        self._selection_order = {}  # Maps (mset, score_name) to counter for ordering selections
        self._selection_count = 0

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))
        
        from Qt.QtWidgets import QListWidget
        class ScoresList(QListWidget):
            def sizeHint(self):
                from Qt.QtCore import QSize
                return QSize(500,50)
        self._mutation_set_list = lw = ScoresList(parent)
        lw.setMinimumHeight(30)
        lw.setSelectionMode(lw.ExtendedSelection)
        lw.itemClicked.connect(self._list_item_clicked)
        layout.addWidget(lw)
        self._update_list()

        from chimerax.ui.widgets import EntriesRow
        br = EntriesRow(parent,
                        ('Heatmap', self._show_heatmap),
                        ('Scatterplot', self._show_scatterplot),
                        ('Histogram', self._show_histogram),
                        ('Color structure', self._show_color_structure),
                         ('Close data', self._close_data),
                         ('Help', self._show_help),
                        spacing = 5)
        layout.addWidget(br.frame)
                
        tw.manage(placement="side")
    
    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Mutation Scores', create=create)

    def _update_list(self):
        mset_list = self._mutation_set_list
        from .ms_data import mutation_all_scores
        msets = mutation_all_scores(self.session)
        mset_list.clear()
        row = 0
        from Qt.QtCore import Qt
        for mset in msets:
            mset_list.addItem(mset.name)
            item = mset_list.item(row)
            row += 1
            item.setData(Qt.UserRole, (mset, None))
            for score_name in mset.score_names():
                mset_list.addItem('\t' + score_name)
                item = mset_list.item(row)
                row += 1
                item.setData(Qt.UserRole, (mset, score_name))

    def _list_item_clicked(self, item):
        # Keep track of the order items were selected in so that
        # order can be used in heatmaps and scatter plots.
        from Qt.QtCore import Qt
        mset, score_name = item.data(Qt.UserRole)
        self._selection_order[(mset, score_name)] = self._selection_count
        self._selection_count += 1
        
    def _selected_mutation_scores(self, default_all = True):
        scores = {}	# Maps mutation set name to list of score names
        update_list = False
        open_msets = self._open_mutation_sets
        sel_items = list(self._mutation_set_list.selectedItems())
        from Qt.QtCore import Qt
        sel_scores = [item.data(Qt.UserRole) for item in sel_items]
        sel_scores.sort(key = lambda s: self._selection_order.get(s,0))
        for mset, score_name in sel_scores:
            if mset not in open_msets:
                update_list = True
            else:
                if mset not in scores:
                    scores[mset] = []
                if score_name is not None:
                    scores[mset].append(score_name)
        if update_list:
            self._update_list()
        if len(scores) == 0 and default_all:
            scores = {mset:[] for mset in open_msets}
        return scores

    def _selected_score_names(self):
        scores = self._selected_mutation_scores()
        mset, score_names = tuple(scores.items())[0] if scores else (None, [])
        return mset, score_names

    def _selected_mutation_sets(self):
        return tuple(self._selected_mutation_scores().keys())

    @property
    def _open_mutation_sets(self):
        from .ms_data import mutation_all_scores
        open_msets = mutation_all_scores(self.session)
        return open_msets

    def _show_heatmap(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        mset_option = f'mutationSet {mset.name}' if len(self._open_mutation_sets) > 1 else ''
        scores_option = f'scores {",".join(score_names)}' if score_names else ''
        self._run_command(f'mutationscores heatmap {scores_option} {mset_option}')

    def _show_scatterplot(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        if len(mset.score_names()) < 2:
            from chimerax.core.users import UserError
            raise UserError(f'Mutation set {mset.name} does not have 2 scores for a scatter plot')
        x_score_name, y_score_name = score_names[:2] if len(score_names) >= 2 else mset.score_names()[:2]
        mset_option = f'mutationSet {mset.name}' if len(self._open_mutation_sets) > 1 else ''
        self._run_command(f'mutationscores scatterplot {x_score_name} {y_score_name} {mset_option}')
    
    def _show_histogram(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        score_name = score_names[0] if score_names else mset.score_names()[0]
        mset_option = f'mutationSet {mset.name}' if len(self._open_mutation_sets) > 1 else ''
        self._run_command(f'mutationscores histogram {score_name} {mset_option}')
    
    def _show_color_structure(self):
        from .ms_color_history import show_structure_coloring_gui
        coloring_gui = show_structure_coloring_gui(self.session)
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        coloring_gui.set_mutation_set(mset)
        if score_names:
            coloring_gui.set_coloring_score(score_names[0])
    
    def _close_data(self):
        msets = self._selected_mutation_sets()
        from .ms_data import mutation_scores_close
        for mset in msets:
            mutation_scores_close(self.session, mset.name)
        self._update_list()

    def _show_help(self):
        self._run_command('help %s' % self.help)

    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)
        
    # ---------------------------------------------------------------------------
    # Session save and restore.
    # Even though this tool has no state we want it displayed if it was shown
    # when the session was saved because it is used to access other mutation tools
    # that don't exist in the menus.
    #
    @property
    def SESSION_SAVE(self):
        return self.tool_window.shown
    def take_snapshot(self, session, flags):
        data = {'version': '1'}
        return data
    @classmethod
    def restore_snapshot(cls, session, data):
        msl = show_mutation_scores_list(session)
        return msl

def show_mutation_scores_list(session):
    msl = MutationScoresList.get_singleton(session, create=True)
    msl._update_list()
    msl.display(True)
    return msl
