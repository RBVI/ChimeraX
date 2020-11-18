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

class FeatureBrowser:

    data_source = 'UniProt'

    def __init__(self, sv, seq, state, tool_window):
        self.sv = sv
        self.seq = seq
        self.tool_window = tool_window

        from PyQt5.QtWidgets import QHBoxLayout, QListWidget
        layout = QHBoxLayout()
        self.feature_map = seq.features(fetch=False).get(self.data_source, {})
        ftypes = list(self.feature_map.keys())
        ftypes.sort()
        self.category_chooser = category_chooser = QListWidget()
        category_chooser.setSelectionMode(QListWidget.SingleSelection)
        # need to initially distinguish from background, text, selection, and default region colors...
        from chimerax.core.colors import distinguish_from, Color
        used_colors = [(0,0,0), (1,1,1)]
        """
        if sv.settings.sel_region_interior:
            used_colors.append(Color(sv.settings.sel_region_interior).rgba[:3])
        if sv.settings.new_region_interior:
            used_colors.append(Color(sv.settings.new_region_interior).rgba[:3])
        """
        self.feature_region = {}
        for ftype in ftypes:
            category_chooser.addItem(ftype)
            features = self.feature_map[ftype]
            color = distinguish_from(used_colors, num_candidates=5)
            used_colors.append(color)
            for feature in features:
                self.feature_region[feature] = sv.new_region(ftype, outline=color, shown=False,
                    fill=[(x+1)/2 for x in color],
                    blocks=[(seq, seq, start-1, end-1) for start, end in feature.positions])
        self._selection = None
        category_chooser.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(category_chooser)
        tool_window.ui_area.setLayout(layout)

    def state(self):
        #TODO
        pass

    def _selection_changed(self):
        if self._selection:
            for feature in self.feature_map[self._selection]:
                self.feature_region[feature].shown = False
        sel_items = self.category_chooser.selectedItems()
        self._selection = sel_items[0].text() if sel_items else None
        if self._selection:
            for feature in self.feature_map[self._selection]:
                self.feature_region[feature].shown = True

