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

from PySide2.QtWidgets import QVBoxLayout, QListWidget, QLabel, QGridLayout, QListWidgetItem, QWidget
from PySide2.QtWidgets import QSizePolicy

class FeatureBrowser:

    data_source = 'UniProt'

    def __init__(self, sv, seq, state, tool_window):
        self.sv = sv
        self.seq = seq
        self.tool_window = tool_window

        from PySide2.QtCore import Qt
        layout = QGridLayout()
        layout.addWidget(QLabel("Feature Types"), 0, 0, alignment=Qt.AlignHCenter|Qt.AlignBottom)
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
        if state is None:
            self.feature_map = seq.features(fetch=False).get(self.data_source, {})
            ftypes = list(self.feature_map.keys())
            ftypes.sort()
            self.feature_region = {}
            for ftype in ftypes:
                category_chooser.addItem(ftype)
                features = self.feature_map[ftype]
                color = distinguish_from(used_colors, num_candidates=5)
                used_colors.append(color)
                for feature in features:
                    self.feature_region[feature] = sv.new_region(ftype, outline=color, shown=False,
                        read_only=True, assoc_with=seq, fill=[(x+1)/2 for x in color],
                        blocks=[(seq, seq, start-1, end-1) for start, end in feature.positions])
            self._selection = None
        else:
            self.feature_map = state['feature_map']
            ftypes = list(self.feature_map.keys())
            ftypes.sort()
            category_chooser.addItems(ftypes)
            self.feature_region = { feat: sv.region_browser.regions[reg_index]
                for feat, reg_index in state['feature_region'].items() }
            self._selection = state['selection']
            if self._selection:
                category_chooser.setCurrentRow(ftypes.index(self._selection))

        category_chooser.itemSelectionChanged.connect(self._category_selection_changed)
        layout.addWidget(category_chooser, 1, 0)
        tool_window.ui_area.setLayout(layout)

        layout.addWidget(QLabel("Features"), 0, 1, alignment=Qt.AlignHCenter|Qt.AlignBottom)
        self.feature_chooser = feature_chooser = FeatureList(feature_browser=self)
        if state is not None:
            feature_chooser.set_state(state['feature_chooser'])
        feature_chooser.itemSelectionChanged.connect(self._feature_selection_changed)
        self.residue_display = res_display = QLabel()
        res_display.setAlignment(Qt.AlignCenter)
        res_display.setWordWrap(True)
        layout.addWidget(feature_chooser, 1, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        layout.addWidget(res_display, 2, 1)

    def state(self):
        return {
            'shown': self.tool_window.shown,
            'feature_chooser': self.feature_chooser.state(),
            'feature_region': { feat: self.sv.region_browser.regions.index(reg)
                for feat, reg in self.feature_region.items() },
            'feature_map': self.feature_map,
            'selection': self._selection
        }

    def _category_selection_changed(self):
        if self._selection:
            for feature in self.feature_map[self._selection]:
                self.feature_region[feature].shown = False
        sel_items = self.category_chooser.selectedItems()
        self._selection = sel_items[0].text() if sel_items else None
        if self._selection:
            self.feature_chooser.set_features(self.feature_map[self._selection])
        else:
            self.feature_chooser.set_features([])

    def _feature_selection_changed(self):
        sel_category_items = self.category_chooser.selectedItems()
        if sel_category_items:
            sel_category = sel_category_items[0].text()
        else:
            return
        features = self.feature_map[sel_category]
        sel_rows = set([mi.row() for mi in self.feature_chooser.selectedIndexes()])
        shown_regions = []
        for i, feature in enumerate(features):
            shown = i in sel_rows
            region = self.feature_region[feature]
            region.shown = shown
            if shown:
                shown_regions.append(region)
        self._update_residue_display(shown_regions)

    def _update_residue_display(self, shown_regions):
        if shown_regions:
            residues=[]
            for region in shown_regions:
                residues.extend(self.sv.region_browser.region_residues(region))
            from chimerax.atomic import concise_residue_spec
            spec = concise_residue_spec(self.sv.session, residues)
            parts = []
            line_limit = 40
            while len(spec) > line_limit:
                try:
                    break_point = spec[:line_limit].rindex(',')
                except ValueError:
                    break
                parts.append(spec[:break_point+1])
                spec = spec[break_point+1:]
            parts.append(spec)
            text = '\n'.join(parts)
        else:
            text = ""
        self.residue_display.setText(text)

class FeatureList(QListWidget):
    def __init__(self, *args, feature_browser=None, **kw):
        super().__init__(*args, **kw)
        self.setStyleSheet('*::item:selected { background: rgb(210,210,210); }')
        self.setSelectionMode(self.ExtendedSelection)
        self.setWordWrap(True)
        self._region_map = feature_browser.feature_region
        self._fb = feature_browser
        self._features = []

    def set_features(self, features, *, selected_rows=None):
        if selected_rows is None:
            for feature in self._features:
                self._region_map[feature].shown = False
        self.clear()
        self._features = features
        shown_regions = []
        for row, feature in enumerate(features):
            region = self._region_map[feature]
            if selected_rows is None:
                region.shown = True
                shown_regions.append(region)
            item = QListWidgetItem()
            label = QLabel()
            label.setWordWrap(True)
            label.setOpenExternalLinks(True)
            if len(feature.details) > 1:
                text = "<dl><dt>" + feature.details[0] + "</dt><dd>" + "<br>".join(
                    feature.details[1:]) + "</dt></dl>"
            else:
                text = feature.details[0]
            label.setText(text)
            self.addItem(item)
            self.setItemWidget(item, label)
            item.setSizeHint(label.sizeHint())
            if selected_rows is not None and row in selected_rows:
                item.setSelected(True)
                shown_regions.append(region)
        self._fb._update_residue_display(shown_regions)

    def set_state(self, state):
        self.set_features(state['features'], selected_rows=set(state['selected rows']))

    def state(self):
        return {
            'features': self._features,
            'selected rows': [self.row(item) for item in self.selectedItems()]
        }
