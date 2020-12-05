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

class FeatureBrowser:

    data_source = 'UniProt'

    def __init__(self, sv, seq, state, tool_window):
        self.sv = sv
        self.seq = seq
        self.tool_window = tool_window

        from PySide2.QtCore import Qt
        layout = QGridLayout()
        layout.addWidget(QLabel("Feature Types"), 0, 0, alignment=Qt.AlignHCenter|Qt.AlignBottom)
        self.feature_map = seq.features(fetch=False).get(self.data_source, {})
        ftypes = list(self.feature_map.keys())
        ftypes.sort()
        self.category_chooser = category_chooser = QListWidget()
        print(self.category_chooser.sizePolicy())
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
                    read_only=True, assoc_with=seq, fill=[(x+1)/2 for x in color],
                    blocks=[(seq, seq, start-1, end-1) for start, end in feature.positions])
        self._selection = None
        category_chooser.itemSelectionChanged.connect(self._category_selection_changed)
        layout.addWidget(category_chooser, 1, 0)
        tool_window.ui_area.setLayout(layout)

        layout.addWidget(QLabel("Features"), 0, 1, alignment=Qt.AlignHCenter|Qt.AlignBottom)
        self.feature_chooser = feature_chooser = FeatureList(region_map=self.feature_region)
        feature_chooser.itemSelectionChanged.connect(self._feature_selection_changed)
        layout.addWidget(feature_chooser, 1, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)

    def state(self):
        #TODO
        pass

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
        #TODO
        pass

class FeatureList(QListWidget):
    def __init__(self, *args, region_map={}, **kw):
        super().__init__(*args, **kw)
        self.setStyleSheet('*::item:selected { background: rgb(210,210,210); }')
        self._region_map = region_map
        self._features = []

    def set_features(self, features):
        for feature in self._features:
            self._region_map[feature].shown = False
        self.clear()
        self._features = features
        for feature in features:
            self._region_map[feature].shown = True
            item = QListWidgetItem()
            widget = QWidget()
            layout = QVBoxLayout()
            # multiline QLabel seemingly doesn't handle URLs after the first line, so do this
            for i, detail in enumerate(feature.details):
                text = QLabel(detail)
                if i > 0:
                    text.setIndent(30)
                text.setOpenExternalLinks(True)
                layout.addWidget(text)
            layout.setSizeConstraint(layout.SetFixedSize)
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(0)
            widget.setLayout(layout)
            self.addItem(item)
            item.setSizeHint(widget.sizeHint())
            self.setItemWidget(item, widget)
