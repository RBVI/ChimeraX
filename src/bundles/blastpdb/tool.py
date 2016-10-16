# vim: set expandtab shiftwidth=4 softtabstop=4:

# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance

_EmptyPage = "<html><body><h2>This space intentionally left blank</body></html>"
_NonEmptyPage = "<html><body><h2>This space unintentionally left blank</body></html>"


class ToolUI(ToolInstance):

    SESSION_ENDURING = False

    def __init__(self, session, tool_name, blast_results=None):
        # Standard template stuff
        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "Blast PDB"
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        parent = self.tool_window.ui_area

        # UI consists of a chain selector and search button on top
        # and HTML widget below for displaying results.
        # First we import/define all the widget classes we need:
        from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
        from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton
        from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage

        # Layout all the widgets
        layout = QVBoxLayout()
        search_layout = QHBoxLayout()
        label = QLabel("Chain:")
        search_layout.addWidget(label)
        self.chain_combobox = QComboBox()
        search_layout.addWidget(self.chain_combobox)
        button = QPushButton("BLAST")
        button.clicked.connect(self._blast_cb)
        search_layout.addWidget(button)
        search_layout.setStretchFactor(label, 0)
        search_layout.setStretchFactor(self.chain_combobox, 10)
        search_layout.setStretchFactor(button, 0)
        layout.addLayout(search_layout)
        self.results_view = QWebEngineView(parent)
        layout.addWidget(self.results_view)

        # Register for model addition/removal so we can update chain list
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        t = session.triggers
        self._add_handler = t.add_handler(ADD_MODELS, self._update_chains)
        self._remove_handler = t.add_handler(REMOVE_MODELS, self._update_chains)

        # Set widget values and go
        self._update_chains()
        if blast_results:
            self._update_blast_results(blast_results)
        parent.setLayout(layout)

    def _blast_cb(self, _):
        # TODO: initiate blast
        pass

    def _update_chains(self, trigger=None, trigger_data=None):
        from chimerax.core.atomic import AtomicStructure
        all_chains = []
        for m in self.session.models.list(type=AtomicStructure):
            all_chains.extend(m.chains)
        all_chains.sort(key=str)
        self.chain_combobox.clear()
        for chain in all_chains:
            self.chain_combobox.addItem(str(chain), userData=chain)

    def _update_blast_results(self, blast_results):
        if blast_results is None:
            self.results_view.setHtml(_EmptyPage)
        else:
            self.results_view.setHtml(_NonEmptyPage)

    def delete(self):
        t = session.triggers
        t.remove_handler(self._add_handler)
        t.remove_handler(self._remove_handler)
