# vim: set expandtab shiftwidth=4 softtabstop=4:

# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance

_EmptyPage = "<h2>This space intentionally left blank</h2>"
_InProgressPage = "<h2>BLAST search in progress&hellip;</h2>"
_NonEmptyPage = "<h2>This space unintentionally left blank</h2>"


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
        # Layout all the widgets
        from PyQt5.QtWidgets import QGridLayout, QLabel, QComboBox, QPushButton
        from .htmlview import HtmlView
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Chain:")
        layout.addWidget(label, 0, 0)
        self.chain_combobox = QComboBox()
        layout.addWidget(self.chain_combobox, 0, 1)
        button = QPushButton("BLAST")
        button.clicked.connect(self._blast_cb)
        layout.addWidget(button, 0, 2)
        self.results_view = HtmlView(parent, size_hint=(575, 300),
                                     interceptor=self._navigate,
                                     schemes=["fetch"])
        layout.addWidget(self.results_view, 1, 0, 1, 3)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 10)
        layout.setColumnStretch(2, 0)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 10)
        parent.setLayout(layout)

        # Register for model addition/removal so we can update chain list
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        t = session.triggers
        self._add_handler = t.add_handler(ADD_MODELS, self._update_chains)
        self._remove_handler = t.add_handler(REMOVE_MODELS, self._update_chains)

        # Set widget values and go
        self._update_chains()
        self._update_blast_results(blast_results)

    def _blast_cb(self, _):
        from .job import BlastPDBJob
        n = self.chain_combobox.currentIndex()
        chain = self.chain_combobox.itemData(n)
        BlastPDBJob(self.session, chain.characters,
                    finish_callback=self._update_blast_results)
        self.results_view.setHtml(_InProgressPage)

    def _update_chains(self, trigger=None, trigger_data=None):
        from chimerax.core.atomic import AtomicStructure
        all_chains = []
        for m in self.session.models.list(type=AtomicStructure):
            all_chains.extend(m.chains)
        all_chains.sort(key=str)
        self.chain_combobox.clear()
        for chain in all_chains:
            self.chain_combobox.addItem(str(chain), userData=chain)

    def _update_blast_results(self, blast_results, job=None):
        # blast_results is either None or a blastp_parser.Parser
        if blast_results is None:
            self.results_view.setHtml(_EmptyPage)
        else:
            html = ["<table><tr>"
                    "<th>Name</th>"
                    "<th>E&#8209;Value</th>"
                    "<th>Score</th>"
                    "<th>Description</th>"
                    "</tr>"]
            for m in blast_results.matches[1:]:
                name = m.pdb if m.pdb else m.name
                name_link = "<a href=\"fetch:%s\">%s</a>" % (name, name)
                html.append("<tr><td>%s</td><td>%s</td>"
                            "<td>%s</td><td>%s</td></tr>" %
                            (name_link, "%.1e" % m.evalue,
                             str(m.score), m.description))
            html.append("</table>")
            self.results_view.setHtml('\n'.join(html))

    def _navigate(self, info):
        # "info" is an instance of QWebEngineUrlRequestInfo
        print("_navigate", info, info.requestUrl())

    def delete(self):
        t = session.triggers
        t.remove_handler(self._add_handler)
        t.remove_handler(self._remove_handler)
