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

# ToolUI classes may also override
#   "delete" - called to clean up before instance is deleted
#
from chimerax.core.tools import ToolInstance

_EmptyPage = "<h2>Please select a chain and press <b>BLAST</b></h2>"
_InProgressPage = "<h2>BLAST search in progress&hellip;</h2>"


class ToolUI(ToolInstance):

    SESSION_ENDURING = False
    CUSTOM_SCHEME = "blastprotein"
    REF_ID_URL = "https://www.ncbi.nlm.nih.gov/protein/%s"
    KNOWN_IDS = ["ref","gi"]

    def __init__(self, session, tool_name, blast_results=None, atomspec=None):
        # Standard template stuff
        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "Blast Protein"
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        parent = self.tool_window.ui_area

        # UI consists of a chain selector and search button on top
        # and HTML widget below for displaying results.
        # Layout all the widgets
        from PyQt5.QtWidgets import QGridLayout, QLabel, QComboBox, QPushButton
        from chimerax.core.ui.widgets import HtmlView
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
                                     schemes=[self.CUSTOM_SCHEME])
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
        self._update_blast_results(blast_results, atomspec)

    def _blast_cb(self, _):
        from .job import BlastProteinJob
        n = self.chain_combobox.currentIndex()
        if n < 0:
            return
        chain = self.chain_combobox.itemData(n)
        BlastProteinJob(self.session, chain.characters, chain.atomspec(),
                        finish_callback=self._blast_job_finished)
        self.results_view.setHtml(_InProgressPage)

    def _update_chains(self, trigger=None, trigger_data=None):
        from chimerax.core.atomic import AtomicStructure
        n = self.chain_combobox.currentIndex()
        selected_chain = None if n == -1 else self.chain_combobox.itemData(n)
        all_chains = []
        for m in self.session.models.list(type=AtomicStructure):
            all_chains.extend(m.chains)
        all_chains.sort(key=str)
        self.chain_combobox.clear()
        for chain in all_chains:
            self.chain_combobox.addItem(str(chain), userData=chain)
        if selected_chain:
            n = self.chain_combobox.findData(selected_chain)
            self.chain_combobox.setCurrentIndex(n)

    def _blast_job_finished(self, blast_results, job):
        self._update_blast_results(blast_results, job.atomspec)

    def _update_blast_results(self, blast_results, atomspec):
        # blast_results is either None or a blastp_parser.Parser
        self.ref_atomspec = atomspec
        if atomspec:
            from chimerax.core.commands import AtomSpecArg
            arg = AtomSpecArg.parse(atomspec, self.session)[0]
            s = arg.evaluate(self.session)
            chains = s.atoms.residues.unique_chains
            if len(chains) == 1:
                n = self.chain_combobox.findData(chains[0])
                self.chain_combobox.setCurrentIndex(n)
        if blast_results is None:
            self.results_view.setHtml(_EmptyPage)
        else:
            html = ["<h2>Blast Protein ",
                    "<small>(an <a href=\"http://www.rbvi.ucsf.edu\">RBVI</a> "
                    "web service)</small> Results</h2>",
                    "<table><tr>"
                    "<th>Name</th>"
                    "<th>E&#8209;Value</th>"
                    "<th>Score</th>"
                    "<th>Description</th>"
                    "</tr>"]
            for m in blast_results.matches[1:]:
                if m.pdb:
                    name = "<a href=\"%s:%s\">%s</a>" % (self.CUSTOM_SCHEME,
                                                         m.pdb, m.pdb)
                else:
                    import re
                    mdb = None
                    mid = None
                    for known in self.KNOWN_IDS:
                        match = re.search(r"\b%s\|([^|]+)\|" % known, m.name)
                        if match is not None:
                            mdb = known
                            mid = match.group(1)
                            break
                    if match is None:
                        name = m.name
                    else:
                        url = self.REF_ID_URL % mid
                        name = "<a href=\"%s\">%s (%s)</a>" % (url, mid, mdb)
                html.append("<tr><td>%s</td><td>%s</td>"
                            "<td>%s</td><td>%s</td></tr>" %
                            (name, "%.3g" % m.evalue,
                             str(m.score), m.description))
            html.append("</table>")
            self.results_view.setHtml('\n'.join(html))

    def _navigate(self, info):
        # "info" is an instance of QWebEngineUrlRequestInfo
        url = info.requestUrl()
        scheme = url.scheme()
        if scheme == self.CUSTOM_SCHEME:
            # self._load_pdb(url.path())
            self.session.ui.thread_safe(self._load_pdb, url.path())
        # For now, we only intercept our custom scheme.  All other
        # requests are processed normally.

    def _load_pdb(self, code):
        from chimerax.core.commands import run
        from chimerax.core.atomic import AtomicStructure
        parts = code.split("_", 1)
        if len(parts) == 1:
            pdb_id = parts[0]
            chain_id = None
        else:
            pdb_id, chain_id = parts
        models = run(self.session, "open pdb:%s" % pdb_id)[0]
        if isinstance(models, AtomicStructure):
            models = [models]
        if not self.ref_atomspec:
            run(self.session, "select clear")
        for m in models:
            if chain_id:
                spec = m.atomspec() + '/' + chain_id
            else:
                spec = m.atomspec()
            if self.ref_atomspec:
                run(self.session, "matchmaker %s to %s" % (spec,
                                                           self.ref_atomspec))
            else:
                run(self.session, "select add %s" % spec)

    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._add_handler)
        t.remove_handler(self._remove_handler)
        super().delete()

    SESSION_SAVE = False
    
    def take_snapshot(self, session, flags):
        # For now, do not save anything in session.
        # Need to figure out which attributes (like UI widgets)
        # should start with _ so that they are not saved in sessions.
        # And add addition data to superclass data.
        return super().take_snapshot(session, flags)

    @classmethod
    def restore_snapshot(cls, session, data):
        # For now do nothing.  Should unpack data and restart tool.
        return None
