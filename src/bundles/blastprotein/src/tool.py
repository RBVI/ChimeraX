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

from chimerax.ui import HtmlToolInstance


class ToolUI(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "blastprotein"

    def __init__(self, session, tool_name, blast_results=None, atomspec=None):
        # ``session`` - ``chimerax.core.session.Session`` instance
        # ``tool_name`` - ``str`` instance

        # Set name displayed on title bar
        self.display_name = "Blast Protein"
        from . import pdbinfo
        self._pdbinfo_list = (pdbinfo.entry_info, pdbinfo.chain_info,
                              pdbinfo.ligand_info)

        # Initialize base class.  ``size_hint`` is the suggested
        # initial tool size in pixels.  For debugging, add
        # "log_errors=True" to get Javascript errors logged
        # to the ChimeraX log window.
        super().__init__(session, tool_name, size_hint=(575, 400),
                         log_errors=True)
        self._chain_map = {}
        self._ref_atomspec = atomspec
        self._blast_results = blast_results
        self._build_ui()

    def _build_ui(self):
        # Fill in html viewer with initial page in the module
        import os.path
        html_file = os.path.join(os.path.dirname(__file__), "gui.html")
        import pathlib
        self.html_view.setUrl(pathlib.Path(html_file).as_uri())

    def handle_scheme(self, url):
        # Called when GUI sets browser URL location.
        # ``url`` - ``PyQt5.QtCore.QUrl`` instance

        # First check that the path is a real command
        command = url.path()
        if command == "initialize":
            self.initialize()
        elif command == "search":
            self.blast(url)
        elif command == "load":
            self.load_pdb(url)
        elif command == "update_models":
            self.update_models()
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown blastprotein command: %s" % command)


    def update_models(self, trigger=None, trigger_data=None):
        # Update the <select> options in the web form with current
        # list of chains in all atomic structures.  Also enable/disable
        # submit buttons depending on whether there are any structures open.

        # Get the list of atomic structures
        from chimerax.atomic import AtomicStructure
        all_chains = []
        for m in self.session.models.list(type=AtomicStructure):
            all_chains.extend(m.chains)
        self._chain_map = {str(chain):chain for chain in all_chains}

        # Construct Javascript for updating <select> and submit buttons
        if not self._chain_map:
            chain_labels = []
        else:
            chain_labels = list(sorted(self._chain_map.keys()))
        import json
        js = "chains_update(%s);" % json.dumps(chain_labels)
        self.html_view.runJavaScript(js)


    #
    # Initialize after GUI is ready
    #


    def initialize(self):
        self.update_models()
        if self._blast_results:
            self._show_results(self._ref_atomspec, self._blast_results)
            self._blast_results = None


    #
    # Code for running BLAST search
    #

    def blast(self, url):
        # Collect the optional parameters from URL query parameters
        # and construct a command to execute
        from urllib.parse import parse_qs
        query = parse_qs(url.query())
        chain = self._arg_chain(query["chain"])
        database = self._arg_database(query["database"])
        cutoff = self._arg_cutoff(query["cutoff"])
        matrix = self._arg_matrix(query["matrix"])
        max_hits = self._arg_max_hits(query["max_hits"])
        cmd_text = ["blastprotein", chain,
                    "database", database,
                    "cutoff", cutoff,
                    "matrix", matrix,
                    "max_hits", max_hits,
                    "tool_id", str(self.id)]
        cmd = ' '.join(cmd_text)
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _arg_chain(self, chains):
        if len(chains) != 1:
            from chimerax.core.errors import UserError
            raise UserError("BlastProtein is limited to one chain only.")
        chain = self._chain_map[chains[0]]
        return chain.atomspec

    def _arg_database(self, databases):
        if len(databases) != 1:
            from chimerax.core.errors import UserError
            raise UserError("BlastProtein is limited to one database only.")
        return databases[0]

    def _arg_cutoff(self, cutoffs):
        if len(cutoffs) != 1:
            from chimerax.core.errors import UserError
            raise UserError("BlastProtein is limited to one cutoff only.")
        return "1e" + cutoffs[0]

    def _arg_matrix(self, matrices):
        if len(matrices) != 1:
            from chimerax.core.errors import UserError
            raise UserError("BlastProtein is limited to one matrix only.")
        return matrices[0]

    def _arg_max_hits(self, max_hits):
        if len(max_hits) != 1:
            from chimerax.core.errors import UserError
            raise UserError("BlastProtein is limited to one hit limit only.")
        return max_hits[0]

    #
    # Callbacks for BlastProteinJob
    #

    def job_finished(self, job, blast_results):
        self._show_results(job.atomspec, blast_results)

    def _show_results(self, atomspec, blast_results):
        # blast_results is either None or a blastp_parser.Parser
        self._ref_atomspec = atomspec
        hits = []
        if blast_results is not None:
            import re
            NCBI_IDS = ["ref","gi"]
            NCBI_ID_URL = "https://www.ncbi.nlm.nih.gov/protein/%s"
            id_pat = re.compile(r"\b(%s)\|([^|]+)\|" % '|'.join(NCBI_IDS))
            pdb_chains = {}
            for m in blast_results.matches[1:]:
                hit = {"evalue":m.evalue, "score":m.score,
                       "description":m.description}
                if m.pdb:
                    hit["name"] = m.pdb
                    hit["url"] = "%s:load?pdb=%s" % (self.CUSTOM_SCHEME, m.pdb)
                    pdb_chains[m.pdb] = hit
                else:
                    mdb = None
                    mid = None
                    match = id_pat.search(m.name)
                    if match is not None:
                        mdb = match.group(1)
                        mid = match.group(2)
                        hit["name"] = "%s (%s)" % (mid, mdb)
                        hit["url"] = NCBI_ID_URL % mid
                    else:
                        hit["name"] = m.name
                        hit["url"] = ""
                hits.append(hit)
            self._add_pdbinfo(pdb_chains)
        import json
        js = "table_update(%s);" % json.dumps(hits)
        self.html_view.runJavaScript(js)

    def _add_pdbinfo(self, pdb_chains):
        chain_ids = pdb_chains.keys()
        for info in self._pdbinfo_list:
            data = info.fetch_info(self.session, chain_ids)
            for chain_id, hit in pdb_chains.items():
                try:
                    hit.update(data[chain_id])
                except KeyError:
                    pass


    def job_failed(self, job, error):
        from chimerax.core.errors import UserError
        raise UserError("BlastProtein failed: %s" % error)


    #
    # Code for loading (and matching) a PDB entry
    #

    def load_pdb(self, url):
        from chimerax.core.commands import run
        from chimerax.atomic import AtomicStructure
        from urllib.parse import parse_qs
        query = parse_qs(url.query())
        for code in query["pdb"]:
            parts = code.split('_', 1)
            if len(parts) == 1:
                pdb_id = parts[0]
                chain_id = None
            else:
                pdb_id, chain_id = parts
            models = run(self.session, "open pdb:%s" % pdb_id)[0]
            if isinstance(models, AtomicStructure):
                models = [models]
            if not self._ref_atomspec:
                run(self.session, "select clear")
            else:
                for m in models:
                    spec = m.atomspec
                    if chain_id:
                        spec += '/' + chain_id
                    if self._ref_atomspec:
                        run(self.session, "matchmaker %s to %s" %
                                          (spec, self._ref_atomspec))
                    else:
                        run(self.session, "select add %s" % spec)
