# we depend on the fact that entry information
# is fetched before chain information
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
import sys


_default_instance_prefix = "bp"
_instance_map = {}


def _make_instance_name():
    n = 1
    while True:
        instance_name = _default_instance_prefix + str(n)
        if instance_name not in _instance_map:
            return instance_name
        n += 1


def find(instance_name):
    return _instance_map.get(instance_name, None)


def find_match(instance_name):
    if instance_name is None:
        if len(_instance_map) == 1:
            for name, inst in _instance_map.items():
                return inst
        from chimerax.core.errors import UserError
        if len(_instance_map) > 1:
            raise UserError("no name specified with multiple "
                            "active blastprotein instances")
        else:
            raise UserError("no active blastprotein instance")
    try:
        return _instance_map[instance_name]
    except KeyError:
        from chimerax.core.errors import UserError
        raise UserError("no blastprotein instance named \"%s\"" % instance_name)


class ToolUI(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True
    CUSTOM_SCHEME = "blastprotein"

    help = "help:user/tools/blastprotein.html"

    def __init__(self, session, tool_name, blast_results=None, params=None, *,
                 instance_name=None):
        # ``session`` - ``chimerax.core.session.Session`` instance
        # ``tool_name`` - ``str`` instance

        # Set name displayed on title bar
        if instance_name is None:
            instance_name = _make_instance_name()
        _instance_map[instance_name] = self
        display_name = "%s [name: %s]" % (tool_name, instance_name)

        # Initialize base class.  ``size_hint`` is the suggested
        # initial tool size in pixels.  For debugging, add
        # "log_errors=True" to get Javascript errors logged
        # to the ChimeraX log window.
        super().__init__(session, display_name, size_hint=(575, 400),
                         log_errors=True)
        self._initialized = False
        self._instance_name = instance_name
        self._params = params
        self._chain_map = {}
        self._hits = None
        self._ref_atomspec = None
        self._blast_results = blast_results
        self._sequences = {}
        self._viewer_index = 1
        self._build_ui()

    def _build_ui(self):
        # Fill in html viewer with initial page in the module
        import os.path
        html_file = os.path.join(os.path.dirname(__file__), "gui.html")
        import pathlib
        self.html_view.setUrl(pathlib.Path(html_file).as_uri())

    def handle_scheme(self, url):
        """
        Called when GUI sets browser URL location.
        url: Qt.QtCore.QUrl instance
        """

        # First check that the path is a real command
        command = url.path()
        if command == "initialize":
            self.initialize()
        elif command == "search":
            self.blast(url)
        elif command == "load":
            self.load(url)
        elif command == "update_models":
            self.update_models()
        elif command == "show_mav":
            from urllib.parse import parse_qs
            query = parse_qs(url.query())
            if 'ids' not in query:
                self.session.logger.info("No sequences to show.")
            else:
                id_list = [int(n) for n in query["ids"][0].split(',')]
                self.show_mav(id_list)
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown blastprotein command: %s" % command)


    def update_models(self, trigger=None, trigger_data=None):
        """
        Update the <select> options in the web form with current
        list of chains in all atomic structures.  Also enable/disable
        submit buttons depending on whether there are any structures open.
        """

        # Get the list of atomic structures
        if not self._initialized:
            return
        from chimerax.atomic import AtomicStructure, Residue
        all_chains = []
        for m in self.session.models.list(type=AtomicStructure):
            all_chains.extend([chain for chain in m.chains
                                     if chain.polymer_type == Residue.PT_AMINO])
        self._chain_map = {chain.atomspec.replace('/', ' ').strip():chain
                           for chain in all_chains}

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
        self._initialized = True
        self.update_models()
        if self._params:
            self._show_params(self._params)
            for k, v in self._params:
                if k == "chain":
                    self._ref_atomspec = v
        if self._blast_results:
            self._show_results(self._ref_atomspec, self._blast_results)
        elif self._hits:
            self._show_hits()


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
        max_seqs = self._arg_max_seqs(query["maxSeqs"])
        cmd_text = ["blastprotein", chain,
                    "database", database,
                    "cutoff", cutoff,
                    "matrix", matrix,
                    "maxSeqs", max_seqs,
                    "name", str(self._instance_name)]
        cmd = ' '.join(cmd_text)
        from chimerax.core.commands import run
        run(self.session, cmd)
        self.html_view.runJavaScript("status('Waiting for results');")

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

    def _arg_max_seqs(self, max_seqs):
        if len(max_seqs) != 1:
            from chimerax.core.errors import UserError
            raise UserError("BlastProtein is limited to one hit limit only.")
        return max_seqs[0]

    #
    # Code for displaying matches as multiple sequence alignment
    #

    def show_mav_cmd(self, selected):
        import json
        js = "show_mav(%s);" % json.dumps(selected)
        self.html_view.runJavaScript(js)

    def show_mav(self, ids):
        # Collect names and sequences of selected matches.
        # All sequences should have the same length because
        # they include gaps generated from BLAST alignment.
        ids.insert(0, 0)
        names = []
        seqs = []
        for sid in ids:
            name, seq = self._sequences[sid]
            names.append(name)
            seqs.append(seq)
        # Find columns that are gaps in all sequences and remove them.
        all_gaps = set()
        for i in range(len(seqs[0])):
            for seq in seqs:
                if seq[i].isalpha():
                    break
            else:
                all_gaps.add(i)
        if all_gaps:
            for i in range(len(seqs)):
                seq = seqs[i]
                new_seq = ''.join([seq[n] for n in range(len(seq))
                                   if n not in all_gaps])
                seqs[i] = new_seq
        # Generate multiple sequence alignment file
        # Ask sequence viewer to display alignment
        from chimerax.atomic import Sequence
        seqs = [Sequence(name=name, characters=seqs[i])
                for i, name in enumerate(names)]
        name = "%s [%d]" % (self._instance_name, self._viewer_index)
        self.session.alignments.new_alignment(seqs, name)

    def _write_fasta(self, f, name, seq):
        print(name, len(seq))
        print(">", name, file=f)
        block_size = 60
        for i in range(0, len(seq), block_size):
            print(seq[i:i+block_size], file=f)

    #
    # Callbacks for BlastProteinJob
    #

    def job_finished(self, job, blast_results, params):
        self._params = params
        self._show_params(params)
        self._blast_results = blast_results
        self._show_results(job.atomspec, self._blast_results)
        self.html_view.runJavaScript("status('');")

    def _show_results(self, atomspec, blast_results):
        # blast_results is either None or a subclass of databases.Database
        self._ref_atomspec = atomspec
        hits = []
        self._sequences = {}
        if blast_results is not None:
            query_match = blast_results.parser.matches[0]
            if self._ref_atomspec:
                name = self._ref_atomspec
            else:
                name = query_match.name
            self._sequences[0] = (name, query_match.sequence)

            match_chains = {}

            for n, m in enumerate(blast_results.parser.matches[1:]):
                sid = n + 1
                hit = {"id":sid, "evalue":m.evalue, "score":m.score,
                       "description":m.description}
                if m.match:
                    hit["name"] = m.match
                    hit["url"] = "%s:load?match=%s" % (self.CUSTOM_SCHEME, m.match)
                    match_chains[m.match] = hit
                else:
                    hit = blast_results.add_url(hit, m)
                hits.append(hit)
                self._sequences[sid] = (hit["name"], m.sequence)

            # TODO: Make what this function does more explicit. It works on the
            # hits that are in match_chain's hit dictionary, but that's not
            # immediately clear.
            blast_results.add_info(self.session, match_chains)

        self._hits = hits
        self._show_hits()

    def _show_params(self, params):
        import json
        js = "params_update(%s)" % json.dumps(params)
        self.html_view.runJavaScript(js)

    def _show_hits(self):
        import json
        js = "table_update(%s);" % json.dumps(self._hits)
        self.html_view.runJavaScript(js)

    def job_failed(self, job, error):
        from chimerax.core.errors import UserError
        raise UserError("BlastProtein failed: %s" % error)


    #
    # Code for loading (and spatially matching) a match entry
    #

    def load(self, url) -> None:
        """Load the model from the results database.
        url: Instance of Qt.QtCore.QUrl
        """
        from chimerax.core.commands import run
        from urllib.parse import parse_qs
        query = parse_qs(url.query())
        for code in query["match"]:
            models, chain_id = self._blast_results.load_model(self.session, code, self._ref_atomspec)
            if not self._ref_atomspec:
                run(self.session, "select clear")
            else:
                for m in models:
                    self._blast_results.display_model(self.session, self._ref_atomspec, m, chain_id)



    #
    # Code for saving and restoring session
    #

    def take_snapshot(self, session, flags):
        data = {
            "version": 1,
            "_super": super().take_snapshot(session, flags),
            "_chain_map": self._chain_map,
            "_hits": self._hits,
            "_params": self._params,
            "_ref_atomspec": self._ref_atomspec,
            "_sequences": self._sequences,
            "_instance_name": self._instance_name,
            "_viewer_index": self._viewer_index,
        }
        return data


    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data["_super"])
        inst._initialized = False
        inst._chain_map = data["_chain_map"]
        inst._hits = data["_hits"]
        inst._params = data["_params"]
        inst._ref_atomspec = data["_ref_atomspec"]
        inst._blast_results = None
        inst._sequences = data["_sequences"]
        try:
            inst._instance_name = data["_instance_name"]
        except KeyError:
            inst._instance_name = _make_instance_name()
        inst._viewer_index = data.get("_viewer_index", 1)
        inst._build_ui()
        return inst
