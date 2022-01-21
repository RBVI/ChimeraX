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

def alphafold_predict(session, sequences, prokaryote = False):
    if not _is_alphafold_available(session):
        return
    ar = show_alphafold_run(session)
    if ar.running:
        from chimerax.core.errors import UserError
        raise UserError('AlphaFold prediction currently running.  Can only run one at a time.')
    ar.start(sequences, prokaryote)

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class AlphaFoldRun(ToolInstance):
    _ipython_notebook_url = 'https://colab.research.google.com/github/RBVI/ChimeraX/blob/develop/src/bundles/alphafold/src/alphafold21_predict_colab.ipynb'
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self._running = False
        self._sequences = None	# List of Sequence or Chain instances
        self._prokaryote = False
        self._download_directory = None

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        # Avoid warning message from Qt when closing colab html panel.
        # "WARNING: Release of profile requested but WebEnginePage still not deleted. Expect troubles !"
        # After the html window is destroyed we remove the profile.
        # Related to ChimeraX bug report #3761.
        profile_parent = None
        
        from chimerax.ui.widgets.htmlview import ChimeraXHtmlView, create_chimerax_profile
        profile = create_chimerax_profile(profile_parent, download = self._download_requested,
                                          storage_name = 'AlphaFold')
        self._browser = b = ChimeraXHtmlView(session, parent, size_hint = (800,500), profile=profile)
        b.destroyed.connect(lambda *,profile=profile: profile.deleteLater())
        layout.addWidget(b)

        tw.manage(placement=None)

    def start(self, sequences, prokaryote = False):
        colab_started = (self._sequences is not None)
        self._sequences = sequences
        self._prokaryote = prokaryote
        if not colab_started:
            b = self._browser
            from Qt.QtCore import QUrl
            b.setUrl(QUrl(self._ipython_notebook_url))
            b.page().loadFinished.connect(self._page_loaded)
        else:
            self._run()

    def _page_loaded(self, okay):
        if okay:
            # Need to delay setting sequence and running or those do nothing
            # probably because it is still waiting for some asynchronous setup.
            delay_millisec = 1000
            self._keep_timer_alive = self.session.ui.timer(delay_millisec, self._run)
            # If we don't save the timer in a variable it is deleted and never fires.

    def _run(self):
        self._set_colab_sequence()
        self._run_colab()
        self.session.logger.info('Running AlphaFold prediction')

    def _set_colab_sequence(self):
        p = self._browser.page()
        seqs = ','.join(seq.characters for seq in self._sequences)
        if self._prokaryote:
            seqs = 'prokaryote,' + seqs
        set_seqs_javascript = ('document.querySelector("paper-input").setAttribute("value", "%s")'
                               % seqs + '; ' +
                              'document.querySelector("paper-input").dispatchEvent(new Event("change"))')
        p.runJavaScript(set_seqs_javascript)

    def _run_colab(self):
        p = self._browser.page()
        p.runJavaScript('document.querySelector("colab-run-button").click()')
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, AlphaFoldRun, 'AlphaFold Run', create=create)

    @property
    def running(self):
        return self._running
    
    def _download_requested(self, item):
        # "item" is an instance of QWebEngineDownloadItem in Qt 5,
        # or QWebEngineDownloadRequest in Qt 6.
        filename = item.suggestedFileName()
        if filename == 'best_model.pdb':
            item.cancel()  # Historical.  Used to just download pdb file.
            return
        dir = self._download_directory
        if dir is None:
            self._download_directory = dir = self._unique_download_directory()
        item.setDownloadDirectory(dir)
        if  filename == 'results.zip':
            if hasattr(item, 'finished'):
                item.finished.connect(self._unzip_results)		# Qt 5
            else:
                item.isFinishedChanged.connect(self._unzip_results)	# Qt 6
        item.accept()

    def _unique_download_directory(self):
        from os.path import expanduser, join, exists
        ddir = expanduser('~/Downloads')
        adir = join(ddir, 'ChimeraX', 'AlphaFold')
        from os import makedirs
        makedirs(adir, exist_ok = True)
        for i in range(1,1000000):
            path = join(adir, 'prediction_%d' % i)
            if not exists(path):
                break
        makedirs(path, exist_ok = True)
        return path

    def _open_prediction(self):
        from os.path import join, exists
        path = join(self._download_directory, 'best_model.pdb')
        if not exists(path):
            self.session.logger.warning('Downloaded prediction file not found: %s' % path)
            return

        from chimerax.pdb import open_pdb
        models, msg = open_pdb(self.session, path)
        from .match import _set_alphafold_model_attributes
        _set_alphafold_model_attributes(models)

        # TODO: Rename and align multiple chains and log info.
        #   AlphaFold relaxed models have chains A,B,C,... I believe ordered by input sequence order.
        #   The Alphafold unrelaxed models use B,C,D,....
        from chimerax.atomic import Chain
        chains = [seq for seq in self._sequences if isinstance(seq, Chain)]
        if len(chains) == len(self._sequences):
            from .match import _align_to_chain, _rename_chains
            chain_ids = [chain.chain_id for chain in chains]
            longest_chain = max(chains, key = lambda c: c.num_residues)
            for m in models:
                _rename_chains(m, chain_ids)
                _align_to_chain(m, longest_chain)
            if len(chains) == 1:
                from .fetch import _log_chain_info
                _log_chain_info(models, chains[0].name)

        self.session.models.add(models)

        from .fetch import _color_by_confidence
        for m in models:
            _color_by_confidence(m)
    
    def _unzip_results(self, *args, **kw):
        if self._download_directory is None:
            return  # If user manages to request two downloads before one completes. Bug #5412
        from os.path import join, exists
        path = join(self._download_directory, 'results.zip')
        if exists(path):
            import zipfile
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(self._download_directory)
        self._open_prediction()
        self.session.logger.info('AlphaFold prediction finished\n' +
                                 'Results in %s' % self._download_directory)
        self._download_directory = None  # Make next run go in a new directory

# ------------------------------------------------------------------------------
#
def _is_alphafold_available(session):
    '''Check if the AlphaFold web service has been discontinued or is down.'''
    url = 'https://www.rbvi.ucsf.edu/chimerax/data/status/alphafold21.html'
    import requests
    try:
        r = requests.get(url)
    except requests.exceptions.ConnectionError:
        return True
    if r.status_code == 200:
        session.logger.error(r.text, is_html = True)
        return False
    return True

# ------------------------------------------------------------------------------
#
def show_alphafold_run(session):
    ar = AlphaFoldRun.get_singleton(session)
    return ar
    
# ------------------------------------------------------------------------------
#
def register_alphafold_predict_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg
    from chimerax.atomic import SequencesArg
    desc = CmdDesc(
        required = [('sequences', SequencesArg)],
        keyword = [('prokaryote', BoolArg)],
        synopsis = 'Predict a structure with AlphaFold'
    )
    register('alphafold predict', desc, alphafold_predict, logger=logger)

