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

def alphafold_predict(session, sequence):
    if not _is_alphafold_available(session):
        return
    ar = show_alphafold_run(session)
    if ar.running:
        from chimerax.core.errors import UserError
        raise UserError('AlphaFold prediction currently running.  Can only run one at a time.')
    ar.start(sequence)

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class AlphaFoldRun(ToolInstance):
    _ipython_notebook_url = 'https://colab.research.google.com/github/RBVI/ChimeraX/blob/develop/src/bundles/alphafold/src/alphafold_predict_colab.ipynb'
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self._running = False
        self._sequence = None	# Sequence instance or subclass such as Chain
        self._download_directory = None

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        from chimerax.ui.widgets.htmlview import ChimeraXHtmlView, create_chimerax_profile
        profile = create_chimerax_profile(parent, download = self._download_requested,
                                          storage_name = 'AlphaFold')
        self._browser = b = ChimeraXHtmlView(session, parent, size_hint = (800,500), profile=profile)
        layout.addWidget(b)

        tw.manage(placement=None)

    def start(self, sequence):
        colab_started = (self._sequence is not None)
        self._sequence = sequence
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
        set_seq_javascript = ('document.querySelector("paper-input").setAttribute("value", "%s")'
                              % self._sequence.characters + '; ' +
                              'document.querySelector("paper-input").dispatchEvent(new Event("change"))')
        p.runJavaScript(set_seq_javascript)

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
        # "item" is an instance of QWebEngineDownloadItem
        filename = item.suggestedFileName()
        dir = self._download_directory
        if dir is None:
            self._download_directory = dir = self._unique_download_directory()
        item.setDownloadDirectory(dir)
        if filename == 'best_model.pdb':
            item.finished.connect(self._downloaded_best_model)
        elif  filename == 'results.zip':
            item.finished.connect(self._unzip_results)
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
    
    def _downloaded_best_model(self, *args, **kw):
        self.session.logger.info('AlphaFold prediction finished\n' +
                                 'Results in %s' % self._download_directory)
        self._open_prediction()

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
        self.session.models.add(models)

        from chimerax.atomic import Chain
        if isinstance(self._sequence, Chain):
            chain = self._sequence
            from .fetch import _color_by_confidence
            from .match import _align_to_chain
            for m in models:
                _color_by_confidence(m)
                _align_to_chain(m, chain)
    
    def _unzip_results(self, *args, **kw):
        from os.path import join, exists
        path = join(self._download_directory, 'results.zip')
        if exists(path):
            import zipfile
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(self._download_directory)

# ------------------------------------------------------------------------------
#
def _is_alphafold_available(session):
    '''Check if the AlphaFold web service has been discontinued or is down.'''
    url = 'https://www.rbvi.ucsf.edu/chimerax/data/status/alphafold_v1.html'
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
    from chimerax.core.commands import CmdDesc, register
    from chimerax.atomic import SequenceArg
    desc = CmdDesc(
        required = [('sequence', SequenceArg)],
        synopsis = 'Predict a structure with AlphaFold'
    )
    register('alphafold predict', desc, alphafold_predict, logger=logger)

