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

def alphafold_predict(session, chain):
    ar = show_alphafold_run(session)
    if ar.running:
        from chimerax.core.errors import UserError
        raise UserError('AlphaFold prediction currenlty running.  Can only run one at a time.')
    ar.start(chain)

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class AlphaFoldRun(ToolInstance):
    _ipython_notebook_url = 'https://colab.research.google.com/github/RBVI/ChimeraX/blob/develop/src/bundles/alphafold/src/alphafold_predict_colab.ipynb'
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self._running = False
        self._chain = None
        self._prediction_path = None

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

    def start(self, chain):
        colab_started = (self._chain is not None)
        self._chain = chain
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
                              % self._chain.characters + '; ' +
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
        if item.suggestedFileName().endswith('.pdb'):
            path = self._unique_download_path()
            self._prediction_path = path
            from os.path import dirname, basename
            item.setDownloadDirectory(dirname(path))
            item.setDownloadFileName(basename(path))
            item.finished.connect(self._download_finished)
        item.accept()

    def _unique_download_path(self):
        from os.path import expanduser, join, exists
        ddir = expanduser('~/Downloads')
        adir = join(ddir, 'ChimeraX', 'AlphaFold')
        from os import makedirs
        makedirs(adir, exist_ok = True)
        for i in range(1,100000):
            path = join(adir, 'prediction_%d.pdb' % i)
            if not exists(path):
                break
        return path
    
    def _download_finished(self, *args, **kw):
        self.session.logger.info('AlphaFold prediction finished')
        self._open_prediction()

    def _open_prediction(self):
        path = self._prediction_path
        from os.path import exists
        if not exists(path):
            self.session.logger.warning('Downloaded prediction file not found: %s' % path)
            return

        from chimerax.pdb import open_pdb
        models, msg = open_pdb(self.session, path)
        self.session.models.add(models)
        for m in models:
            from .match import _align_to_chain
            _align_to_chain(m, self._chain)
        
# ------------------------------------------------------------------------------
#
def show_alphafold_run(session):
    ar = AlphaFoldRun.get_singleton(session)
    return ar
    
# ------------------------------------------------------------------------------
#
def register_alphafold_predict_command(logger):
    from chimerax.core.commands import CmdDesc, register
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        synopsis = 'Predict a structure with AlphaFold'
    )
    register('alphafold predict', desc, alphafold_predict, logger=logger)

