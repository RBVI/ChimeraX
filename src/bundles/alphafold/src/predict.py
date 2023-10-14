# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

default_results_directory = '~/Downloads/ChimeraX/AlphaFold/prediction_[N]'

def alphafold_predict(session, sequences, minimize = False, templates = False, directory = None):
    if not _is_alphafold_available(session):
        return

    if len(sequences) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'No sequences specified')
    
    if not hasattr(session, '_cite_colabfold'):
        msg = 'Please cite <a href="https://www.nature.com/articles/s41592-022-01488-1">ColabFold: Making protein folding accessible to all. Nature Methods (2022)</a> if you use these predictions.'
        session.logger.info(msg, is_html = True)
        session._cite_colabfold = True  # Only log this message once per session.
        
    ar = show_alphafold_run(session)
    if ar.running:
        from chimerax.core.errors import UserError
        raise UserError('AlphaFold prediction currently running.  Can only run one at a time.')
    ar.start(sequences, energy_minimize=minimize, use_pdb_templates=templates,
             results_directory=directory)
    return ar

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class AlphaFoldRun(ToolInstance):
    # Even though the notebook name alphafold21_predict_colab.ipynb suggests it is AlphaFold 2.1
    # it has been updated to AlphaFold 2.2.0.  I am using the same file for the update so older
    # ChimeraX versions make use of the latest AlphaFold version.
    _ipython_notebook_url = 'https://colab.research.google.com/github/RBVI/ChimeraX/blob/develop/src/bundles/alphafold/src/alphafold21_predict_colab.ipynb'
    # Do not use alphafold_test_colab.ipynb since that was accidentally used by ChimeraX distributions
    # from April 4, 2022 to May 25, 2022.  Bug #6958.  So use a new alphafold_test2_colab.ipynb instead.
    # _ipython_notebook_url = 'https://colab.research.google.com/github/RBVI/ChimeraX/blob/develop/src/bundles/alphafold/src/alphafold_test2_colab.ipynb'
    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        self._running = False
        self._sequences = None	# List of Sequence or Chain instances
        self._energy_minimize = False
        self._use_pdb_templates = False
        self._results_directory = default_results_directory

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

    def start(self, sequences, energy_minimize = False, use_pdb_templates = False,
              results_directory = None):
        colab_started = (self._sequences is not None)
        self._sequences = sequences
        self._energy_minimize = energy_minimize
        self._use_pdb_templates = use_pdb_templates
        self._results_directory = results_directory
        if results_directory is None:
            self._results_directory = default_results_directory
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
        seqs = ','.join(seq.ungapped() for seq in self._sequences)
        if not self._energy_minimize:
            seqs = 'dont_minimize,' + seqs
        if self._use_pdb_templates:
            seqs = 'use_pdb_templates,' + seqs
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
        dir = self._results_directory
        from os.path import isdir
        if not isdir(dir):
            self._results_directory = dir = self._unique_results_directory()
        item.setDownloadDirectory(dir)
        if  filename == 'results.zip':
            if hasattr(item, 'finished'):
                item.finished.connect(self._unzip_results)		# Qt 5
            else:
                item.isFinishedChanged.connect(self._unzip_results)	# Qt 6
        item.accept()

    def _unique_results_directory(self):
        from os.path import expanduser, join, exists
        rdir = expanduser(self._results_directory)
        if '[N]' in rdir:
            for i in range(1,1000000):
                path = rdir.replace('[N]', str(i))
                if not exists(path):
                    rdir = path
                    break
        from os import makedirs
        try:
            makedirs(rdir, exist_ok = True)
        except Exception as e:
            if self._results_directory != default_results_directory:
                msg = f'Could not create AlphaFold prediction directory {rdir}: {str(e)}'
                self.session.logger.warning(msg)
                self._results_directory = default_results_directory
                return self._unique_results_directory()
        if exists(join(rdir, 'results.zip')):
            # Already have a previous prediction in this directory.  Avoid overwriting it.
            msg = f'AlphaFold prediction directory {rdir} already has results.zip, making new directory'
            self.session.logger.warning(msg)
            self._results_directory += '_[N]'
            return self._unique_results_directory()
        return rdir

    def _open_prediction(self):
        from os.path import join, exists
        path = join(self._results_directory, 'best_model.pdb')
        if not exists(path):
            self.session.logger.warning('Downloaded prediction file not found: %s' % path)
            return

        from chimerax.pdb import open_pdb
        models, msg = open_pdb(self.session, path)

        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(self.session, f'open {path}')

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
                # TODO: Improve code to log RMSD per-chain for multimer predictions.
                from .fetch import _log_chain_info
                _log_chain_info(models, _chain_names(chains), prediction_method = 'AlphaFold')

        self.session.models.add(models)

        from .fetch import _color_by_confidence
        for m in models:
            _color_by_confidence(m)

        # Put entry in file history for opening this model
        from chimerax.core.filehistory import remember_file
        remember_file(self.session, path, 'pdb', models)
    
    def _unzip_results(self, *args, **kw):
        dir = self._results_directory
        from os.path import join, exists
        path = join(dir, 'results.zip')
        if exists(path):
            import zipfile
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(dir)
        self.session.logger.info(f'AlphaFold prediction finished\nResults in {dir}')
        self._open_prediction()

# ------------------------------------------------------------------------------
#
def _chain_names(chains):
    sc = {}
    for chain in chains:
        s = chain.structure
        if s not in sc:
            sc[s] = []
        sc[s].append(chain.chain_id)
    return ''.join(str(s) + '/' + ','.join(schains) for s, schains in sc.items())

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
    from chimerax.core.commands import CmdDesc, register, BoolArg, SaveFolderNameArg
    from chimerax.atomic import SequencesArg
    desc = CmdDesc(
        required = [('sequences', SequencesArg)],
        keyword = [('minimize', BoolArg),
                   ('templates', BoolArg),
                   ('directory', SaveFolderNameArg)],
        synopsis = 'Predict a structure with AlphaFold'
    )
    register('alphafold predict', desc, alphafold_predict, logger=logger)

