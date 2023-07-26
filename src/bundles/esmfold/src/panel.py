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

# -----------------------------------------------------------------------------
# Panel for searching AlphaFold database or predicting structure from sequence.
#
from chimerax.alphafold.panel import PredictedStructureGUI
class ESMFoldGUI(PredictedStructureGUI):
    method = 'ESMFold'
    command = 'esmfold'
    can_use_structure_templates = False
    can_minimize = False
    help = 'help:user/tools/esmfold.html'

    def show_coloring_gui(self):
        show_esmfold_coloring_panel(self.session)

    def show_error_plot(self):
        from . import pae
        pae.show_esmfold_error_plot_panel(self.session)

    def default_results_directory(self):
        from . import predict
        return predict.results_directory


# -----------------------------------------------------------------------------
#
def esmfold_panel(session, create = False):
    return ESMFoldGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_esmfold_panel(session):
    return esmfold_panel(session, create = True)

# -----------------------------------------------------------------------------
# Panel for coloring predicted structures by confidence or alignment errors.
#
from chimerax.alphafold.colorgui import PredictedStructureColoringGUI
class ESMFoldColoringGUI(PredictedStructureColoringGUI):
    method = 'ESMFold'
    default_confidence_cutoff = 0.5
    help = 'help:user/tools/esmfold.html#coloring'

    def is_predicted_model(self, m):
        return _is_esmfold_model(m)

# -----------------------------------------------------------------------------
#
def _is_esmfold_model(m):
    if 'ESM' in m.name or 'MGYP' in m.name:
        return True
    
    title_recs = m.metadata.get('TITLE', None)
    if title_recs:
        for t in title_recs:
            if 'ESMFOLD' in t:
                return True
    return False
    
# -----------------------------------------------------------------------------
#
def esmfold_coloring_panel(session, create = False):
    return ESMFoldColoringGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_esmfold_coloring_panel(session):
    return esmfold_coloring_panel(session, create = True)
