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

def close(session, models=None):
    '''
    Close models.

    Parameters
    ----------
    models : list of models
        These models and any submodels are closed.  If models is none all models are closed.
    '''
    m = session.models
    if models is None:
        models = m.list()

    # Avoid closing grouping models if not all child models are closed.
    # This is so that "close ~#1.1" does not close grouping model #1.
    hc = have_all_child_models(models)
    cmodels = [cm for cm in models if cm in hc]

    m.close(cmodels)

def have_all_child_models(models):
    '''
    Return a set containing those models in the given models that have all
    child and descendant models in the given models.
    '''
    contains = set()
    mset = set(models)
    for m in models:
        _contains_model_tree(m, mset, contains)
    return contains

def _contains_model_tree(m, mset, contains):
    if not m in mset:
        return

    cmodels = m.child_models()
    for c in cmodels:
        _contains_model_tree(c, mset, contains)

    for c in cmodels:
        if not c in contains:
            return

    contains.add(m)

def close_session(session):
    session.reset()

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ModelsArg
    desc = CmdDesc(optional=[('models', ModelsArg)],
                   synopsis='close models')
    register('close', desc, close, logger=logger)
    desc = CmdDesc(synopsis="clear session contents")
    register('close session', desc, close_session, logger=logger)
