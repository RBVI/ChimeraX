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

def rename(session, models, name):
    '''
    Rename a model.  Might extend this command in the future to rename chains, residues, atoms...

    Parameters
    ----------
    models : list of models
    name : string
    '''
    for m in models:
        m.name = name

def register_command(session):
    from . import CmdDesc, register, TopModelsArg, StringArg
    desc = CmdDesc(required=[('models', TopModelsArg),
                             ('name', StringArg)],
                   synopsis='rename a model')
    register('rename', desc, rename)
