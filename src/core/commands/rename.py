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

def rename(session, models, name = None, id = None):
    '''
    Rename a model.  Might extend this command in the future to rename chains, residues, atoms...

    Parameters
    ----------
    models : list of models
    name : string
    id : tuple of integers
    '''
    if name is None and id is None:
        from ..errors import UserError
        raise UserError('No name or id option specified for renaming')
        
    if name is not None and (id is None or len(models) == 1):
        for m in models:
            m.name = name

    if id is not None and models:
        nname = 'group' if name is None else name
        change_model_id(session, models, id, new_name = nname)

def change_model_id(session, models, id, new_name = 'group'):
    '''
    If multiple models or specified id already exists, then make models submodels
    of the specified id, otherwise give this model the specified id.  Missing parent
    models are created.
    '''
    ml = session.models
    p = _find_model(session, id, create = (len(models) > 1), new_name = new_name)
    if p:
        ml.add(models, parent = p)
    else:
        # Find or create parent model of new id.
        p = _find_model(session, id[:-1], create = True, new_name = new_name)
        # Reparent
        ml.assign_id(models[0], id)

# Find parent of model with given id or create a new model or models
# extending up to an existing model.
def _find_model(session, id, create = False, new_name = 'group'):
    if len(id) == 0:
        return None
    ml = session.models
    pl = ml.list(model_id = id)
    if pl:
        p = pl[0]
    elif create:
        from ..models import Model
        p = Model(new_name, session)
        p.id = id
        pp = _find_model(session, id[:-1], True, new_name)
        ml.add([p], parent = pp)
    else:
        p = None

    return p
    
def register_command(session):
    from . import CmdDesc, register, TopModelsArg, StringArg, ModelIdArg
    desc = CmdDesc(required=[('models', TopModelsArg)],
                   optional = [('name', StringArg)],
                   keyword = [('id', ModelIdArg)],
                   synopsis='rename a model or change its id number')
    register('rename', desc, rename, logger=session.logger)
