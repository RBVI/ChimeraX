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
    p = _find_model(session, id, create = (len(models) > 1), new_name = new_name)
    if p:
        next_id = max([c.id[-1] for c in p.child_models()], default = 0) + 1
        for m in models:
            _reparent_model(session, m, id + (next_id,), p)
            next_id += 1
    else:
        p = _find_model(session, id[:-1], create = True, new_name = new_name)
        _reparent_model(session, models[0], id, p)

def _reparent_model(session, m, id, p):
    mids = _new_model_ids(m, id, p)
    ml = session.models
    ml.remove([m])
    for model, new_id, parent in mids:
        model.id = new_id
        ml.add([model], parent = parent)

def _new_model_ids(model, id, parent):
    mids = [(model, id, parent)]
    for c in model.child_models():
        new_id = id + (c.id[-1],)
        mids.extend(_new_model_ids(c, new_id, model))
    return mids

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
    register('rename', desc, rename)
