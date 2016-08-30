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
        
    if name is not None:
        for m in models:
            m.name = name

    if id is not None and models:
        if len(models) > 1:
            from ..errors import UserError
            raise UserError('Cannot change multiple models (%d) to have same id'
                            % len(models))
        change_model_id(models[0], id)

def change_model_id(m, id):

    # Check if model id already exists.
    ml = m.session.models
    mid = ml.list(model_id = id)
    if len(mid) > 0:
        from ..errors import UserError
        raise UserError('An existing model already has id #%s'
                        % '.'.join(str(i) for i in id))

    # Record new ids and parents for model and children since remove sets them all to None.
    mids = _new_model_ids(m, id, _parent_model(m.session, id, m.name))
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
def _parent_model(session, id, new_name):
    if len(id) == 1:
        return None
    ml = session.models
    pl = ml.list(model_id = id[:-1])
    if pl:
        p = pl[0]
    else:
        from ..models import Model
        p = Model(new_name, session)
        p.id = id[:-1]
        pp = _parent_model(session, p.id, new_name)
        ml.add([p], parent = pp)

    return p
    
def register_command(session):
    from . import CmdDesc, register, TopModelsArg, StringArg, ModelIdArg
    desc = CmdDesc(required=[('models', TopModelsArg)],
                   optional = [('name', StringArg)],
                   keyword = [('id', ModelIdArg)],
                   synopsis='rename a model or change its id number')
    register('rename', desc, rename)
