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
        from chimerax.core.errors import UserError
        raise UserError('No name or id option specified for renaming')
        
    if name is not None and (id is None or len(models) == 1):
        for m in models:
            m.name = name

    if id is not None and models:
        _prevent_overlay_rename(models, id)
        nname = 'group' if name is None else name
        change_model_id(session, models, id, new_name = nname)

def change_model_id(session, models, id, new_name = 'group'):
    '''
    If multiple models or specified id already exists, then make models submodels
    of the specified id, otherwise give this model the specified id.  Missing parent
    models are created.
    '''
    from chimerax.core.models import MODEL_ID_CHANGED
    with session.triggers.block_trigger(MODEL_ID_CHANGED):
        ml = session.models
        # If id we are changing to is one of the models being moved
        # or a child of one of the models being moved then
        # change that model's id so a new group model can be made.
        for m in models:
            if m.id == id[:len(m.id)]:
                temp_id = (12345678,)
                ml.assign_id(m, temp_id)
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
        from chimerax.core.models import Model
        p = Model(new_name, session)
        p.id = id
        pp = _find_model(session, id[:-1], True, new_name)
        ml.add([p], parent = pp)
    else:
        p = None

    return p

def _prevent_overlay_rename(models, id):
    overlays = [m for m in models if not _is_scene_model(m)]
    if overlays:
        oids = ', '.join(str(m) for m in overlays)
        from chimerax.core.errors import UserError
        raise UserError('Cannot change id of 2D overlay models (%s)' % oids)
    if models:
        session = models[0].session
        if [m for m in session.models.list(model_id = id[:1]) if not _is_scene_model(m)]:
            did = '.'.join(str(i) for i in id)
            from chimerax.core.errors import UserError
            raise UserError('Cannot place models under an overlay models (#%s)' % did)

def _is_scene_model(model):
    if model is None:
        return False
    if model is model.session.models.scene_root_model:
        return True
    return _is_scene_model(model.parent)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, TopModelsArg, StringArg, ModelIdArg
    desc = CmdDesc(required=[('models', TopModelsArg)],
                   optional = [('name', StringArg)],
                   keyword = [('id', ModelIdArg)],
                   synopsis='rename a model or change its id number')
    register('rename', desc, rename, logger=logger)
