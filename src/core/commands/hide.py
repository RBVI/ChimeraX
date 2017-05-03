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

def hide(session, objects=None, what=None, target=None):
    '''Hide specified atoms, bonds or models.

    Parameters
    ----------
    objects : Objects or None
        Atoms, bonds or models to hide. If None then all are hidden.
    what : 'atoms', 'bonds', 'pseudobonds', 'pbonds', 'cartoons', 'ribbons', 'models' or None
        What to hide.  If None then 'atoms' if any atoms specified otherwise 'models'.
    target : set of "what" values, or None
        Alternative to the "what" option for specifying what to hide.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)

    what_to_hide = set() if target is None else set(target)
    if what is not None:
        what_to_hide.update([what])
    if len(what_to_hide) == 0:
        what_to_hide = set(['atoms' if objects.atoms else 'models'])

    if 'atoms' in what_to_hide:
        objects.atoms.displays = False
    if 'bonds' in what_to_hide:
        atoms = objects.atoms
        atoms.intra_bonds.displays = False
    if 'pseudobonds' in what_to_hide or 'pbonds' in what_to_hide:
        from .. import atomic
        pbonds = atomic.interatom_pseudobonds(objects.atoms)
        pbonds.displays = False
    if 'cartoons' in what_to_hide or 'ribbons' in what_to_hide:
        res = objects.atoms.unique_residues
        res.ribbon_displays = False
    if 'surfaces' in what_to_hide:
        from ..atomic import molsurf
        molsurf.hide_surface_atom_patches(objects.atoms, session.models)
    if 'models' in what_to_hide:
        hide_models(objects)

def hide_models(objects):
    minst = objects.model_instances
    if minst:
        from numpy import logical_and, logical_not
        for m,inst in minst.items():
            dp = m.display_positions
            ninst = logical_not(inst)
            if dp is None:
                dp = ninst
            else:
                logical_and(dp, ninst, dp)
            m.display_positions = dp
    else:
        for m in objects.models:
            m.display = False

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, EnumOf, EmptyArg, Or, create_alias
    from .show import WhatArg, TargetArg
    desc = CmdDesc(
        optional=[('objects', Or(ObjectsArg, EmptyArg)),
                  ('what', WhatArg)],
        keyword=[('target', TargetArg)],
        url='help:user/commands/show.html#hide',
        synopsis='hide specified objects')
    register('hide', desc, hide, logger=session.logger)
    create_alias('~show', 'hide $*', logger=session.logger)
    create_alias('~display', 'hide $*', logger=session.logger)
