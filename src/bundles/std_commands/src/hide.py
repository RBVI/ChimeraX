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
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from .show import what_objects
    what_to_hide = what_objects(target, what, objects)

    from chimerax.core.undo import UndoState
    undo_state = UndoState("hide")
    if 'atoms' in what_to_hide:
        atoms = objects.atoms
        undo_state.add(atoms, "displays", atoms.displays, False)
        atoms.displays = False
        atoms.update_ribbon_backbone_atom_visibility()
    if 'bonds' in what_to_hide:
        bonds = objects.bonds
        undo_state.add(bonds, "displays", bonds.displays, False)
        bonds.displays = False
    if 'pseudobonds' in what_to_hide or 'pbonds' in what_to_hide:
        from chimerax import atomic
        pbonds = objects.pseudobonds
        undo_state.add(pbonds, "displays", pbonds.displays, False)
        pbonds.displays = False
    if 'cartoons' in what_to_hide or 'ribbons' in what_to_hide:
        res = objects.residues
        undo_state.add(res, "ribbon_displays", res.ribbon_displays, False)
        res.ribbon_displays = False
    if 'surfaces' in what_to_hide:
        from chimerax.atomic import molsurf
        # TODO: save undo data
        molsurf.hide_surface_atom_patches(objects.atoms)
    if 'models' in what_to_hide:
        hide_models(objects, undo_state)

    session.undo.register(undo_state)

def hide_models(objects, undo_state):
    models, instances = _models_and_instances(objects)
    ud_positions = {}
    ud_display = {}

    # Hide model instances
    from numpy import logical_and, logical_not
    for m,inst in instances.items():
        dp = m.display_positions
        ninst = logical_not(inst)
        if dp is None:
            dp = ninst
        else:
            logical_and(dp, ninst, dp)
        if m in ud_positions:
            ud_positions[m][1] = dp
        else:
            ud_positions[m] = [m.display_positions, dp]
        m.display_positions = dp

    # Hide models
    for m in models:
        if m in ud_display:
            ud_display[m][1] = False
        else:
            ud_display[m] = [m.display, True]
        m.display = False

    # Record undo state
    for m, values in ud_positions.items():
        undo_state.add(m, "display_positions", *values)
    for m, values in ud_display.items():
        undo_state.add(m, "display", *values)

def _models_and_instances(objects):
    models = set(objects.models)
    instances = dict()
    for m,inst in objects.model_instances.items():
        ni = inst.sum()
        np = len(m.positions)
        if ni > 0 and ni < np:
            instances[m] = inst
        elif ni == np:
            models.add(m)
    return models, instances

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ObjectsArg, EnumOf, EmptyArg, Or, \
        create_alias
    from .show import WhatArg, TargetArg
    desc = CmdDesc(
        optional=[('objects', Or(ObjectsArg, EmptyArg)),
                  ('what', WhatArg)],
        keyword=[('target', TargetArg)],
        url='help:user/commands/show.html#hide',
        synopsis='hide specified objects')
    register('hide', desc, hide, logger=logger)
    create_alias('~show', 'hide $*', logger=logger)
    create_alias('~display', 'hide $*', logger=logger,
            url="help:user/commands/show.html")
