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

def show(session, objects=None, what=None, target=None, only=False):
    '''Show specified atoms, bonds or models.

    Parameters
    ----------
    objects : Objects or None
        Atoms, bonds or models to show.  If None then all are shown.
        Objects that are already shown remain shown.
    what : 'atoms', 'bonds', 'pseudobonds', 'pbonds', 'cartoons', 'ribbons', 'surfaces', 'models' or None
        What to show.  If None then 'atoms' if any atoms specified otherwise 'models'.
    target : set of "what" values, or None
        Alternative to the "what" option for specifying what to show.
    only : bool
        Show only the specified atoms/bonds/residues in each specified molecule.
        If what is models then hide models that are not specified.
    '''
    if objects is None:
        from . import atomspec
        objects = atomspec.all_objects(session)

    what_to_show = set() if target is None else set(target)
    if what is not None:
        what_to_show.update([what])
    if len(what_to_show) == 0:
        what_to_show = set(['atoms' if objects.atoms else 'models'])

    from ..undo import UndoState
    undo_state = UndoState("show")
    if 'atoms' in what_to_show:
        show_atoms(session, objects, only, undo_state)
    if 'bonds' in what_to_show:
        show_bonds(session, objects, only, undo_state)
    if 'pseudobonds' in what_to_show or 'pbonds' in what_to_show:
        show_pseudobonds(session, objects, only, undo_state)
    if 'cartoons' in what_to_show or 'ribbons' in what_to_show:
        show_cartoons(session, objects, only, undo_state)
    if 'surfaces' in what_to_show:
        show_surfaces(session, objects, only, undo_state)
    if 'models' in what_to_show:
        show_models(session, objects, only, undo_state)

    session.undo.register(undo_state)

def show_atoms(session, objects, only, undo_state):
    atoms = objects.atoms
    undo_state.add(atoms, "displays", atoms.displays, True)
    atoms.displays = True
    atoms.update_ribbon_visibility()
    if only:
        from ..atomic import structure_atoms
        other_atoms = structure_atoms(atoms.unique_structures) - atoms
        undo_state.add(other_atoms, "displays", other_atoms.displays, False)
        other_atoms.displays = False

def show_bonds(session, objects, only, undo_state):
    bonds = objects.atoms.intra_bonds
    undo_state.add(bonds, "displays", bonds.displays, True)
    bonds.displays = True
    a1, a2 = bonds.atoms
    undo_state.add(a1, "displays", a1.displays, True)
    a1.displays = True	   # Atoms need to be displayed for bond to appear
    undo_state.add(a2, "displays", a2.displays, True)
    a2.displays = True
    if only:
        mbonds = [m.bonds for m in atoms.unique_structures]
        if mbonds:
            from ..atomic import concatenate
            all_bonds = concatenate(mbonds)
            other_bonds = all_bonds - bonds
            undo_state.add(other_bonds, "displays", other_bonds.displays, False)
            other_bonds.displays = False

def show_pseudobonds(session, objects, only, undo_state):
    atoms = objects.atoms
    from .. import atomic
    pbonds = atomic.interatom_pseudobonds(atoms)
    undo_state.add(pbonds, "displays", pbonds.displays, True)
    pbonds.displays = True
    a1, a2 = pbonds.atoms
    undo_state.add(a1, "displays", a1.displays, True)
    a1.displays = True	   # Atoms need to be displayed for bond to appear
    undo_state.add(a2, "displays", a1.displays, True)
    a2.displays = True
    if only:
        pbs = sum([[pbg.pseudobonds for pbg in m.pbg_map.values()]
                   for m in atoms.unique_structures], [])
        if pbs:
            from ..atomic import concatenate
            all_pbonds = concatenate(pbs)
            other_pbonds = all_pbonds - pbonds
            undo_state.add(other_pbonds, "displays", other_pbonds.displays, False)
            other_pbonds.displays = False

def show_cartoons(session, objects, only, undo_state):
    atoms = objects.atoms
    res = atoms.unique_residues
    undo_state.add(res, "ribbon_displays", res.ribbon_displays, True)
    res.ribbon_displays = True
    if only:
        from ..atomic import structure_residues
        other_res = structure_residues(atoms.unique_structures) - res
        undo_state.add(other_res, "ribbon_displays", other_res.ribbon_displays, False)
        other_res.ribbon_displays = False

def show_surfaces(session, objects, only, undo_state):
    # TODO: fill in undo data
    atoms = objects.atoms
    if len(atoms) == 0:
        return

    # Show existing surfaces
    from ..atomic import molsurf, concatenate, Atoms
    surfs = molsurf.show_surface_atom_patches(atoms, session.models, only = only)

    # Create new surfaces if they don't yet exist.
    if surfs:
        patoms, all_small = molsurf.remove_solvent_ligands_ions(atoms)
        extra_atoms = patoms - concatenate([s.atoms for s in surfs], Atoms)
    else:
        extra_atoms = atoms
    if extra_atoms:
        from .surface import surface
        surface(session, extra_atoms)

def show_models(session, objects, only, undo_state):
    from ..models import ancestor_models
    models = objects.models
    minst = objects.model_instances
    ud_positions = {}
    ud_display = {}
    if minst:
        for m,inst in minst.items():
            dp = m.display_positions
            if dp is None or only:
                dp = inst
            else:
                from numpy import logical_or
                logical_or(dp, inst, dp)
            if m in ud:
                ud_positions[m][1] = dp
            else:
                ud_positions[m] = [m.display_positions, dp]
            m.display_positions = dp
        for m in ancestor_models(minst.keys()):
            if m in ud:
                ud_display[m][1] = True
            else:
                ud_display[m] = [m.display, True]
            m.display = True
    else:
        for m in models:
            ud_display[m] = [m.display, True]
            m.display = True
        for m in ancestor_models(models):
            ud_display[m] = [m.display, True]
            m.display = True
    for m, values in ud_positions.items():
        undo_state.add(m, "display_positions", *values)
    for m, values in ud_display.items():
        undo_state.add(m, "display", *values)
    if only:
        mset = set(models)
        mset.update(ancestor_models(models))
        for m in session.models.list():
            if m not in mset:
                undo_state.add(m, "display", m.display, False)
                m.display = False

from . import EnumOf, Annotation
WhatArg = EnumOf(('atoms', 'bonds', 'pseudobonds', 'pbonds', 'cartoons', 'ribbons',
                  'surfaces', 'models'))

class TargetArg(Annotation):
    '''
    Character string indicating what to show or hide,
    a = atoms, b = bonds, p = pseudobonds, c = cartoons, r = cartoons, s = surfaces, m = models.
    '''
    name = 'object type'
    
    @staticmethod
    def parse(text, session):
        from . import StringArg
        token, text, rest = StringArg.parse(text, session)
        target_chars = {'a':'atoms', 'b':'bonds', 'p':'pseudobonds', 'c':'cartoons', 'r':'cartoons',
                        's':'surfaces', 'm':'models'}
        for c in token:
            if c not in target_chars:
                from . import AnnotationError
                raise AnnotationError('Target option can only include letters ' +
                                      ', '.join('%s = %s' % (ch,name) for ch,name in target_chars.items()) +
                                      ', got %s' % c)
        targets = set(target_chars[char] for char in token)
        return targets, text, rest

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, EnumOf, EmptyArg, Or, NoArg, create_alias
    desc = CmdDesc(optional=[('objects', Or(ObjectsArg, EmptyArg)),
                             ('what', WhatArg)],
                   keyword=[('target', TargetArg),
                            ('only', NoArg)],
                   hidden=['only'],
                   synopsis='show specified objects')
    register('show', desc, show, logger=session.logger)
    create_alias('display', 'show $*', logger=session.logger)
