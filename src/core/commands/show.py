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

    undo_data = {}
    if 'atoms' in what_to_show:
        show_atoms(session, objects, only, undo_data)
    if 'bonds' in what_to_show:
        show_bonds(session, objects, only, undo_data)
    if 'pseudobonds' in what_to_show or 'pbonds' in what_to_show:
        show_pseudobonds(session, objects, only, undo_data)
    if 'cartoons' in what_to_show or 'ribbons' in what_to_show:
        show_cartoons(session, objects, only, undo_data)
    if 'surfaces' in what_to_show:
        show_surfaces(session, objects, only, undo_data)
    if 'models' in what_to_show:
        show_models(session, objects, only, undo_data)

    def undo(data=undo_data):
        _show_undo(data)
    def redo(data=undo_data):
        _show_redo(data)
    session.undo.register("show", undo, redo)

def show_atoms(session, objects, only, undo_data):
    atoms = objects.atoms
    undo_data['atoms'] = (atoms, atoms.displays, True)
    atoms.displays = True
    if only:
        from ..atomic import structure_atoms
        other_atoms = structure_atoms(atoms.unique_structures) - atoms
        undo_data['atoms_other'] = (other_atoms, other_atoms.displays, False)
        other_atoms.displays = False

def show_bonds(session, objects, only, undo_data):
    bonds = objects.atoms.intra_bonds
    undo_data['bonds'] = (bonds, bonds.displays, True)
    bonds.displays = True
    a1, a2 = bonds.atoms
    undo_data['bonds_atoms1'] = (a1, a1.displays, True)
    undo_data['bonds_atoms2'] = (a2, a2.displays, True)
    a1.displays = True	   # Atoms need to be displayed for bond to appear
    a2.displays = True
    if only:
        mbonds = [m.bonds for m in atoms.unique_structures]
        if mbonds:
            from ..atomic import concatenate
            all_bonds = concatenate(mbonds)
            other_bonds = all_bonds - bonds
            undo_data['bonds_other'] = (other_bonds, other_bonds.displays, False)
            other_bonds.displays = False

def show_pseudobonds(session, objects, only, undo_data):
    atoms = objects.atoms
    from .. import atomic
    pbonds = atomic.interatom_pseudobonds(atoms)
    undo_data['pseudobonds'] = (pbonds, pbonds.displays, True)
    pbonds.displays = True
    a1, a2 = pbonds.atoms
    undo_data['pseudobonds_atoms1'] = (a1, a1.displays, True)
    undo_data['pseudobonds_atoms2'] = (a2, a2.displays, True)
    a1.displays = True	   # Atoms need to be displayed for bond to appear
    a2.displays = True
    if only:
        pbs = sum([[pbg.pseudobonds for pbg in m.pbg_map.values()]
                   for m in atoms.unique_structures], [])
        if pbs:
            from ..atomic import concatenate
            all_pbonds = concatenate(pbs)
            other_pbonds = all_pbonds - pbonds
            undo_data['pseudobonds_other'] = (other_pbonds, other_pbonds.displays, False)
            other_pbonds.displays = False

def show_cartoons(session, objects, only, undo_data):
    atoms = objects.atoms
    res = atoms.unique_residues
    undo_data['cartoons'] = (res, res.ribbon_displays, True)
    res.ribbon_displays = True
    if only:
        from ..atomic import structure_residues
        other_res = structure_residues(atoms.unique_structures) - res
        undo_data['cartoons_other'] = (other_res, other_res.ribbon_displays, False)
        other_res.ribbon_displays = False

def show_surfaces(session, objects, only, undo_data):
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

def show_models(session, objects, only, undo_data):
    from ..models import ancestor_models
    models = objects.models
    minst = objects.model_instances
    ud_positions = {}
    ud_display = {}
    undo_data['models'] = (ud_positions, ud_display)
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
    if only:
        mset = set(models)
        mset.update(ancestor_models(models))
        for m in session.models.list():
            if m not in mset:
                ud_display[m] = [m.display, False]
                m.display = False

def _show_undo(undo_data):
    def _update_attr(key, attr):
        try:
            container, old_values, new_values = undo_data[key]
        except KeyError:
            pass
        else:
            setattr(container, attr, old_values)
    # Atoms
    _update_attr('atoms', 'displays')
    _update_attr('atoms_other', 'displays')
    # Bonds
    _update_attr('bonds', 'displays')
    _update_attr('bonds_atoms1', 'displays')
    _update_attr('bonds_atoms2', 'displays')
    _update_attr('bonds_other', 'displays')
    # Pseudobonds
    _update_attr('pseudobonds', 'displays')
    _update_attr('pseudobonds_atoms1', 'displays')
    _update_attr('pseudobonds_atoms2', 'displays')
    # Cartoons
    _update_attr('cartoons', 'ribbon_displays')
    _update_attr('cartoons_other', 'ribbon_displays')
    # TODO: Surfaces
    # Models
    _update_models(undo_data, 0)

def _update_models(undo_data, which):
    try:
        ud_positions, ud_displays = undo_data['models']
    except KeyError:
        pass
    else:
        for m, v in ud_positions.items():
            m.display_positions = v[which]
        for m, v in ud_display.items():
            m.display = v[which]

def _show_redo(undo_data):
    def _update_attr(key, attr):
        try:
            container, old_values, new_values = undo_data[key]
        except KeyError:
            pass
        else:
            setattr(container, attr, new_values)
    # Atoms
    _update_attr('atoms', 'displays')
    _update_attr('atoms_other', 'displays')
    # Bonds
    _update_attr('bonds', 'displays')
    _update_attr('bonds_atoms1', 'displays')
    _update_attr('bonds_atoms2', 'displays')
    _update_attr('bonds_other', 'displays')
    # Pseudobonds
    _update_attr('pseudobonds', 'displays')
    _update_attr('pseudobonds_atoms1', 'displays')
    _update_attr('pseudobonds_atoms2', 'displays')
    # Cartoons
    _update_attr('cartoons', 'ribbon_displays')
    _update_attr('cartoons_other', 'ribbon_displays')
    # TODO: Surfaces
    # Models
    _update_models(undo_data, 1)

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
