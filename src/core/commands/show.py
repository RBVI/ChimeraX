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

    if 'atoms' in what_to_show:
        show_atoms(session, objects, only)
    if 'bonds' in what_to_show:
        show_bonds(session, objects, only)
    if 'pseudobonds' in what_to_show or 'pbonds' in what_to_show:
        show_pseudobonds(session, objects, only)
    if 'cartoons' in what_to_show or 'ribbons' in what_to_show:
        show_cartoons(session, objects, only)
    if 'surfaces' in what_to_show:
        show_surfaces(session, objects, only)
    if 'models' in what_to_show:
        show_models(session, objects, only)

def show_atoms(session, objects, only):
    atoms = objects.atoms
    atoms.displays = True
    if only:
        from ..atomic import structure_atoms
        other_atoms = structure_atoms(atoms.unique_structures) - atoms
        other_atoms.displays = False

def show_bonds(session, objects, only):
    bonds = objects.atoms.intra_bonds
    bonds.displays = True
    a1, a2 = bonds.atoms
    a1.displays = True	   # Atoms need to be displayed for bond to appear
    a2.displays = True
    if only:
        mbonds = [m.bonds for m in atoms.unique_structures]
        if mbonds:
            from ..atomic import concatenate
            all_bonds = concatenate(mbonds)
            other_bonds = all_bonds - bonds
            other_bonds.displays = False

def show_pseudobonds(session, objects, only):
    atoms = objects.atoms
    from .. import atomic
    pbonds = atomic.interatom_pseudobonds(atoms)
    pbonds.displays = True
    a1, a2 = pbonds.atoms
    a1.displays = True	   # Atoms need to be displayed for bond to appear
    a2.displays = True
    if only:
        pbs = sum([[pbg.pseudobonds for pbg in m.pbg_map.values()]
                   for m in atoms.unique_structures], [])
        if pbs:
            from ..atomic import concatenate
            all_pbonds = concatenate(pbs)
            other_pbonds = all_pbonds - pbonds
            other_pbonds.displays = False

def show_cartoons(session, objects, only):
    atoms = objects.atoms
    res = atoms.unique_residues
    res.ribbon_displays = True
    if only:
        from ..atomic import structure_residues
        other_res = structure_residues(atoms.unique_structures) - res
        other_res.ribbon_displays = False

def show_surfaces(session, objects, only):
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

def show_models(session, objects, only):
    from ..models import ancestor_models
    models = objects.models
    minst = objects.model_instances
    if minst:
        for m,inst in minst.items():
            dp = m.display_positions
            if dp is None or only:
                dp = inst
            else:
                from numpy import logical_or
                logical_or(dp, inst, dp)
            m.display_positions = dp
        for m in ancestor_models(minst.keys()):
            m.display = True
    else:
        for m in models:
            m.display = True
        for m in ancestor_models(models):
            m.display = True
    if only:
        mset = set(models)
        mset.update(ancestor_models(models))
        for m in session.models.list():
            if m not in mset:
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
