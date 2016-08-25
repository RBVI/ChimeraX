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


def style(session, objects, atom_style = None, dashes = None):
    '''Set the atom display style.

    Parameters
    ----------
    atoms : Atoms or None
        Change the style of these atoms. If not specified then all atoms are changed.
    atom_style : "sphere", "ball" or "stick"
        Controls how atoms and bonds are depicted.
    '''
    from ..atomic import all_atoms, Atom
    atoms = all_atoms(session) if objects is None else objects.atoms
    if atom_style is not None:
        s = {
            'sphere': Atom.SPHERE_STYLE,
            'ball': Atom.BALL_STYLE,
            'stick': Atom.STICK_STYLE,
        }[atom_style.lower()]
        atoms.draw_modes = s

    if dashes is not None:
        for pbg in pseudobond_groups(objects, session):
            pbg.dashes = dashes

def pseudobond_groups(objects, session):
    from ..atomic import PseudobondGroup, all_atoms

    # Explicitly specified global pseudobond groups
    models = session.models.list() if objects is None else objects.models
    pbgs = set(m for m in models if isinstance(m, PseudobondGroup))

    atoms = all_atoms(session) if objects is None else objects.atoms

    # Intra-molecular pseudobond groups with bonds between specified atoms.
    for m in atoms.unique_structures:
        molpbgs = [pbg for pbg in m.pbg_map.values()
                   if pbg.pseudobonds.between_atoms(atoms).any()]
        pbgs.update(molpbgs)

    # Global pseudobond groups with bonds between specified atoms
    gpbgs = [pbg for pbg in session.models.list(type = PseudobondGroup)
             if pbg.pseudobonds.between_atoms(atoms).any()]
    pbgs.update(gpbgs)

    return pbgs


def register_command(session):
    from . import register, CmdDesc, ObjectsArg, EmptyArg, EnumOf, Or, IntArg
    desc = CmdDesc(required = [("objects", Or(ObjectsArg, EmptyArg))],
                   optional = [('atom_style', EnumOf(('sphere', 'ball', 'stick'))),
                               ('dashes', IntArg)],
                   synopsis='change atom and bond depiction')
    register('style', desc, style)
