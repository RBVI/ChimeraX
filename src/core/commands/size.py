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

def size(session, objects, atom_radius=None, bond_radius=None):
    '''
    Adjust atom or bond radii.

    Parameters
    ----------
    objects : Objects
      Atoms or bonds whose size should be changed
    atom_radius : float or "default"
      New radius value for atoms.
    bond_radius : float
      New radius value for bonds.
    '''
    what = []
    if atom_radius is not None:
        a = objects.atoms
        if atom_radius == 'default':
            a.radii = a.default_radii
        else:
            a.radii = atom_radius
        what.append('%d atom radii' % len(a))

    if bond_radius is not None:
        a = objects.atoms
        b = a.inter_bonds
        b.radii = bond_radius
        what.append('%d bond radii' % len(b))

    if what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

def register_command(session):
    from . import CmdDesc, register, ObjectsArg, FloatArg, EnumOf, Or
    desc = CmdDesc(
        required = [('objects', ObjectsArg)],
        keyword=[('atom_radius', Or(EnumOf(['default']), FloatArg)),
                 ('bond_radius', FloatArg)],
        synopsis='change atom or bond radii')
    register('size', desc, size)
