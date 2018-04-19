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

def delete(session, atoms):
    '''Delete atoms.

    Parameters
    ----------
    atoms : Atoms collection
        Delete these atoms.  If all atoms of a model are closed then the model is closed.
    '''
    delete_atoms(session, atoms)

def delete_atoms(session, atoms):
    '''Delete atoms.

    Parameters
    ----------
    atoms : Atoms collection or None (all atoms)
        Delete these atoms.  If all atoms of a model are closed then the model is closed.
    '''
    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)
    atoms.delete()
    session.models.close([s for s in session.models if s.deleted])

def delete_bonds(session, bonds):
    '''Delete bonds.

    Parameters
    ----------
    bonds : Bonds collection or None (all bonds)
        Delete these bonds.
    '''
    if bonds is None:
        from chimerax.atomic import all_structures
        bonds = all_structures(session).bonds
    bonds.delete()

def delete_pbonds(session, pbonds, name=None):
    '''Delete pseudobonds.

    Parameters
    ----------
    pbonds : Pseudobonds collection or None (all pseudobonds)
        Delete these pseudobonds.
    category: string
        If specified, restrict deletion to pseudobonds matching the category.
    '''
    if pbonds is None:
        from chimerax.atomic import all_pseudobond_groups
        pbonds = all_pseudobond_groups(session).pseudobonds
    if name:
        pbonds = pbonds.filter(pbonds.groups.names == name)
    pbonds.delete()

def register_command(session):
    from . import cli, create_alias
    desc = cli.CmdDesc(required=[('atoms', cli.AtomsArg)],
                       synopsis='delete atoms')
    cli.register('delete', desc, delete, logger=session.logger)
    desc = cli.CmdDesc(required=[('atoms', cli.Or(cli.AtomsArg, cli.EmptyArg))],
                       synopsis='delete atoms')
    cli.register('delete atoms', desc, delete_atoms, logger=session.logger)
    desc = cli.CmdDesc(required=[('bonds', cli.Or(cli.BondsArg, cli.EmptyArg))],
                       synopsis='delete bonds')
    cli.register('delete bonds', desc, delete_bonds, logger=session.logger)
    desc = cli.CmdDesc(required=[('pbonds', cli.Or(cli.PseudobondsArg, cli.EmptyArg))],
                       keyword=[('name', cli.StringArg)],
                       synopsis='delete pseudobonds')
    cli.register('delete pbonds', desc, delete_pbonds, logger=session.logger)
    create_alias('delete pseudobonds', 'delete pbonds $*', logger=session.logger)
