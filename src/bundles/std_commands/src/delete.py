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

def delete(session, atoms, attached_hyds=True):
    '''Delete atoms.

    Parameters
    ----------
    atoms : Atoms collection
        Delete these atoms.  If all atoms of a model are closed then the model is closed.
    '''
    delete_atoms(session, atoms, attached_hyds=attached_hyds)

def delete_atoms(session, atoms, attached_hyds=True):
    '''Delete atoms.

    Parameters
    ----------
    atoms : Atoms collection or None (all atoms)
        Delete these atoms.  If all atoms of a model are closed then the model is closed.
    '''
    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)
    if attached_hyds:
        nbs = atoms.neighbors
        hyds = nbs.filter(nbs.elements.numbers == 1)
        atoms = atoms.merge(hyds)
    atoms.delete()

def delete_bonds(session, bonds):
    '''Delete bonds.

    Parameters
    ----------
    bonds : Bonds collection or None (all bonds)
        Delete these bonds.
    '''
    if bonds is None:
        from chimerax.atomic import all_structures, Structures
        bonds = Structures(all_structures(session)).bonds
    bonds.delete()

def delete_pbonds(session, pbonds, name=None):
    '''Delete pseudobonds.

    Parameters
    ----------
    pbonds : Pseudobonds collection or None (all pseudobonds)
        Delete these pseudobonds.
    name: string
        If specified, restrict deletion to pseudobonds in the named pseudobond group.
    '''
    if pbonds is None:
        from chimerax.atomic import all_pseudobond_groups
        pbonds = all_pseudobond_groups(session).pseudobonds
    if name:
        pbonds = pbonds.filter(pbonds.groups.names == name)
    pbonds.delete()

def register_command(logger):
    from chimerax.core.commands import create_alias, CmdDesc, register, Or, EmptyArg, StringArg, BoolArg
    from chimerax.atomic import AtomsArg, BondsArg, PseudobondsArg
    desc = CmdDesc(required=[('atoms', AtomsArg)],
                       keyword=[('attached_hyds', BoolArg)],
                       synopsis='delete atoms')
    register('delete', desc, delete, logger=logger)
    desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))],
                       keyword=[('attached_hyds', BoolArg)],
                       synopsis='delete atoms')
    register('delete atoms', desc, delete_atoms, logger=logger)
    desc = CmdDesc(required=[('bonds', Or(BondsArg, EmptyArg))],
                       synopsis='delete bonds')
    register('delete bonds', desc, delete_bonds, logger=logger)
    desc = CmdDesc(required=[('pbonds', Or(PseudobondsArg, EmptyArg))],
                       keyword=[('name', StringArg)],
                       synopsis='delete pseudobonds')
    register('delete pbonds', desc, delete_pbonds, logger=logger)
    create_alias('delete pseudobonds', 'delete pbonds $*', logger=logger)
