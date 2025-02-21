# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.errors import UserError

def check_atoms(session, atoms):
    from chimerax.atomic import all_atoms
    if atoms is None:
        atoms = all_atoms(session)

    if not atoms:
        raise UserError("No atoms specified")

    atoms = atoms.filter(atoms.has_aniso_u)
    if not atoms:
        raise UserError("None of the specified atoms have anisotropic temperature factors")

    return atoms

def aniso_show(session, atoms=None):
    ''' Command to display thermal ellipsoids '''

    atoms = check_atoms(session, atoms)

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAnisoManager(session, s)
        mgr_info[s].show(atoms=s_atoms)

def aniso_style(session, structures=None, **kw):
    ''' Command to display thermal ellipsoids '''

    if structures is None:
        from chimerax.atomic import all_atomic_structures
        atoms = all_atomic_structures(session).atoms
    else:
        atoms = structures.atoms
    atoms = check_atoms(session, atoms)

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAnisoManager(session, s)
        mgr_info[s].style(**kw)

def aniso_hide(session, atoms=None):
    ''' Command to hide thermal ellipsoids '''

    atoms = check_atoms(session, atoms)

    from .mgr import _StructureAnisoManager
    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            continue
        mgr_info[s].hide(atoms=s_atoms)

def register_command(logger, name):
    from chimerax.core.commands import register, CmdDesc
    from chimerax.core.commands import Or, EmptyArg, Color8TupleArg, NoneArg, PositiveFloatArg, BoolArg
    from chimerax.core.commands import PositiveIntArg, Bounded, FloatArg
    from chimerax.atomic import AtomsArg, AtomicStructuresArg

    tilde_desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))], synopsis='hide thermal ellipsoids')
    if name == "aniso":
        show_desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))],
            synopsis='show depictions of thermal ellipsoids')
        style_desc = CmdDesc(required=[('structures', Or(AtomicStructuresArg, EmptyArg))],
            keyword=[
                ('axis_color', Or(Color8TupleArg, NoneArg)),
                ('axis_factor', Or(PositiveFloatArg, NoneArg)),
                ('axis_thickness', PositiveFloatArg),
                ('ellipse_color', Or(Color8TupleArg, NoneArg)),
                ('ellipse_factor', Or(PositiveFloatArg, NoneArg)),
                ('ellipse_thickness', PositiveFloatArg),
                ('ellipsoid_color', Or(Color8TupleArg, NoneArg)),
                ('scale', PositiveFloatArg),
                ('show_ellipsoid', BoolArg),
                ('smoothing', PositiveIntArg),
                ('transparency', Bounded(FloatArg, min=0, max=100,
                    name="a percentage (number between 0 and 100)")),
            ],
            synopsis='change style of depictions of thermal ellipsoids')
        register('aniso', show_desc, aniso_show, logger=logger)
        register('aniso style', style_desc, aniso_style, logger=logger)
        register('aniso hide', tilde_desc, aniso_hide, logger=logger)
    else:
        register('~aniso', tilde_desc, aniso_hide, logger=logger)
