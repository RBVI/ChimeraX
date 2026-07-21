# vim: set expandtab ts=4 sw=4:

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
class IterationError(UserError):
    pass

from chimerax.axes_planes.cmd import determine_axes

def align(session, atoms, to=None, *, per_model=False, bring=None):
    """Move atoms to align on axis"""
    from chimerax.core.errors import UserError
    from chimerax.core.commands import Axis
    if not isinstance(to, Axis):
        if len(to) < 2:
            raise UserError("Must specify at least two atoms to form an axis")
        name, center, vec, extent, radius, color = determine_axes(to, "temp", None, 0, 1, False, True,
            False, False, None)[0]
        class FakeAxis:
            def __init__(self, center, vec, extent):
                self.center = center
                self.vec = vec
                self.extent = extent

            def base_point(self):
                return self.center
        to = FakeAxis(center, vec, extent)

    if per_model:
        if bring:
            raise UserError("Cannot specify 'bring' models if 'perModel' is true")
        for s, s_atoms in atoms.by_structure:
            axis_align(session, [s], s_atoms, to)
    else:
        axis_align(session, atoms.unique_structures, atoms, to, bring)

def axis_align(session, structures, atoms, to, bring=None):
    if bring:
        bring = set(bring)
        for s in structures:
            if s in bring:
                bring.discard(s)
            else:
                for b in bring:
                    if b.id == s.id[:len(b.id)]:
                        raise UserError("Cannot 'bring' parent model of structure being aligned")
        if len(bring) == 0:
            session.logger.warning("'bring' arg specifies no models that aren't already being aligned")
            bring = None

    from chimerax.geometry import translation, vector_rotation
    if len(atoms) == 1:
        atom = atoms[0]
        session.logger.info("Moving single atom %s to axis base" % atom)
        for s in structures:
            s.scene_position *= translation(to.base_point() - atom.scene_coord)
        if bring is not None:
            for b in bring:
                b.scene_position *= translation(to.base_point() - atom.scene_coord)
        return
    name, center, vec, extent, radius, color = determine_axes(atoms, "temp", None, 0, 1, False, True,
        False, False, None)[0]
    alignment = translation(to.base_point()) * vector_rotation(vec, to.vec) * translation(-center)
    for s in structures:
        s.scene_position = alignment * s.scene_position
    if bring is not None:
        for b in bring:
            b.scene_position = alignment * b.scene_position

def register_command(logger):

    from chimerax.core.commands import CmdDesc, register, EnumOf, AxisArg, Or, BoolArg, TopModelsArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required = [('atoms', AtomsArg)],
                   keyword = [
                    ('to', Or(AxisArg, AtomsArg)),
                    ('per_model', BoolArg),
                    ('bring', TopModelsArg),
                    ],
                   required_arguments = ['to'],
                   synopsis = 'Align atoms onto axis')
    register('axis align', desc, align, logger=logger)
