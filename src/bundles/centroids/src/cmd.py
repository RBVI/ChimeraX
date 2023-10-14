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

from chimerax.atomic import Structure
from chimerax.dist_monitor import SimpleMeasurable

def simplified_string(a, style=None, **kw):
    return a.structure.string(style=style)

# to make it easy to identify centroid models...
class CentroidModel(Structure, SimpleMeasurable):
    @property
    def coord(self):
        return self.atoms[0].coord

    @property
    def radius(self):
        return self.atoms[0].radius

    @property
    def scene_coord(self):
        return self.atoms[0].scene_coord

    def take_snapshot(self, session, flags):
        return { 'base data': super().take_snapshot(session, flags) }

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = cls(session, auto_style=False, log_info=False)
        if 'base data' in data:
            restore_data = data['base data']
        else:
            restore_data = data
        inst.set_state_from_snapshot(session, restore_data)
        a = inst.atoms[0]
        a.string = lambda a=a, **kw: simplified_string(a, **kw)
        return inst

from . import centroid

def cmd_centroid(session, atoms=None, *, mass_weighting=False, name="centroid", color=None, radius=2.0,
        show_tool=True):
    """Wrapper to be called by command line.

       Use chimerax.centroids.centroid for other programming applications.
    """
    from chimerax.core.errors import UserError

    from chimerax.atomic import AtomicStructure, concatenate
    if atoms is None:
        structures_atoms = [m.atoms for m in session.models if isinstance(m, AtomicStructure)]
        if structures_atoms:
            atoms = concatenate(structures_atoms)
    if not atoms:
        raise UserError("Atom specifier selects no atoms")

    structures = atoms.unique_structures
    if len(structures) > 1:
        crds = atoms.scene_coords
    else:
        crds = atoms.coords
    if mass_weighting:
        masses = atoms.elements.masses
        avg_mass = masses.sum() / len(masses)
        import numpy
        weights = masses[:, numpy.newaxis] / avg_mass
    else:
        weights = None
    xyz = centroid(crds, weights=weights)
    s = CentroidModel(session, name=name)
    r = s.new_residue('centroid', 'centroid', 1)
    from chimerax.atomic.struct_edit import add_atom
    a = add_atom(name, 'C', r, xyz)
    if color:
        a.color = color.uint8x4()
    else:
        from chimerax.atomic.colors import element_color, predominant_color
        color = predominant_color(atoms)
        if color is None:
            a.color = element_color(a.element.number)
        else:
            a.color = color
    a.radius = radius
    # override string() to avoid 4(!) "centroid"s in the output
    a.string = lambda a=a, **kw: simplified_string(a, **kw)
    if len(structures) > 1:
        session.models.add([s])
    else:
        structures[0].add([s])

    session.logger.info("Centroid '%s' placed at %s" % (name, xyz))
    if show_tool and session.ui.is_gui and not session.in_script:
        from chimerax.core.commands import run
        run(session, "ui tool show Axes/Planes/Centroids", log=False)
    return a

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, ColorArg, Or, StringArg, EmptyArg, \
        FloatArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required=[('atoms', Or(AtomsArg,EmptyArg))],
        keyword = [('mass_weighting', BoolArg), ('name', StringArg), ('color', ColorArg),
            ('radius', FloatArg), ('show_tool', BoolArg)],
        synopsis = 'Show centroid'
    )
    register('define centroid', desc, cmd_centroid, logger=logger)
