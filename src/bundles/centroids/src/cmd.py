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

from . import centroid

def cmd_centroid(session, atoms=None, *, mass_weighting=False, name="centroid", color=None, radius=2.0):
    """Wrapper to be called by command line.

       Use chimerax.centroids.centroid for other programming applications.
    """
    from chimerax.core.errors import UserError

    from chimerax.atomic import AtomicStructure, concatenate, Structure
    if atoms is None:
        structures_atoms = [m.atoms for m in session.models if isinstance(m, AtomicStructure)]
        if structures_atoms:
            atoms = concatenate(structures_atoms)
        else:
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
    s = Structure(session, name=name)
    r = s.new_residue('centroid', 'centroid', 1)
    from chimerax.atomic.struct_edit import add_atom
    a = add_atom('cent', 'C', r, xyz)
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
    if len(structures) > 1:
        session.models.add([s])
    else:
        structures[0].add([s])

    session.logger.info("Centroid '%s' placed at %s" % (name, xyz))
    return a

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, ColorArg, Or, StringArg, EmptyArg, \
        FloatArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required=[('atoms', Or(AtomsArg,EmptyArg))],
        keyword = [('mass_weighting', BoolArg), ('name', StringArg), ('color', ColorArg),
            ('radius', FloatArg)],
        synopsis = 'Show centroid'
    )
    register('define centroid', desc, cmd_centroid, logger=logger)
