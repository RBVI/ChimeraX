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

def prep_input(structure, origin, extent, probe_in, probe_out, step):
# Get atomic information of the target structure
    atom_infos = []
    atoms = structure.atoms
    for a in atoms.filter(atoms.structure_categories == "main"):
        atom_infos.append(
            (
                a.residue.number,
                a.residue.chain_id,
                a.residue.name,
                a.name,
                *a.coord,
                a.radius,
            )
        )
    import numpy

    atom_infos = numpy.asarray(atom_infos)

    # Whole protein mode
    if origin is None:
        from pyKVFinder import get_vertices

        vertices = get_vertices(atom_infos, probe_out, step)
    # Box adjustment mode
    else:
        if isinstance(extent, float):
            extents = [extent, extent, extent]
        else:
            extents = extent

        # Create box
        box_info = {
            "p1": list(origin),  # origin
            "p2": [origin[0] + extents[0], origin[1], origin[2]],  # x-axis
            "p3": [origin[0], origin[1] + extents[1], origin[2]],  # y-axis
            "p4": [origin[0], origin[1], origin[2] + extents[2]],  # z-axis
        }
        #NOTE: for reference when implementing this as an option
        '''
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("wt", suffix=".bild") as bild_f:
            print(".color .7 .7 .7", file=bild_f)
            print(".transparency .5", file=bild_f)
            box_corners = tuple(origin) + tuple(origin[i]+extents[i] for i in range(3))
            print(".box %f %f %f %f %f %f" % box_corners, file=bild_f)
            bild_f.flush()
            from chimerax.core.commands import run, StringArg
            run(structure.session, "open %s id %s"% (StringArg.unparse(bild_f.name), structure.atomspec + ".9"))
        '''

        from pyKVFinder.grid import _get_vertices_from_box, _get_sincos, _get_dimensions
        from _pyKVFinder import _filter_pdb
        import os

        # Extract xyzr from atomic
        xyzr = atom_infos[:, 4:].astype(numpy.float64)

        # Get vertices from box and select atoms
        vertices = _get_vertices_from_box(box_info, probe_out)

        # Get sincos
        sincos = numpy.round(_get_sincos(vertices), 4)

        # Get dimensions
        nx, ny, nz = _get_dimensions(vertices, step)

        # Get atoms inside box only
        _filter_pdb(
            nx,
            ny,
            nz,
            xyzr,
            vertices[0],
            sincos,
            step,
            probe_in,
            os.cpu_count() - 1,
        )

        # Get indexes of the atoms inside the box
        indexes = xyzr[:, 3] != 0

        # Slice atominfo and xyzr
        atom_infos = atom_infos[indexes, :]

    return atom_infos, vertices
