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
    atom_infos = []
    atoms = structure.atoms
    for a in atoms.filter(atoms.structure_categories == "main"):
        atom_infos.append((a.residue.number, a.residue.chain_id, a.residue.name, a.name, *a.coord, a.radius))
    import numpy
    atom_infos = numpy.asarray(atom_infos)
    if origin is None:
        from pyKVFinder import get_vertices
        vertices = get_vertices(atom_infos, probe_out, step)
    else:
        raise NotImplementedError("Call to get_vertices_from_file() not implemented")
    return atom_infos, vertices
