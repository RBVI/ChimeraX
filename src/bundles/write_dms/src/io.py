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

def write_dms(session, file_name, *, surface=None, displayed_only=True, save_normals=True, status=None):
    """Write a DMS file.

    Parameters
    ----------

    file_name : str, or file object open for writing
        Output file.

    surface : a MolecularSurface model to write out.  If None, then look for open MolecularSurfaces.
        If there is exactly one, write it out -- otherwise raise an error.

    status : function or None
        If not None, a function that takes a string -- used to report the progress of the write.

    displayed_only : bool
        Whether to limit the output to just the displayed part of the surface.

    save_normals : bool
        Whether to also save vector normals in the file.
    """

    if surface is None:
        from chimerax.atomic import MolecularSurface
        surfs = [m for m in session.models is isinstance(m, MolecularSurface)]
        if len(surfs) != 1:
            raise UserError("If molecular surface not specified, there must be exactly one open;"
                " there are %d" % len(surfs))
        surface = surfs[0]

    if status:
        status("Writing DMS file %s; assigning vertex types" % file_name)

    vertex_types = find_vertex_types(session, surface, displayed_only, status)

    from chimerax import io
    f = io.open_output(file_name, "utf-8")


    if file_name != f:
        f.close()

    if status:
        status("Wrote DMS file %s" % file_name)

def find_vertex_types(session, surface, displayed_only, status):
    atoms = surface.atoms
    if status:
        status("Creating search tree of atom centers", secondary=True)
    from chimerax.atom_search = AtomSearchTree
    tree = AtomSearchTree(atoms, scene_coords=(not atoms.single_structure))

    if status:
        status("Finding probe centers", secondary=True)
    max_radius = max(atoms.radii)
    probe_centers = surface.vertices + surface.probe_radius * surface.normals

    if status:
        status("Finding relevant vertices", secondary=True)
    if displayed_only and surface.triangle_mask is not None:
        import numpy
        shown_vertices = set(numpy.unique(surface.triangles[surface.triangle_mask,:]))
    else:
        shown_vertices = set(surface.vertices)

    if status:
        status("", secondary=True)
