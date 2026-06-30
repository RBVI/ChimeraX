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
from .settings import defaults

def write_dms(session, file_name, *, surface=None, displayed_only=defaults['displayed_only'],
        save_normals=defaults['save_normals'], status=None):
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
        surfs = [m for m in session.models if isinstance(m, MolecularSurface)]
        if len(surfs) != 1:
            raise UserError("If molecular surface not specified, there must be exactly one open;"
                " there are %d" % len(surfs))
        surface = surfs[0]

    if status:
        status("Writing DMS file %s; assigning vertex types" % file_name)

    vertex_info = find_vertex_info(session, surface, displayed_only, status)

    if status:
        status("Writing DMS file %s to disk" % file_name)

    from chimerax import io
    with io.open_output(file_name, "utf-8") as stream:
        for a, a_crd in zip(surface.atoms, surface.atom_coords()):
            atom_part = atom_format(a)
            print("%s%8.3f %8.3f %8.3f A" % (atom_part, *a_crd), file=stream)

            if a not in vertex_info:
                continue
            for vertex, normal, area, num_contacts in vertex_info[a]:
                dms_type = 'C' if num_contacts == 1 else ('S' if num_contacts == 2 else 'R')
                line = "%s%8.3f %8.3f %8.3f S%s0 %6.3f" % (atom_part, *vertex, dms_type, area)
                if save_normals:
                    line += " %6.3f %6.3f %6.3f" % (*normal,)
                print(line, file=stream)

    if status:
        status("Wrote DMS file %s" % file_name)

def find_vertex_info(session, surface, displayed_only, status):
    atoms = surface.atoms
    if status:
        status("Creating search tree of atom centers", secondary=True)
    from chimerax.atom_search import AtomSearchTree
    single_structure = atoms.single_structure
    tree = AtomSearchTree(atoms, scene_coords=(not single_structure))

    if status:
        status("Finding probe centers", secondary=True)
    max_radius = max(atoms.radii)
    probe_radius = surface.probe_radius
    probe_centers = surface.vertices + probe_radius * surface.normals

    if status:
        status("Finding relevant vertices", secondary=True)
    if displayed_only and surface.triangle_mask is not None:
        import numpy
        shown_vertex_indices = set(numpy.unique(surface.triangles[surface.triangle_mask,:]))
    else:
        shown_vertex_indices = set(range(len(surface.vertices)))

    if status:
        status("Getting vertex/normal coordinates", secondary=True)
    atom_to_vinfo = {}
    if single_structure:
        vertices = surface.vertices
        normals = surface.normals
        coord_func = lambda a: a.coord
    else:
        vertices = surface.scene_position.transform_points(surface.vertices)
        normals = surface.scene_position.transform_points(surface.normals)
        coord_func = lambda a: a.scene_coord

    if status:
        status("Finding vertex surface areas", secondary=True)
    from chimerax.surface import vertex_areas
    areas = vertex_areas(surface.vertices, surface.triangles)

    if status:
        status("Finding contacting atoms", secondary=True)
    from math import sqrt
    search_cutoff = max_radius + probe_radius + surface.grid_spacing
    # distance from center of grid cube to corner is sqrt(3/4) * grid-spacing
    grid_chk = sqrt(0.75) * surface.grid_spacing
    from chimerax.geometry import distance
    for i, v_n_a in enumerate(zip(vertices, normals, areas)):
        if status and i % 10000 == 0:
            status("Finding contacting atoms (%.1f%%)" % (100 * i / len(vertices)), secondary=True)
        if i not  in shown_vertex_indices:
            continue
        vertex, normal, area = v_n_a
        probe_center = vertex + probe_radius * normal
        nearby = tree.search(probe_center, search_cutoff)
        nearest = None
        contacts = []
        for a in nearby:
            a_crd = coord_func(a)
            d = distance(a_crd, probe_center)
            delta = d - a.radius - probe_radius
            if delta > grid_chk:
                # atom not in contact
                continue
            contacts.append(a)
            if nearest is None or delta < nearest_delta:
                nearest = a
                nearest_delta = delta
        if not nearest:
            raise ValueError("No surface atom near surface vertex?!?")
        atom_to_vinfo.setdefault(nearest, []).append((vertex, normal, area, len(contacts)))

    if status:
        status("", secondary=True)
    return atom_to_vinfo

def atom_format(a):
    r = a.residue
    if r.insertion_code == ' ':
        insert = ""
    else:
        insert = r.insertion_code
    if len(r.chain_id) > 1:
        chain_id = '*'
    else:
        chain_id = r.chain_id
    res_seq = str(r.number) + insert + chain_id
    return "%3s %4s %4.4s" % (r.name, res_seq, a.name)
