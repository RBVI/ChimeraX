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

# -----------------------------------------------------------------------------
#
def zone_operation(v, atoms, radius, bond_point_spacing = None,
                   minimal_bounds = False, new_map = True, invert = False,
                   subregion = 'all', step = 1, model_id = None):

    points = atoms.scene_coords
    if bond_point_spacing is not None:
        bonds = atoms.intra_bonds
        from chimerax.atomic.path import bond_points
        bpoints = bond_points(bonds, bond_point_spacing)
        from numpy import concatenate
        points = concatenate((points, bpoints))

    v.position.inverse().transform_points(points, in_place = True)   # Convert points to map coordinates.

    if new_map:
        vz = zone_volume(v, points, radius, minimal_bounds, invert,
                         subregion, step, model_id)
    else:
        r = _bounding_region(points, radius, v.data) if minimal_bounds else v.full_region()
        v.new_region(r[0], r[1])
        from chimerax.surface.zone import surface_zone
        for s in v.surfaces:
            surface_zone(s, points, radius, auto_update=True)
        vz = v
    return vz

# -----------------------------------------------------------------------------
#
def _bounding_region(points, radius, grid_data):
    from chimerax.map_data.regions import points_ijk_bounds, clamp_region, integer_region
    r = points_ijk_bounds(points, radius, grid_data)
    r = clamp_region(integer_region(r), grid_data.size)
    return r

# -----------------------------------------------------------------------------
#
def zone_volume(volume, points, radius,
                minimal_bounds = False, invert = False,
                subregion = 'all', step = 1, model_id = None):

    region = volume.subregion(step, subregion)
    from chimerax import map_data
    sg = map_data.GridSubregion(volume.data, *region)

    mg = map_data.zone_masked_grid_data(sg, points, radius, invert, minimal_bounds)
    mg.name = volume.name + ' zone'

    from chimerax.map import volume_from_grid_data
    vz = volume_from_grid_data(mg, volume.session, model_id = model_id)
    vz.copy_settings_from(volume, copy_colors = False, copy_zone = False)
    volume.display = False

    return vz
