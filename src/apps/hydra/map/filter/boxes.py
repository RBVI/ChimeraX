# ------------------------------------------------------------------------------
# Make separate volumes of a specified size around markers.
#
def boxes(volume, atoms, size = 0, use_atom_size = False,
          step = None, subregion = None, base_model_id = None):

    vlist = []
    vxfinv = volume.openState.xform.inverse()
    ijk_rmin, ijk_rmax = volume.ijk_bounds(step, subregion, integer = True)
    for i,a in enumerate(atoms):
        center = vxfinv.apply(a.xformCoord()).data()
        r = 0.5*size
        if use_atom_size:
            r += a.radius
        ijk_min, ijk_max, ijk_step = volume.bounding_region([center], r, step)
        ijk_min = [max(s,t) for s,t in zip(ijk_min, ijk_rmin)]
        ijk_max = [min(s,t) for s,t in zip(ijk_max, ijk_rmax)]
        region = (ijk_min, ijk_max, ijk_step)
        from VolumeViewer.volume import is_empty_region
        if is_empty_region(region):
            continue
        from VolumeData import Grid_Subregion
        g = Grid_Subregion(volume.data, *region)
        g.name = 'box'
        if base_model_id is None:
            mid = None
        else:
            mid = base_model_id[0] + i
        from VolumeViewer import volume_from_grid_data
        v = volume_from_grid_data(g, model_id = mid, show_data = False,
                                  show_dialog = False)
        v.copy_settings_from(volume, copy_region = False,
                             copy_active = False, copy_zone = False)
        v.show()
        vlist.append(v)
    return vlist
