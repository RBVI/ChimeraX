# -----------------------------------------------------------------------------
#
def bond_points_and_colors(bonds, bond_point_spacing):

    bpoints = bond_points(bonds, bond_point_spacing)
    bcolors = bond_point_colors(bonds, bond_point_spacing)

    return bpoints, bcolors

# -----------------------------------------------------------------------------
# Interpolate points along bonds.  Return scene coordinates.
#
def bond_points(bonds, bond_point_spacing):

    xyz_list = []
    for b in bonds:
        c = bond_point_count(b, bond_point_spacing)
        if c > 0:
            xyz1, xyz2 = [a.scene_coord for a in b.atoms]
            for k in range(c):
                fb = float(k+1) / (c+1)
                fa = 1-fb
                xyz = fa*xyz1 + fb*xyz2
                xyz_list.append(xyz)

    from numpy import array, single as floatc, zeros
    if len(xyz_list) > 0:
        points = array(xyz_list, floatc)
    else:
        points = zeros((0,3), floatc)
    
    return points
    
# -----------------------------------------------------------------------------
#
def bond_point_colors(bonds, bond_point_spacing):

    rgba_list = []
    for b in bonds:
        c = bond_point_count(b, bond_point_spacing)
        if c > 0:
            if b.halfbond:
                rgba1, rgba2 = [atom.color for a in b.atoms]
                rgba_list.extend([rgba1]*(c/2))
                rgba_list.extend([rgba2]*(c-c/2))
            else:
                rgba_list.extend([b.color]*c)
    return rgba_list

# -----------------------------------------------------------------------------
#
def bond_point_count(bond, bond_point_spacing):

    from math import floor
    return int(floor(bond.length / bond_point_spacing))

# -----------------------------------------------------------------------------
#
def concatenate_points(points1, points2):

    from numpy import concatenate
    return concatenate((points1, points2))
