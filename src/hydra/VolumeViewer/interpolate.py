#
# Example code to get interpolated volume data set values at specific points.
#
# Uses trilinear interpolation from 8 nearest grid points.  Value is zero
# if point lies outside volume.  Points are specified in physical coordinates
# (typically Angstroms) using the voxel size and origin.
#
# Results are printed to Reply Log (under Favorites menu).
#
from VolumeViewer import volume_list
v = volume_list()[0]

points = [(10.243, 12.534, 25.352),
          (54.2112, 32.352, 29.3525),
          (-15.153, 139.23, 12.44)]
values = v.interpolated_values(points)
print values
