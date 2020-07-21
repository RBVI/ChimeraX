..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

segment: Segmentations of density maps
======================================

Routines for calculating properties of index map segmentations.

Index map segmentations are 3D numpy arrays of uint32 values where the value
indicates which region the grid point belongs to.  Region index values start at 1,
the value 0 indicates the grid point does not belong to any region.  Corresponding
image data 3d arrays can have any scalar type supported by numpy (int8, uint8, int16,
uint16, int32, uint32, int64, uint64, float32, float64).

.. automodule:: chimerax.segment._segment
    :members: watershed_regions, region_index_lists, region_contacts, region_bounds, region_point_count, region_points, region_maxima, interface_values, find_local_maxima, crosssection_midpoints, segmentation_surface, segmentation_surfaces
