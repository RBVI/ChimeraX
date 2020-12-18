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

map: Density maps
=================

3D image data from electron microscopy, X-ray crystallography, light microscopy, and
medical imaging, and segmentations can be read from dozens of file formats, saved to
a few formats (e.g. MRC or HDF5),  rendered as transparent 3D image stacks or contour surfaces,
filtered (e,g. smoothing), fit with atomic models, annotated with markers, measured,
segmented.

Classes
-------

 * :class:`.Volume` - model for drawing 3d images.
 * :class:`.VolumeSurface` - calculates and renders contour surfaces, child model of Volume.
 * :class:`.VolumeImage` - volumetric image rendering, child model of Volume.
 * :class:`.RenderingOptions` - settings for surface and image renderings.
 * :class:`~chimerax.map_data.GridData` - 3d image data, does not render it.
 * :class:`~chimerax.map_data.ArrayGridData` - 3d image data created from a numpy array.

Read, Write, Create
-------------------

 * :func:`.open_map` - read 3d image data from a file
 * :func:`.save_map` - write 3d image data to a file
 * :func:`.volume_from_grid_data` - create a :class:`.Volume` from image data.

Operations
----------

These operations create new :class:`.Volume` models from existing ones and
implement the Chimera volume subcommands (e.g. "volume resample").

 * :func:`~.volume_add` - add two or more volumes
 * :func:`~.volume_bin` - reduce map size by averaging over bins
 * :func:`~.volume_boxes` - extract subvolumes centered on markers
 * :func:`~.volume_copy` - copy a volume
 * :func:`~.volume_cover` - use symmetry to extend a volume
 * :func:`~.volume_erase` - erase volume inside or outside a sphere
 * :func:`~.volume_falloff` - smooth the falloff of a masked map at its boundary
 * :func:`~.volume_flatten` - scale values to flatten map baseline
 * :func:`~.volume_flip` - reverse data planes along a specified axis
 * :func:`~.volume_fourier` - Fourier transform
 * :func:`~.volume_gaussian` - Gaussian smoothing or sharpening
 * :func:`~.volume_laplacian` - Laplacian filtering
 * :func:`~.volume_local_correlation` - calculate map-map correlation over a sliding box
 * :func:`~.mask` - mask a map to a surface; see also :func:`~.ones_mask`
 * :func:`~.volume_maximum` - take the maximum values pointwise from two or more maps
 * :func:`~.volume_median` - set each value to the median of values in a surrounding box
 * :func:`~.volume_minimum` - take the minimum values pointwise from two or more maps
 * :func:`~.volume_morph` - morph (interpolate) between two or more maps
 * :func:`~.volume_multiply` - multiply values in two or more maps
 * :func:`~.volume_new` - create empty (zero-valued) map; see also volume onesmask
 * :func:`~.volume_octant` - erase all but the positive octant
 * :func:`~.ones_mask` - create map with values of 1 bounded by a surface
 * :func:`~.volume_permute_axes` - permute axes
 * :func:`~.volume_resample` - resample a map on a different grid
 * :func:`~.volume_ridges` - skeletonize; emphasize ridges or filaments in the density
 * :func:`~.volume_scale` - scale, shift, normalize, and/or cast to a different data value type
 * :func:`~.split_volume_by_color_zone` - split map by zones previously colored with color zone
 * :func:`~.volume_subtract` - subtract another map from the first
 * :func:`~.volume_threshold` - reassign values that are below a specified minimum and/or above a specified maximum
 * :func:`~.volume_tile` - make a single-plane volume from tiled slices of another volume
 * :func:`~.volume_unbend` - unbend a map near a path formed by markers/links or atoms/bonds
 * :func:`~.volume_unroll` - unroll a cylindrical slab into a flat slab
 * :func:`~.volume_unzone` - show the full extent of a map previously limited to a zone with volume zone
 * :func:`~.volume_zone` - limit the display to a zone around specified atoms, or make a new map with values of zero at grid points within or beyond the zone
   
Volume
------

.. autoclass:: chimerax.map.Volume
    :members:
    :show-inheritance:
    :exclude-members: delete, first_intercept, planes_pick, restore_snapshot, selected, showing_transparent, take_snapshot

RenderingOptions
----------------

.. autoclass:: chimerax.map.volume.RenderingOptions
    :members:

VolumeSurface
-------------

.. autoclass:: chimerax.map.VolumeSurface
    :members:
    :show-inheritance:
    :exclude-members: delete, restore_snapshot, take_snapshot

VolumeImage
-----------

.. autoclass:: chimerax.map.VolumeImage
    :members:
    :show-inheritance:
    :exclude-members: delete, restore_snapshot, single_color, take_snapshot
		      
GridData
--------

.. autoclass:: chimerax.map_data.GridData
    :members:

ArrayGridData
-------------

.. autoclass:: chimerax.map_data.ArrayGridData	       
    :members:
    :exclude-members: read_matrix
    :show-inheritance:

Open and Save 3D Image Data
---------------------------

.. automodule:: chimerax.map.volume
    :members: open_map, save_map

Create a Volume from GridData
-----------------------------

.. autofunction:: chimerax.map.volume_from_grid_data
