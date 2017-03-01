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

geometry: Points and coordinate systems
=======================================

* :ref:`place <place-anchor>` - position and orientation of objects
* :ref:`vector <vector-anchor>` - operations on 3-vectors
* :ref:`bounds <bounds-anchor>` - bounds of objects in a scene
* :ref:`misc-geom-anchor` - geometric calculations

.. _place-anchor:

.. automodule:: chimerax.core.geometry.place
    :members:
    :show-inheritance:

.. _vector-anchor:

.. automodule:: chimerax.core.geometry.vector
    :members:  distance, norm, normalize_vector, normalize_vectors, vector_sum, cross_product, inner_product
    :show-inheritance:

.. autofunction:: chimerax.core.geometry._geometry.inner_product_64

.. _bounds-anchor:

.. automodule:: chimerax.core.geometry.bounds
    :members:
    :show-inheritance:

.. _misc-geom-anchor:

Miscellaneous Routines
----------------------

.. autofunction:: chimerax.core.geometry.align.align_points

.. automodule:: chimerax.core.geometry._geometry
    :members: closest_cylinder_intercept, closest_sphere_intercept, closest_triangle_intercept, find_close_points, find_closest_points, find_close_points_sets, natural_cubic_spline
    :show-inheritance:
