geometry: Points and coordinate systems
=======================================

* :ref:`place <place-anchor>` - position and orientation of objects
* :ref:`vector <vector-anchor>` - operations on 3-vectors
* :ref:`bounds <bounds-anchor>` - bounds of objects in a scene
* :ref:`misc-geom-anchor` - geometric calculations

.. _place-anchor:

.. automodule:: chimera.core.geometry.place
    :members:

.. _vector-anchor:

.. automodule:: chimera.core.geometry.vector
    :members:  distance, norm, normalize_vector, normalize_vectors, vector_sum, cross_product, inner_product

.. autofunction:: chimera.core.geometry._geometry.inner_product_64

.. _bounds-anchor:

.. automodule:: chimera.core.geometry.bounds
    :members:

.. _misc-geom-anchor:

Miscellaneous Routines
----------------------

.. autofunction:: chimera.core.geometry.align.align_points

.. automodule:: chimera.core.geometry._geometry
    :members: closest_cylinder_intercept, closest_sphere_intercept, closest_triangle_intercept, find_close_points, find_closest_points, find_close_points_sets, natural_cubic_spline
