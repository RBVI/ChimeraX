// ----------------------------------------------------------------------------
// Compute solvent accessible surface areas.
//
#ifndef SASA_HEADER_INCLUDED
#define SASA_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// bool surface_area_of_spheres(centers, radii, areas).  Can fail in degenerate cases returning false.
PyObject *surface_area_of_spheres(PyObject *s, PyObject *args, PyObject *keywds);

// Use points on unit sphere, count how many are inside other spheres.
//   estimate_surface_area_of_spheres(centers, radii, sphere_points, point_weights, areas)
PyObject *estimate_surface_area_of_spheres(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
