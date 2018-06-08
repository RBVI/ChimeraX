// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

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
