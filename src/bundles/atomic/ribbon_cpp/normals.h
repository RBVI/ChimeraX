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
//
#ifndef NORMALS_HEADER_INCLUDED
#define NORMALS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Modify array of normals by rotating around tangents along path so that
// final normal aligns with end_normal.
//  
// smooth_twist(tangents, normals, end_normal)
//
PyObject *smooth_twist_py(PyObject *s, PyObject *args, PyObject *keywds);

// parallel_transport(tangents, normal) -> normals
PyObject *parallel_transport_py(PyObject *s, PyObject *args, PyObject *keywds);  

// dihedral_angle(u, v, t) -> angle in radians
PyObject *dihedral_angle_py(PyObject *s, PyObject *args, PyObject *keywds);  

// path_plane_normals(path, tangents) -> normals
PyObject *path_plane_normals(PyObject *s, PyObject *args, PyObject *keywds);  

// path_guide_normals(path, guides, tangents) -> normals
PyObject *path_guide_normals(PyObject *s, PyObject *args, PyObject *keywds);  
}

void smooth_twist(const float *tangents, int num_pts, float *normals, const float *n_end);
void parallel_transport(int num_pts, const float* tangents, const float* n0, float* normals,
			bool backwards = false);
float dihedral_angle(const float *u, const float *v, const float *t);

#endif
