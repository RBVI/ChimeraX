// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
//
#ifndef INTERCEPT_HEADER_INCLUDED
#define INTERCEPT_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
// Find first triangle intercept along line segment from xyz1 to xyz2.
//   closest_triangle_intercept(float varray[n,3], int tarray[m,3], float xyz1[3], float xyz2[3])
//     -> (float fmin, int tnum)
PyObject *closest_triangle_intercept(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *closest_triangle_intercept_doc;

// Find first sphere intercept along line segment from xyz1 to xyz2.
// closest_sphere_intercept(float centers[n,3], float radii[n], float xyz1[3], float xyz2[3])
//   -> (float fmin, int snum)
PyObject *closest_sphere_intercept(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *closest_sphere_intercept_doc;

// Find which spheres intercept a line segment from xyz1 to xyz2.
// segment_intercepts_spheres(float centers[n,3], float radius, float xyz1[3], float xyz2[3])
//   -> mask bool array
PyObject *segment_intercepts_spheres(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *segment_intercepts_spheres_doc;

// Find first cylinder intercept along line segment from xyz1 to xyz2.
// closest_cylinder_intercept(float cxyz1[n,3], float cxyz2[n,3], float radii[n], float xyz1[3], float xyz2[3])
//   -> (float fmin, int cnum)
PyObject *closest_cylinder_intercept(PyObject *s, PyObject *args, PyObject *keywds);
extern const char *closest_cylinder_intercept_doc;

}

#endif
