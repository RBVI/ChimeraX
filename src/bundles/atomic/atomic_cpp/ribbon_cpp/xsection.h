// vi: set expandtab ts=4 sw=4:

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
#ifndef XSECTION_HEADER_INCLUDED
#define XSECTION_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{
//
// Create a RibbonXSection.
//  
// rxsection_new(coords, coords2, normals, norals2, faceted, test) -> RibbonXSection pointer
//
PyObject *rxsection_new(PyObject *s, PyObject *args, PyObject *keywds);

//
// Delete a RibbonXSection.
//  
// rxsection_delete(xs)
//
PyObject *rxsection_delete(PyObject *s, PyObject *args, PyObject *keywds);

//
// Triangulate an extruded cross-section.
//  
// rxsection_extrude(xs, centers, tangents, normals, cap_front, cap_back, offset)
//    -> vertices, normals, triangles, front_band, back_band
//
PyObject *rxsection_extrude(PyObject *s, PyObject *args, PyObject *keywds);

//
// Stitch together two extrusions with a band of triangles.
//  
// rxsection_blend(xs, front_band, back_band) -> triangles
//
PyObject *rxsection_blend(PyObject *s, PyObject *args, PyObject *keywds);

//
// Make a scaled cross-section.
//  
// rxsection_scale(xs, x_scale, y_scale) -> RibbonsXSection pointer
//
PyObject *rxsection_scale(PyObject *s, PyObject *args, PyObject *keywds);

//
// Make a tapering cross-section.
//  
// rxsection_scale(xs, x1_scale, y1_scale, x2_scale, y2_scale) -> RibbonsXSection pointer
//
PyObject *rxsection_arrow(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
