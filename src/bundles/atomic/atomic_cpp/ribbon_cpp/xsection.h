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
// Create or delete RibbonXSection.
//  
// rxsection_new(coords, coords2, normals, norals2, faceted, test) -> RibbonXSection pointer
// rxsection_delete(xs)
//
PyObject *rxsection_new(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *rxsection_delete(PyObject *s, PyObject *args, PyObject *keywds);

//
// Triangulate an extruded cross-section.
//  
// rxsection_extrude(xs, centers, tangents, normals, cap_front, cap_back, offset)
//    -> vertices, normals, triangles, front_band, back_band
//
PyObject *rxsection_extrude(PyObject *s, PyObject *args, PyObject *keywds);

//
// Make a scaled or tapering cross-section.
//  
// rxsection_scale(xs, x_scale, y_scale) -> RibbonsXSection pointer
// rxsection_arrow(xs, x1_scale, y1_scale, x2_scale, y2_scale) -> RibbonsXSection pointer
//
PyObject *rxsection_scale(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *rxsection_arrow(PyObject *s, PyObject *args, PyObject *keywds);

// Compute ribbon extrusions for a spline path with varying cross-sections adding them
// to geometry.
//  
//  ribbon_extrusions(center, tangents, normals, ranges, num_res, xs_front, xs_back, geometry)
//
PyObject *ribbon_extrusions(PyObject *s, PyObject *args, PyObject *keywds);

//
// Accumulate triangles from multiple extrusions.
// Also keeps track of triangle and vertex ranges for each residue.
//
PyObject *geometry_new(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *geometry_delete(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *geometry_add_range(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *geometry_set_range_offset(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *geometry_ranges(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *geometry_empty(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *geometry_arrays(PyObject *s, PyObject *args, PyObject *keywds);

// Color vertices according to residue ribbon colors.
PyObject *ribbon_vertex_colors(PyObject *s, PyObject *args, PyObject *keywds);
}

#endif
