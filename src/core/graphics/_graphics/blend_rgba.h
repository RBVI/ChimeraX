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
// Blend images for motion blur.
//
#ifndef BLEND_RGBA_HEADER_INCLUDED
#define BLEND_RGBA_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

// void blur_blend_images(float f, PyObject *rgba1, PyObject *rgba2, PyObject *bgcolor, float alpha, PyObject *rgba);
PyObject *blur_blend_images(PyObject *s, PyObject *args, PyObject *keywds);
PyObject *accumulate_images(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
