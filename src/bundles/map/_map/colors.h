// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Routines to convert color formats.
//
#ifndef COLORS_HEADER_INCLUDED
#define COLORS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

namespace Map_Cpp
{

extern "C" {

// ----------------------------------------------------------------------------
// Copy array of luminosity+alpha uint8 values to array of rgba uint8 values.
//
// copy_la_to_rgba(la, color, rgba)
//
PyObject *copy_la_to_rgba(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Blend array of luminosity+alpha uint8 values with array of rgba uint8 values.
//
// blend_la_to_rgba(la, color, rgba)
//
PyObject *blend_la_to_rgba(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Copy array of luminosity uint8 values to array of rgba uint8 values.
//
// copy_l_to_rgba(l, color, rgba)
//
PyObject *copy_l_to_rgba(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Blend array of luminosity uint8 values with array of rgba uint8 values.
//
// blend_l_to_rgba(l, color, rgba)
//
PyObject *blend_l_to_rgba(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Blend array of rgb values with array of rgba uint8 values.
//
// blend_rgb_to_rgba(rgb, rgba)
//
PyObject *blend_rgb_to_rgba(PyObject *, PyObject *args, PyObject *keywds);

// ----------------------------------------------------------------------------
// Blend two arrays with rgba uint8 values.
//
// blend_rgba(la, color, rgba)
//
PyObject *blend_rgba(PyObject *, PyObject *args, PyObject *keywds);

}	// end extern C

}	// end of namespace Map_Cpp

#endif
