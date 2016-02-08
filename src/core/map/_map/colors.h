// vi: set expandtab shiftwidth=4 softtabstop=4:
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
// Blend two arrays with rgba uint8 values.
//
// blend_rgba(la, color, rgba)
//
PyObject *blend_rgba(PyObject *, PyObject *args, PyObject *keywds);

}	// end extern C

}	// end of namespace Map_Cpp

#endif
