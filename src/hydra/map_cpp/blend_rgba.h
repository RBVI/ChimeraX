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
