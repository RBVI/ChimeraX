// ----------------------------------------------------------------------------
// Enable touch events on Mac
//
#ifndef TOUCHEVENTS_HEADER_INCLUDED
#define TOUCHEVENTS_HEADER_INCLUDED

#include <Python.h>			// use PyObject

extern "C"
{

PyObject *accept_touch_events(PyObject *s, PyObject *args, PyObject *keywds);

}

#endif
