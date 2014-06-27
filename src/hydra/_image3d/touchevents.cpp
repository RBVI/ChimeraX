#include <iostream>
#include <Python.h>
#include "touchevents_mac.h"

extern "C" PyObject *accept_touch_events(PyObject *s, PyObject *args, PyObject *keywds)
{
  long nsview;
  const char *kwlist[] = {"nsview", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("l"),
				   (char **)kwlist,
				   &nsview))
    return NULL;

#ifdef __APPLE__
  mac_accept_touch_events((void *)nsview);
#endif

  Py_INCREF(Py_None);
  return Py_None;
}
