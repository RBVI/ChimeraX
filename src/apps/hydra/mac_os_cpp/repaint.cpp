#include <Python.h>

#include "repaint_mac.h"

extern "C" PyObject *repaint_window(PyObject *s, PyObject *args, PyObject *keywds)
{
  unsigned long win;
  const char *kwlist[] = {"window_id", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("k"), (char **)kwlist, &win))
    return NULL;

  repaint_nsview(win);

  Py_INCREF(Py_None);
  return Py_None;
}
