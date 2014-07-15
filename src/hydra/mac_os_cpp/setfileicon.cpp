#include <Python.h>

#ifdef __APPLE__
#include "setfileicon_mac.h"
#endif

extern "C" PyObject *set_file_icon(PyObject *s, PyObject *args, PyObject *keywds)
{
  const char *file_path, *image_data;
  int image_bytes;
  const char *kwlist[] = {"file_path", "image_data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("sz#"),
				   (char **)kwlist,
				   &file_path, &image_data, &image_bytes))
    return NULL;

#ifdef __APPLE__
  bool set = set_mac_file_icon(file_path, image_data, image_bytes);
#else
  bool set = false;
#endif

  PyObject *r = (set ? Py_True : Py_False);
  Py_INCREF(r);
  return r;
}

extern "C" PyObject *can_set_file_icon(PyObject *s, PyObject *args, PyObject *keywds)
{
  const char *kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>(""),
				   (char **)kwlist))
    return NULL;
#ifdef __APPLE__
  return Py_True;
#else
  return Py_False;
#endif
}
