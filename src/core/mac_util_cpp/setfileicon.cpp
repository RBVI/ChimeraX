#include <Python.h>

#include "setfileicon_mac.h"

extern "C" PyObject *set_file_icon(PyObject *, PyObject *args, PyObject *keywds)
{
  const char *file_path, *image_data;
  int image_bytes;
  const char *kwlist[] = {"file_path", "image_data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("sz#"),
				   (char **)kwlist,
				   &file_path, &image_data, &image_bytes))
    return NULL;

  bool set = set_mac_file_icon(file_path, image_data, image_bytes);
  PyObject *r = (set ? Py_True : Py_False);
  Py_INCREF(r);
  return r;
}
