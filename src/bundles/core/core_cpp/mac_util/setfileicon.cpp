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
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "setfileicon_mac.h"

extern "C" PyObject *set_file_icon(PyObject *, PyObject *args, PyObject *keywds)
{
  const char *file_path, *image_data;
  Py_ssize_t image_bytes;
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
