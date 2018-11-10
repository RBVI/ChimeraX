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

#include <Python.h>

#include "enablemultitouch_mac.h"

extern "C" PyObject *enable_multitouch(PyObject *, PyObject *args, PyObject *keywds)
{
  long win_id;
  const char *kwlist[] = {"window_id", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("l"),
				   (char **)kwlist,
				   &win_id))
    return NULL;

  enable_mac_multitouch(win_id);
  PyObject *r = Py_None;
  Py_INCREF(r);
  return r;
}
