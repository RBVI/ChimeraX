/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
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
