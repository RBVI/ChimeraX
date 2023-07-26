// vi: set expandtab ts=4 sw=4:

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
#include <string>
#include <vector>

namespace mmcif {

PyObject*   parse_coreCIF_file(const char* filename, PyObject* logger);
PyObject*   parse_coreCIF_file(const char* filename, const std::vector<std::string> &extra_categories, PyObject* logger);
PyObject*   parse_coreCIF_buffer(const unsigned char* buffer, PyObject* logger);
PyObject*   parse_coreCIF_buffer(const unsigned char* buffer, const std::vector<std::string> &extra_categories, PyObject* logger);

}  // namespace mmcif
