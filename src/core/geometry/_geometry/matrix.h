// vi: set expandtab shiftwidth=4 softtabstop=4:

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

#ifndef MATRIX_HEADER_INCLUDED
#define  MATRIX_HEADER_INCLUDED

extern "C" {

PyObject *multiply_matrices(PyObject *, PyObject *args, PyObject *keywds);
PyObject *multiply_matrix_lists(PyObject *, PyObject *args, PyObject *keywds);
PyObject *same_matrix(PyObject *, PyObject *args, PyObject *keywds);
PyObject *is_identity_matrix(PyObject *, PyObject *args, PyObject *keywds);
PyObject *set_scale_matrix(PyObject *, PyObject *args, PyObject *keywds);
PyObject *set_translation_matrix(PyObject *, PyObject *args, PyObject *keywds);
PyObject *opengl_matrix(PyObject *, PyObject *args, PyObject *keywds);
PyObject *opengl_matrices(PyObject *, PyObject *args, PyObject *keywds);
PyObject *invert_orthonormal(PyObject *, PyObject *args, PyObject *keywds);
PyObject *look_at(PyObject *, PyObject *args);

}

#endif
