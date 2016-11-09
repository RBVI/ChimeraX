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

#include <algorithm>
#include <string>

#include "support.h"

namespace align_algs {

static const char *BadKey = "dictionary key must be tuple of two characters";
static const char *MissingKey = "no score for '%c' vs. '%c'";
static const char *MissingSSKey = "no score for gap open between '%c' and '%c'";
static const char *SeqLenMismatch = "sequence lengths don't match their secondary"
					" structure strings";

#if 0
//
// make_matrix
//	Convert a Python similarity dictionary into a C++ similarity map
//
//	The keys in the Python dictionary must be 2-tuples of
//	single character strings, each representing a base type.
//	This routine does not automatically generate the inverse
//	pair (i.e., does not generate (b,a) when given (a,b) so
//	the caller must provide the full matrix instead of just
//	the upper or lower half.  The values in the dictionary
//	must be floating point numbers.  The '*' character is
//	a wildcard.
//
int
make_matrix(PyObject *dict, Similarity &matrix)
{
	if (!PyDict_Check(dict)) {
		PyErr_SetString(PyExc_TypeError, "matrix must be a dictionary");
		return -1;
	}
	PyObject *key, *value;
	Py_ssize_t pos = 0;
	while (PyDict_Next(dict, &pos, &key, &value)) {
		//
		// Verify key type
		//
		if (!PyTuple_Check(key) || PyTuple_Size(key) != 2) {
			PyErr_SetString(PyExc_TypeError, BadKey);
			return -1;
		}
		PyObject *pk0 = PyTuple_GetItem(key, 0);
		if (!PyUnicode_Check(pk0)) {
			PyErr_SetString(PyExc_TypeError, BadKey);
			return -1;
		}
		char *k0 = PyUnicode_AsUTF8(pk0);
		if (strlen(k0) != 1) {
			PyErr_SetString(PyExc_TypeError, BadKey);
			return -1;
		}
		PyObject *pk1 = PyTuple_GetItem(key, 1);
		if (!PyUnicode_Check(pk1)) {
			PyErr_SetString(PyExc_TypeError, BadKey);
			return -1;
		}
		char *k1 = PyUnicode_AsUTF8(pk1);
		if (strlen(k1) != 1) {
			PyErr_SetString(PyExc_TypeError, BadKey);
			return -1;
		}

		//
		// Verify value type
		//
		if (!PyFloat_Check(value)) {
			PyErr_SetString(PyExc_TypeError,
				"dictionary value must be float");
			return -1;
		}
		double v = PyFloat_AsDouble(value);

		//
		// Store in C++ map
		//
		matrix[Pair(*k0, *k1)] = v;
	}
	return 0;
}
#endif

//
// matrix_lookup
//	Look up the matrix value for the given characters
//
//	Uses wildcards if the characters are not found directly.
//
Similarity::const_iterator
matrix_lookup(const Similarity &matrix, char c1, char c2)
{
	Similarity::const_iterator it = matrix.find(Pair(c1, c2));
	if (it != matrix.end())
		return it;
	
	it = matrix.find(Pair('*', c2));
	if (it != matrix.end())
		return it;
	
	it = matrix.find(Pair(c1, '*'));
	if (it != matrix.end())
		return it;
	
	return matrix.find(Pair('*', '*'));
}

}  // namespace align_algs
