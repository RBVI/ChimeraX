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

#ifndef align_algs
#define align_algs

//#include <Python.h>
#include <map>
#include <utility>

namespace align_algs {

typedef std::pair<char, char> Pair;
typedef std::map<Pair, double> Similarity;

// int make_matrix(PyObject *dict, Similarity &matrix);
Similarity::const_iterator matrix_lookup(const Similarity &matrix, char c1, char c2);

}  // namespace align_algs

#endif  // align_algs
