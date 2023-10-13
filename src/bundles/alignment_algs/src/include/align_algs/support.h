// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef align_algs_support
#define align_algs_support

#include <Python.h>
#include <map>
#include <utility>

#ifdef DYNAMIC_LIBRARY
# ifdef _WIN32
#  ifdef ALIGN_ALGS_EXPORT
#   define ALIGN_ALGS_IMEX __declspec(dllexport)
#  else
#   define ALIGN_ALGS_IMEX __declspec(dllimport)
#  endif
# else
#  if (__GNUC__ > 4) || (__GNUC__ == 4 && (defined(__APPLE__) || __GNUC_MINOR__ >= 3))
#   define ALIGN_ALGS_IMEX __attribute__((__visibility__("default")))
#  else
#   define ALIGN_ALGS_IMEX
#  endif
# endif
#else
# define ALIGN_ALGS_IMEX
#endif

namespace align_algs {

typedef std::pair<char, char> Pair;
typedef std::map<Pair, double> Similarity;

ALIGN_ALGS_IMEX
int make_matrix(PyObject *dict, Similarity &matrix, bool is_ss_matrix = false);
ALIGN_ALGS_IMEX
Similarity::const_iterator matrix_lookup(const Similarity &matrix, char c1, char c2);

#define SIM_MATRIX_EXPLAIN \
"The keys in the similarity dictionary must be 2-tuples of\n" \
"single character strings, each representing a residue type.\n" \
"This routine does not automatically generate the inverse\n" \
"pair (i.e., does not generate (b,a) when given (a,b) so\n" \
"the caller must provide the full matrix instead of just\n" \
"the upper or lower half.  The values in the dictionary\n" \
"must be floating point numbers.  The '*' character is\n" \
"a wildcard."

}  // namespace align_algs

#endif  // align_algs_support
