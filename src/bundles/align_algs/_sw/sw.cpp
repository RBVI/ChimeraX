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
#include <algorithm>
#include <string>
#include <map>

#include "align_algs/support.h"

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

typedef std::pair<char, char> Pair;
typedef std::map<Pair, double> Similarity;

const char *BadKey = "dictionary key must be tuple of two characters";
const char *MissingKey = "no score for '%c' vs. '%c'";
const char *MissingSSKey = "no score for gap open between '%c' and '%c'";
const char *SeqLenMismatch = "sequence lengths don't match their secondary structure strings";

using align_algs::make_matrix;
using align_algs::matrix_lookup;

//
// score
//	Compute the score of the best Smith-Waterman alignment
//
//	The Python function takes five arguments:
//		seq1		first sequence
//		seq2		second sequence
//		matrix		similarity dictionary (see above)
//		gap_open		gap opening penalty
//		gap_extend	gap extension penalty
//	and returns the best score.  This function is mostly
//	useful for optimizing similarity matrices.
//
extern "C" {

static
PyObject *
score(PyObject *, PyObject *args)
{
	char *seq1, *seq2;
	PyObject *m;
	double gap_open, gap_extend;
	if (!PyArg_ParseTuple(args, PY_STUPID "ssOdd", &seq1, &seq2, &m,
		&gap_open, &gap_extend))
		return nullptr;
	Similarity matrix;
	if (make_matrix(m, matrix) < 0)
		return nullptr;
	int rows = strlen(seq1) + 1;
	int cols = strlen(seq2) + 1;
	double **H = new double *[rows];
	for (int i = 0; i < rows; ++i) {
		H[i] = new double[cols];
		for (int j = 0; j < cols; ++j)
			H[i][j] = 0;
	}
	double best_score = 0;
	for (int i = 1; i < rows; ++i) {
		for (int j = 1; j < cols; ++j) {
			Similarity::const_iterator it = matrix_lookup(matrix, seq1[i - 1], seq2[j - 1]);
			if (it == matrix.end()) {
				char buf[80];
				(void) sprintf(buf, MissingKey, seq1[i - 1], seq2[j - 1]);
				PyErr_SetString(PyExc_KeyError, buf);
				return nullptr;
			}
			double best = H[i - 1][j - 1] + (*it).second;
			for (int k = 1; k < i; ++k) {
				double score = H[i - k][j] - gap_open - k * gap_extend;
				if (score > best)
					best = score;
			}
			for (int l = 1; l < j; ++l) {
				double score = H[i][j - l] - gap_open - l * gap_extend;
				if (score > best)
					best = score;
			}
			if (best < 0)
				best = 0;
			H[i][j] = best;
			if (best > best_score)
				best_score = best;
		}
	}
	for (int i = 0; i < rows; ++i)
		delete [] H[i];
	delete [] H;
	return PyFloat_FromDouble(best_score);
}

//
// align
//	Compute the best Smith-Waterman score and alignment
//
//	The Python function takes five mandatory arguments:
//		seq1		first sequence
//		seq2		second sequence
//		matrix		similarity dictionary (see above)
//		gap_open		gap opening penalty
//		gap_extend	gap extension penalty
//	and the following optional keyword arguments:
//		gap_char		character used in gaps (default: '-')
//		ss_matrix	secondary-structure scoring dictionary (nullptr)
//		ss_fraction	fraction of weight given to SS scoring (0.3)
//		gap_open_helix	intra-helix gap opening penalty (18)
//		gap_open_strand	intra-strand gap opening penalty (18)
//		gap_open_other	other gap opening penalty (6)
//		ss1		first SS "sequence" (i.e. composed of H/S/O)
//		ss2		second SS "sequence" (i.e. composed of H/S/O)
//	and returns the best score and the alignment.
//	The alignment is represented by a 2-tuple of strings,
//	where the first and second elements of the tuple
//	represent bases (and gaps) from seq1 and seq2 respectively.
//
//	Secondary-structure features are only enabled if the ss_matrix dictionary
//	is provided and is not None.  In that case the ssGapOpen penalties are
//	used and gap_open is ignored.  The residue-matching score is a
//	combination of the secondary-structure and similarity scores, weighted
//	by the ss_fraction.
static
PyObject *
align(PyObject *, PyObject *args, PyObject *kwdict)
{
	char *seq1, *seq2;
	PyObject *m;
	double gap_open, gap_extend;
	char gap_char = '-';
	PyObject *ss_m = nullptr;
	double ss_fraction = 0.3;
	double gap_open_helix = 18.0;
	double gap_open_strand = 18.0;
	double gap_open_other = 18.0;
	char *ss1 = nullptr;
	char *ss2 = nullptr;
	char *gap_char_string = nullptr;
	static char *kwlist[] = { PY_STUPID "seq1", PY_STUPID "seq2",
		PY_STUPID "matrix", PY_STUPID "gap_open", PY_STUPID "gap_extend",
		PY_STUPID "gap_char", PY_STUPID "ss_matrix",
		PY_STUPID "ss_fraction", PY_STUPID "gap_open_helix",
		PY_STUPID "gap_open_strand", PY_STUPID "gap_open_other",
		PY_STUPID "ss1", PY_STUPID "ss2",
		nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwdict,
			PY_STUPID "ssOdd|sOddddss",
			kwlist, &seq1, &seq2, &m, &gap_open, &gap_extend,
			&gap_char_string, &ss_m, &ss_fraction, &gap_open_helix,
			&gap_open_strand, &gap_open_other, &ss1, &ss2))
		return nullptr;
	// Don't want caller to have to figure out how to send a single-byte gap character,
	// (i.e. have to use "decode" in the call), so accept a string as the gap character
	// and process/check it
	if (gap_char_string != nullptr) {
		if (strlen(gap_char_string) != 1) {
			PyErr_SetString(PyExc_ValueError, "gap_char must be a single-character string");
			return nullptr;
		}
		gap_char = gap_char_string[0];
	}

	//
	// Convert Python similarity dictionary into C++ similarity map
	//
	Similarity matrix, ss_matrix;
	if (make_matrix(m, matrix) < 0)
		return nullptr;
	size_t rows = strlen(seq1) + 1;
	size_t cols = strlen(seq2) + 1;

	// handle secondary-structure setup if appropriate
	double *row_gap_opens = new double[rows];
	double *col_gap_opens = new double[cols];
	bool doing_ss = ss_m != nullptr && ss1 != nullptr && ss2 != nullptr && ss_m != Py_None;
	row_gap_opens[0] = col_gap_opens[0] = 0;
	if (doing_ss) {
		if (strlen(ss1) + 1 != rows || strlen(ss2) + 1 != cols) {
			PyErr_SetString(PyExc_ValueError, SeqLenMismatch);
			return nullptr;
		}
		if (make_matrix(ss_m, ss_matrix) < 0)
			return nullptr;
		size_t r, c;
		for (r = 1; r < rows; ++r) {
			char ssl = ss1[r-1];
			char ssr = ss1[r];
			if (ssl == 'H' && ssr == 'H')
				row_gap_opens[r] = gap_open_helix;
			else if (ssl == 'S' && ssr == 'S')
				row_gap_opens[r] = gap_open_strand;
			else
				row_gap_opens[r] = gap_open_other;
		}
		for (c = 1; c < cols; ++c) {
			char ssl = ss2[c-1];
			char ssr = ss2[c];
			if (ssl == 'H' && ssr == 'H')
				col_gap_opens[c] = gap_open_helix;
			else if (ssl == 'S' && ssr == 'S')
				col_gap_opens[c] = gap_open_strand;
			else
				col_gap_opens[c] = gap_open_other;
		}
	} else {
		size_t r, c;
		for (r = 1; r < rows; ++r)
			row_gap_opens[r] = gap_open;
		for (c = 1; c < cols; ++c)
			col_gap_opens[c] = gap_open;
		
	}

	//
	// Allocate space for the score matrix and the backtracking matrix
	//
	double **H = new double *[rows];
	int **bt = new int *[rows];
	for (size_t i = 0; i < rows; ++i) {
		H[i] = new double[cols];
		bt[i] = new int[cols];
		for (size_t j = 0; j < cols; ++j) {
			H[i][j] = 0;
			bt[i][j] = 0;
		}
	}

	//
	// Fill in all cells of the score matrix
	//
	double best_score = 0;
	int best_row = 0, best_column = 0;
	for (size_t i = 1; i < rows; ++i) {
		for (size_t j = 1; j < cols; ++j) {
			//
			// Start with the matching score
			//
			Similarity::const_iterator it = matrix_lookup(matrix, seq1[i - 1], seq2[j - 1]);
			if (it == matrix.end()) {
				char buf[80];
				(void) sprintf(buf, MissingKey, seq1[i - 1], seq2[j - 1]);
				PyErr_SetString(PyExc_KeyError, buf);
				return nullptr;
			}
			double match_score = (*it).second;
			if (doing_ss) {
				Similarity::const_iterator it = matrix_lookup(ss_matrix, ss1[i - 1], ss2[j - 1]);
				if (it == ss_matrix.end()) {
					char buf[80];
					(void) sprintf(buf, MissingSSKey, ss1[i - 1], ss2[j - 1]);
					PyErr_SetString(PyExc_KeyError, buf);
					return nullptr;
				}
				match_score = (1.0 - ss_fraction) * match_score + ss_fraction * (*it).second;
			}
			double best = H[i - 1][j - 1] + match_score;
			int op = 0;

			//
			// Check if insertion is better
			//
			double go = col_gap_opens[j];
			for (size_t k = 1; k < i; ++k) {
				double score = H[i - k][j] - go - k * gap_extend;
				if (score > best) {
					best = score;
					op = k;
				}
			}

			//
			// Check if deletion is better
			//
			go = row_gap_opens[i];
			for (size_t l = 1; l < j; ++l) {
				double score = H[i][j - l] - go - l * gap_extend;
				if (score > best) {
					best = score;
					op = -l;
				}
			}

			//
			// Check if this is just a bad place
			// to start/end an alignment
			//
			if (best < 0) {
				best = 0;
				op = 0;
			}

			//
			// Save the best score in the score Matrix
			//
			H[i][j] = best;
			bt[i][j] = op;
			if (best > best_score) {
				best_score = best;
				best_row = i;
				best_column = j;
			}
		}
	}

	//
	// Use the backtrack matrix to create the best alignment
	//
	std::string a1, a2;
	while (H[best_row][best_column] > 0) {
		int op = bt[best_row][best_column];
		if (op > 0) {
			for (int k = 0; k < op; ++k) {
				--best_row;
				a1.append(1, seq1[best_row]);
				a2.append(1, gap_char);
			}
		}
		else if (op == 0) {
			--best_row;
			--best_column;
			a1.append(1, seq1[best_row]);
			a2.append(1, seq2[best_column]);
		}
		else {
			op = -op;
			for (int k = 0; k < op; ++k) {
				--best_column;
				a1.append(1, gap_char);
				a2.append(1, seq2[best_column]);
			}
		}
	}
	std::reverse(a1.begin(), a1.end());
	std::reverse(a2.begin(), a2.end());
	PyObject *alignment = PyTuple_New(2);
	PyTuple_SetItem(alignment, 0, PyUnicode_FromString(a1.c_str()));
	PyTuple_SetItem(alignment, 1, PyUnicode_FromString(a2.c_str()));

	//
	// Release the score and backtrack matrix memory
	//
	for (size_t i = 0; i < rows; ++i) {
		delete [] H[i];
		delete [] bt[i];
	}
	delete [] H;
	delete [] bt;
	delete [] row_gap_opens;
	delete [] col_gap_opens;

	//
	// Return our results
	//
	return Py_BuildValue(PY_STUPID "fO", best_score, alignment);
}

}


#define MATRIX_EXPLAIN \
"\n\nThe keys in the similarity dictionary must be 2-tuples of\n" \
"single character strings, each representing a residue type.\n" \
"This routine does not automatically generate the inverse\n" \
"pair (i.e., does not generate (b,a) when given (a,b) so\n" \
"the caller must provide the full matrix instead of just\n" \
"the upper or lower half.  The values in the dictionary\n" \
"must be floating point numbers.  The '*' character is\n" \
"a wildcard."

static const char* docstr_score =
"score\n"
"Compute the score of the best Smith-Waterman alignment\n"
"\n"
"The function takes five arguments:\n"
"	seq1		first sequence\n"
"	seq2		second sequence\n"
"	matrix		similarity dictionary (see below)\n"
"	gap_open		gap opening penalty\n"
"	gap_extend	gap extension penalty\n"
"and returns the best score.  This function is mostly\n"
"useful for optimizing similarity matrices."
MATRIX_EXPLAIN;

static const char* docstr_align =
"align\n"
"Compute the best Smith-Waterman score and alignment\n"
"\n"
"The function takes five mandatory arguments:\n"
"	seq1		first sequence\n"
"	seq2		second sequence\n"
"	matrix		similarity dictionary (see above)\n"
"	gap_open		gap opening penalty\n"
"	gap_extend	gap extension penalty\n"
"and the following optional keyword arguments:\n"
"	gap_char		character used in gaps (default: '-')\n"
"	ss_matrix	secondary-structure scoring dictionary (nullptr)\n"
"	ss_fraction	fraction of weight given to SS scoring (0.3)\n"
"	gap_open_helix	intra-helix gap opening penalty (18)\n"
"	gap_open_strand	intra-strand gap opening penalty (18)\n"
"	gap_open_other	other gap opening penalty (6)\n"
"	ss1		first SS \"sequence\" (i.e. composed of H/S/O)\n"
"	ss2		second SS \"sequence\" (i.e. composed of H/S/O)\n"
"and returns the best score and the alignment.\n"
"The alignment is represented by a 2-tuple of strings,\n"
"where the first and second elements of the tuple\n"
"represent bases (and gaps) from seq1 and seq2 respectively.\n"
"\n"
"Secondary-structure features are only enabled if the ss_matrix dictionary\n"
"is provided and is not None.  In that case the ss_gap_open penalties are\n"
"used and gap_open is ignored.  The residue-matching score is a\n"
"combination of the secondary-structure and similarity scores, weighted\n"
"by the ss_fraction."
MATRIX_EXPLAIN;

static PyMethodDef sw_methods[] = {
	{ PY_STUPID "score", score,	METH_VARARGS, PY_STUPID docstr_score	},
	{ PY_STUPID "align", (PyCFunction)align, METH_VARARGS|METH_KEYWORDS, PY_STUPID docstr_align	},
	{ nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef sw_def =
{
	PyModuleDef_HEAD_INIT,
	"_sw",
	"Smith-Waterman alignment methods",
	-1,
	sw_methods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

PyMODINIT_FUNC
PyInit__sw()
{
	return PyModule_Create(&sw_def);
}
