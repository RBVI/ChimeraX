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

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

typedef std::pair<char, char> Pair;
typedef std::map<Pair, double> Similarity;

const char *BadKey = "dictionary key must be tuple of two characters";
const char *MissingKey = "no score for '%c' vs. '%c'";
const char *MissingSSKey = "no score for gap open between '%c' and '%c'";
const char *SeqLenMismatch = "sequence lengths don't match their secondary"
					" structure strings";

//
// makeMatrix
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
static
int
makeMatrix(PyObject *dict, Similarity &matrix)
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

//
// matrixLookup
//	Look up the matrix value for the given characters
//
//	Uses wildcards if the characters are not found directly.
//
static
Similarity::const_iterator
matrixLookup(const Similarity &matrix, char c1, char c2)
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

//
// score
//	Compute the score of the best Smith-Waterman alignment
//
//	The Python function takes five arguments:
//		seq1		first sequence
//		seq2		second sequence
//		matrix		similarity dictionary (see above)
//		gapOpen		gap opening penalty
//		gapExtend	gap extension penalty
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
	double gapOpen, gapExtend;
	if (!PyArg_ParseTuple(args, PY_STUPID "ssOdd", &seq1, &seq2, &m,
		&gapOpen, &gapExtend))
		return NULL;
	Similarity matrix;
	if (makeMatrix(m, matrix) < 0)
		return NULL;
	int rows = strlen(seq1) + 1;
	int cols = strlen(seq2) + 1;
	double **H = new double *[rows];
	for (int i = 0; i < rows; ++i) {
		H[i] = new double[cols];
		for (int j = 0; j < cols; ++j)
			H[i][j] = 0;
	}
	double bestScore = 0;
	for (int i = 1; i < rows; ++i) {
		for (int j = 1; j < cols; ++j) {
			Similarity::const_iterator it =
				matrixLookup(matrix, seq1[i - 1], seq2[j - 1]);
			if (it == matrix.end()) {
				char buf[80];
				(void) sprintf(buf, MissingKey, seq1[i - 1],
							seq2[j - 1]);
				PyErr_SetString(PyExc_KeyError, buf);
				return NULL;
			}
			double best = H[i - 1][j - 1] + (*it).second;
			for (int k = 1; k < i; ++k) {
				double score = H[i - k][j] - gapOpen
						- k * gapExtend;
				if (score > best)
					best = score;
			}
			for (int l = 1; l < j; ++l) {
				double score = H[i][j - l] - gapOpen
						- l * gapExtend;
				if (score > best)
					best = score;
			}
			if (best < 0)
				best = 0;
			H[i][j] = best;
			if (best > bestScore)
				bestScore = best;
		}
	}
	for (int i = 0; i < rows; ++i)
		delete [] H[i];
	delete [] H;
	return PyFloat_FromDouble(bestScore);
}

//
// align
//	Compute the best Smith-Waterman score and alignment
//
//	The Python function takes five mandatory arguments:
//		seq1		first sequence
//		seq2		second sequence
//		matrix		similarity dictionary (see above)
//		gapOpen		gap opening penalty
//		gapExtend	gap extension penalty
//	and the following optional keyword arguments:
//		gapChar		character used in gaps (default: '-')
//		ssMatrix	secondary-structure scoring dictionary (NULL)
//		ssFraction	fraction of weight given to SS scoring (0.3)
//		gapOpenHelix	intra-helix gap opening penalty (18)
//		gapOpenStrand	intra-strand gap opening penalty (18)
//		gapOpenOther	other gap opening penalty (6)
//		ss1		first SS "sequence" (i.e. composed of H/S/O)
//		ss2		second SS "sequence" (i.e. composed of H/S/O)
//	and returns the best score and the alignment.
//	The alignment is represented by a 2-tuple of strings,
//	where the first and second elements of the tuple
//	represent bases (and gaps) from seq1 and seq2 respectively.
//
//	Secondary-structure features are only enabled if the ssMatrix dictionary
//	is provided and is not None.  In that case the ssGapOpen penalties are
//	used and gapOpen is ignored.  The residue-matching score is a
//	combination of the secondary-structure and similarity scores, weighted
//	by the ssFraction.
static
PyObject *
align(PyObject *, PyObject *args, PyObject *kwdict)
{
	char *seq1, *seq2;
	PyObject *m;
	double gapOpen, gapExtend;
	char gapChar = '-';
	PyObject *ssM = NULL;
	double ssFraction = 0.3;
	double gapOpenHelix = 18.0;
	double gapOpenStrand = 18.0;
	double gapOpenOther = 18.0;
	char *ss1 = NULL;
	char *ss2 = NULL;
	static char *kwlist[] = { PY_STUPID "seq1", PY_STUPID "seq2",
		PY_STUPID "matrix", PY_STUPID "gapOpen", PY_STUPID "gapExtend",
		PY_STUPID "gapChar", PY_STUPID "ssMatrix",
		PY_STUPID "ssFraction", PY_STUPID "gapOpenHelix",
		PY_STUPID "gapOpenStrand", PY_STUPID "gapOpenOther",
		PY_STUPID "ss1", PY_STUPID "ss2",
		NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwdict,
			PY_STUPID "ssOdd|cOddddss",
			kwlist, &seq1, &seq2, &m, &gapOpen, &gapExtend,
			&gapChar, &ssM, &ssFraction, &gapOpenHelix,
			&gapOpenStrand, &gapOpenOther, &ss1, &ss2))
		return NULL;

	//
	// Convert Python similarity dictionary into C++ similarity map
	//
	Similarity matrix, ssMatrix;
	if (makeMatrix(m, matrix) < 0)
		return NULL;
	size_t rows = strlen(seq1) + 1;
	size_t cols = strlen(seq2) + 1;

	// handle secondary-structure setup if appropriate
	double *rowGapOpens = new double[rows];
	double *colGapOpens = new double[cols];
	bool doingSS = ssM != NULL && ss1 != NULL && ss2 != NULL
						&& ssM != Py_None;
	rowGapOpens[0] = colGapOpens[0] = 0;
	if (doingSS) {
		if (strlen(ss1) + 1 != rows || strlen(ss2) + 1 != cols) {
			PyErr_SetString(PyExc_ValueError, SeqLenMismatch);
			return NULL;
		}
		if (makeMatrix(ssM, ssMatrix) < 0)
			return NULL;
		size_t r, c;
		for (r = 1; r < rows; ++r) {
			char ssl = ss1[r-1];
			char ssr = ss1[r];
			if (ssl == 'H' && ssr == 'H')
				rowGapOpens[r] = gapOpenHelix;
			else if (ssl == 'S' && ssr == 'S')
				rowGapOpens[r] = gapOpenStrand;
			else
				rowGapOpens[r] = gapOpenOther;
		}
		for (c = 1; c < cols; ++c) {
			char ssl = ss2[c-1];
			char ssr = ss2[c];
			if (ssl == 'H' && ssr == 'H')
				colGapOpens[c] = gapOpenHelix;
			else if (ssl == 'S' && ssr == 'S')
				colGapOpens[c] = gapOpenStrand;
			else
				colGapOpens[c] = gapOpenOther;
		}
	} else {
		size_t r, c;
		for (r = 1; r < rows; ++r)
			rowGapOpens[r] = gapOpen;
		for (c = 1; c < cols; ++c)
			colGapOpens[c] = gapOpen;
		
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
	double bestScore = 0;
	int bestRow = 0, bestColumn = 0;
	for (size_t i = 1; i < rows; ++i) {
		for (size_t j = 1; j < cols; ++j) {
			//
			// Start with the matching score
			//
			Similarity::const_iterator it =
				matrixLookup(matrix, seq1[i - 1], seq2[j - 1]);
			if (it == matrix.end()) {
				char buf[80];
				(void) sprintf(buf, MissingKey, seq1[i - 1],
							seq2[j - 1]);
				PyErr_SetString(PyExc_KeyError, buf);
				return NULL;
			}
			double matchScore = (*it).second;
			if (doingSS) {
				Similarity::const_iterator it =
					matrixLookup(ssMatrix, ss1[i - 1],
								ss2[j - 1]);
				if (it == ssMatrix.end()) {
					char buf[80];
					(void) sprintf(buf, MissingSSKey,
							ss1[i - 1], ss2[j - 1]);
					PyErr_SetString(PyExc_KeyError, buf);
					return NULL;
				}
				matchScore = (1.0 - ssFraction) * matchScore
					+ ssFraction * (*it).second;
			}
			double best = H[i - 1][j - 1] + matchScore;
			int op = 0;

			//
			// Check if insertion is better
			//
			double go = colGapOpens[j];
			for (size_t k = 1; k < i; ++k) {
				double score = H[i - k][j] - go - k * gapExtend;
				if (score > best) {
					best = score;
					op = k;
				}
			}

			//
			// Check if deletion is better
			//
			go = rowGapOpens[i];
			for (size_t l = 1; l < j; ++l) {
				double score = H[i][j - l] - go - l * gapExtend;
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
			if (best > bestScore) {
				bestScore = best;
				bestRow = i;
				bestColumn = j;
			}
		}
	}

	//
	// Use the backtrack matrix to create the best alignment
	//
	std::string a1, a2;
	while (H[bestRow][bestColumn] > 0) {
		int op = bt[bestRow][bestColumn];
		if (op > 0) {
			for (int k = 0; k < op; ++k) {
				--bestRow;
				a1.append(1, seq1[bestRow]);
				a2.append(1, gapChar);
			}
		}
		else if (op == 0) {
			--bestRow;
			--bestColumn;
			a1.append(1, seq1[bestRow]);
			a2.append(1, seq2[bestColumn]);
		}
		else {
			op = -op;
			for (int k = 0; k < op; ++k) {
				--bestColumn;
				a1.append(1, gapChar);
				a2.append(1, seq2[bestColumn]);
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
	delete [] rowGapOpens;
	delete [] colGapOpens;

	//
	// Return our results
	//
	return Py_BuildValue(PY_STUPID "fO", bestScore, alignment);
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
"	gapOpen		gap opening penalty\n"
"	gapExtend	gap extension penalty\n"
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
"	gapOpen		gap opening penalty\n"
"	gapExtend	gap extension penalty\n"
"and the following optional keyword arguments:\n"
"	gapChar		character used in gaps (default: '-')\n"
"	ssMatrix	secondary-structure scoring dictionary (NULL)\n"
"	ssFraction	fraction of weight given to SS scoring (0.3)\n"
"	gapOpenHelix	intra-helix gap opening penalty (18)\n"
"	gapOpenStrand	intra-strand gap opening penalty (18)\n"
"	gapOpenOther	other gap opening penalty (6)\n"
"	ss1		first SS \"sequence\" (i.e. composed of H/S/O)\n"
"	ss2		second SS \"sequence\" (i.e. composed of H/S/O)\n"
"and returns the best score and the alignment.\n"
"The alignment is represented by a 2-tuple of strings,\n"
"where the first and second elements of the tuple\n"
"represent bases (and gaps) from seq1 and seq2 respectively.\n"
"\n"
"Secondary-structure features are only enabled if the ssMatrix dictionary\n"
"is provided and is not None.  In that case the ssGapOpen penalties are\n"
"used and gapOpen is ignored.  The residue-matching score is a\n"
"combination of the secondary-structure and similarity scores, weighted\n"
"by the ssFraction."
MATRIX_EXPLAIN;

static PyMethodDef sw_methods[] = {
	{ PY_STUPID "score", score,	METH_VARARGS, PY_STUPID docstr_score	},
	{ PY_STUPID "align", (PyCFunction)align, METH_VARARGS|METH_KEYWORDS, PY_STUPID docstr_align	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef sw_def =
{
	PyModuleDef_HEAD_INIT,
	"_sw",
	"Smith-Waterman alignment methods",
	-1,
	sw_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__sw()
{
	return PyModule_Create(&sw_def);
}
