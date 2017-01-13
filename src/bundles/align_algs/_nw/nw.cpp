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
#include <sstream>
#include <string>
#include <map>
#include <vector>

#include <align_algs/support.h>
#include <arrays/pythonarray.h>
#include <pysupport/convert.h>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

using align_algs::make_matrix;
using align_algs::matrix_lookup;
using align_algs::Similarity;

typedef std::map<char, double> FreqDict;
typedef std::vector<FreqDict>  FreqDicts;

class Evaluator
{
public:
	virtual  ~Evaluator() {}
	virtual double  score(int i, int j) = 0;
};

class SimEvaluator: public Evaluator
{
	Similarity  _sim_m;
	char*  _seq1;
	char*  _seq2;
public:
	SimEvaluator(PyObject* py_m, char* seq1, char* seq2) {
		if (make_matrix(py_m, _sim_m) < 0)
			throw std::invalid_argument("bad similarity matrix");
		_seq1 = seq1;
		_seq2 = seq2;
	}
	double  score(int i, int j)  { return matrix_lookup(_sim_m, _seq1[i], _seq2[j])->second; }
};

class ScoreEvaluator: public Evaluator
{
	double**  _score_m;
public:
	ScoreEvaluator(double** score_m): _score_m(score_m) {}
	double  score(int i, int j)  { return _score_m[i][j]; }
};

class FreqEvaluator: public Evaluator
{
	FreqDicts  _freq_dicts;
	char*  _seq2;
public:
	FreqEvaluator(char* seq2): _seq2(seq2) {}
	FreqDicts&  freq_dicts() { return _freq_dicts; }
	double  score(int i, int j)  { return _freq_dicts[i][_seq2[j]]; }
};

class SimpleEvaluator: public Evaluator
{
	char*  _seq1;
	char*  _seq2;
	double  _match, _mismatch;
public:
	SimpleEvaluator(char* seq1, char* seq2, double match, double mismatch):
		_seq1(seq1), _seq2(seq2), _match(match), _mismatch(mismatch) {}
	double  score(int i, int j)  { return (_seq1[i] == _seq2[j]) ? _match : _mismatch; }
};

class SSEvaluator: public Evaluator
{
	char*  _ss_types1;
	char*  _ss_types2;
	FreqDicts  _freq_dicts1;
	FreqDicts  _freq_dicts2;
	double  _ss_fraction;
	Evaluator*  _prev_eval;
	Similarity  _ss_m;

	double  _ss_eval(int i, int j) {
		double val = 0.0;
		for (auto& ss_freq_1: _freq_dicts1[i]) {
			auto ss1 = ss_freq_1.first;
			auto freq1 = ss_freq_1.second;
			for (auto& ss_freq_2: _freq_dicts2[j]) {
				auto ss2 = ss_freq_2.first;
				auto freq2 = ss_freq_2.second;
				val += freq1 * freq2 * matrix_lookup(_ss_m, ss1, ss2)->second;
			}
		}
		return val;
	}
public:
	SSEvaluator(char* ss_types1, char* ss_types2, double ss_fraction, Evaluator* prev_eval,
			PyObject* py_ss_m) {
		if (make_matrix(py_ss_m, _ss_m) < 0)
			throw std::invalid_argument("bad SS matrix");
		_ss_types1 = ss_types1;
		_ss_types2 = ss_types2;
		_ss_fraction = ss_fraction;
		_prev_eval = prev_eval;
	}
	~SSEvaluator() { delete _prev_eval; }
	FreqDicts&  freq_dicts1() { return _freq_dicts1; }
	FreqDicts&  freq_dicts2() { return _freq_dicts2; }
	double  score(int i, int j) {
		return _ss_fraction * _ss_eval(i, j) + (1.0 - _ss_fraction) * _prev_eval->score(i, j);
	}
};

extern "C" {

static bool
extract_py_freq_dict(PyObject* dict, FreqDict& fdict, const char* descript, Py_ssize_t i)
{
	PyObject *key, *value;
	Py_ssize_t pos = 0;
	while (PyDict_Next(dict, &pos, &key, &value)) {
		try {
			std::stringstream err_msg;
			err_msg << "residue character in " << descript << " dictionary at index " << i;
			auto cstring = pysupport::pystring_to_cchar(key, err_msg.str().c_str());
			if (cstring[0] == '\0' || cstring[1] != '\0') {
				err_msg << descript << " matrix dictionary at index " << i
					<< " has a key that is not a single character: '" << cstring << "'";
				throw pysupport::PySupportError(err_msg.str().c_str());
			}
			if (!PyFloat_Check(value)) {
				err_msg << descript << " matrix dictionary at index " << i << " with key "
					<< cstring << " has a value that is not a float";
				throw pysupport::PySupportError(err_msg.str().c_str());
			}
			fdict[cstring[0]] = PyFloat_AS_DOUBLE(value);
		} catch (pysupport::PySupportError& e) {
			PyErr_SetString(PyExc_ValueError, e.what());
			return false;
		}
	}
	return true;
}

static bool
populate_freqs_dict(FreqDicts& freq_dicts, PyObject* ss_freqs_val, char* ss_types)
{
	Py_ssize_t size = strlen(ss_types);
	freq_dicts.resize(size);
	if (ss_freqs_val == Py_None) {
		size_t i = 0;
		for (char* ss_ptr = ss_types; *ss_ptr != '\0'; ++ss_ptr, ++i) {
			freq_dicts[i][*ss_ptr] = 1.0;
		}
	} else {
		if (!PySequence_Check(ss_freqs_val)) {
			PyErr_SetString(PyExc_ValueError, "SS frequencies is not a list/tuple or None");
			return false;
		}
		if (PySequence_Size(ss_freqs_val) != size) {
			PyErr_SetString(PyExc_ValueError,
				"SS frequencies list/tuple is not same length as sequence");
			return false;
		}
		for (Py_ssize_t i = 0; i < size; ++i) {
			auto& freq_dict = freq_dicts[i];
			PyObject* py_freq_dict = PySequence_GetItem(ss_freqs_val, i);
			if (!PyDict_Check(py_freq_dict)) {
				std::stringstream err_msg;
				err_msg << "SS frequencies item at index " << i << " is not a dictionary";
				PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
				return false;
			}
			if (!extract_py_freq_dict(py_freq_dict, freq_dict, "SS frequency", i))
				return false;
		}
	}
	return true;
}

//
// match
//	Compute the best Needleman-Wunsch score and match list
//
static PyObject *
match(PyObject *, PyObject *args)
{
	char *seq1, *seq2;
	double score_match, score_mismatch;
	double gap_open, gap_extend;
	PyObject *score_m, *sim_m, *ss_m, *freq_m;
	int ends_are_gaps;
	PyObject* ss_fract;
	int ss_specific_gaps;
	double gap_open_helix;
	double gap_open_strand;
	double gap_open_other;
	PyObject* gap_freqs1;
	PyObject* gap_freqs2;
	char *ss_types1, *ss_types2;
	PyObject *ss_freqs_val1, *ss_freqs_val2;
	PyObject *occupancies_val1, *occupancies_val2;
	if (!PyArg_ParseTuple(args, PY_STUPID "ssddddpOOOOOpdddOOssOOOO", &seq1, &seq2,
			&score_match, &score_mismatch, &gap_open, &gap_extend, &ends_are_gaps,
			&sim_m, &score_m, &freq_m, &ss_m, &ss_fract,
			&ss_specific_gaps, &gap_open_helix, &gap_open_strand, &gap_open_other,
			&gap_freqs1, &gap_freqs2,
			&ss_types1, &ss_types2, &ss_freqs_val1, &ss_freqs_val2,
			&occupancies_val1, &occupancies_val2))
		return nullptr;

	size_t rows = strlen(seq1) + 1;
	size_t cols = strlen(seq2) + 1;

	// create cell-evaluation function
	Evaluator *eval;
	auto array = Numeric_Array();
	if (sim_m != Py_None) {
		try {
			eval = new SimEvaluator(sim_m, seq1, seq2);
		} catch (std::invalid_argument& e) {
			return nullptr;
		}
	} else if (score_m != Py_None) {
		if (!array_from_python(score_m, 2, Numeric_Array::Double, &array, false))
			return nullptr;
		auto dims = array.sizes();
		if (static_cast<size_t>(dims[0]) != rows-1 || static_cast<size_t>(dims[1]) != cols-1) {
			std::stringstream err_msg;
			err_msg << "Score array is not of dimension (len(seq1) x len(seq2));"
				<< " Is (" << dims[0] << " x " << dims[1] << "),"
				<< " should be (" << rows-1 << " x " << cols-1 << ")";
			PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
			return nullptr;
		}
		eval = new ScoreEvaluator(static_cast<double**>(array.values()));
	} else if (freq_m != Py_None) {
		if (!PySequence_Check(freq_m)) {
			PyErr_SetString(PyExc_ValueError, "Frequency matrix is not a sequence");
			return nullptr;
		}
		if (static_cast<size_t>(PySequence_Size(freq_m)) != cols-1) {
			std::stringstream err_msg;
			err_msg << "Frequency matrix length (" << PySequence_Size(freq_m)
				<< ") not the same as length of second sequence (" << cols-1 << ")";
			PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
			return nullptr;
		}
		eval = new FreqEvaluator(seq2);
		auto& freq_dicts = static_cast<FreqEvaluator*>(eval)->freq_dicts();
		freq_dicts.resize(cols-1);
		Py_ssize_t end = cols-1;
		for (Py_ssize_t i = 0; i < end; ++i) {
			auto dict = PySequence_GetItem(freq_m, i);
			if (!PyDict_Check(dict)) {
				std::stringstream err_msg;
				err_msg << "Item at index " << i << " in frequency matrix is not a dictionary";
				PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
				delete eval;
				return nullptr;
			}
			auto& fdict = freq_dicts[i];
			if (!extract_py_freq_dict(dict, fdict, "frequency", i)) {
				delete eval;
				return nullptr;
			}
		}
	} else {
		eval = new SimpleEvaluator(seq1, seq2, score_match, score_mismatch);
	}

	// if including secondary structure scoring, mix in above scoring
	bool doing_ss = ss_fract != Py_None && ss_fract != Py_False && ss_m != Py_None;
	if (doing_ss) {
		if (!PyFloat_Check(ss_fract)) {
			PyErr_SetString(PyExc_ValueError, "ss_fraction is not None, False, or a float");
			delete eval;
			return nullptr;
		}
		auto ss_fraction = PyFloat_AS_DOUBLE(ss_fract);
		if (strlen(ss_types1) != rows-1 || strlen(ss_types2) != cols-1) {
			PyErr_SetString(PyExc_ValueError,
				"ss_types length does not match corresponding sequence");
			delete eval;
			return nullptr;
		}

		try {
			eval = new SSEvaluator(ss_types1, ss_types2, ss_fraction, eval, ss_m);
		} catch (std::invalid_argument& e) {
			delete eval;
			return nullptr;
		}
		auto& freq_dicts1 = static_cast<SSEvaluator*>(eval)->freq_dicts1();
		auto& freq_dicts2 = static_cast<SSEvaluator*>(eval)->freq_dicts2();
		if (!populate_freqs_dict(freq_dicts1, ss_freqs_val1, ss_types1)
		|| !populate_freqs_dict(freq_dicts2, ss_freqs_val2, ss_types2)) {
			delete eval;
			return nullptr;
		}
	}

    // precompute appropriate gap-open penalties
	std::vector<double> gap_open_1(rows, gap_open);
	std::vector<double> gap_open_2(cols, gap_open);
	if (ends_are_gaps) {
		if (ss_specific_gaps)
			gap_open_1[0] = gap_open_2[0] = gap_open_other;
	} else
		gap_open_1[0] = gap_open_2[0] = 0.0;
	if (doing_ss && ss_specific_gaps) {
		for (int si = 1; si <= 2 ; ++si) {
			char* seq;
			PyObject* py_gap_freqs;
			char* ss_types;
			size_t dim;
			std::vector<double>* gap_opens;
			if (si == 1) {
				seq = seq1;
				py_gap_freqs = gap_freqs1;
				ss_types = ss_types1;
				dim = rows;
				gap_opens = &gap_open_1;
			} else {
				seq = seq2;
				py_gap_freqs = gap_freqs2;
				ss_types = ss_types2;
				dim = cols;
				gap_opens = &gap_open_2;
			}
			if (py_gap_freqs == Py_None) {
				char* ss;
				char* next_ss;
				size_t i;
				for (i = 0, ss = ss_types, next_ss = ss_types+1; *next_ss != '\0';
						++i, ++ss, ++next_ss) {
					if (*ss == *next_ss && *ss == 'H')
						(*gap_opens)[i+1] = gap_open_helix;
					else if (*ss == *next_ss && *ss == 'S')
						(*gap_opens)[i+1] = gap_open_strand;
					else
						(*gap_opens)[i+1] = gap_open_other;
				}
			} else {
				FreqDicts freq_dicts;
				if (!populate_freqs_dict(freq_dicts, py_gap_freqs, ss_types)) {
					delete eval;
					return nullptr;
				}
				for (FreqDicts::size_type i = 0; i < freq_dicts.size(); ++i) {
					auto& gap_freq = freq_dicts[i];
					(*gap_opens)[i+1] = \
						gap_freq['H'] * gap_open_helix + \
						gap_freq['S'] * gap_open_strand + \
						gap_freq['O'] * gap_open_other;
				}
				
			}
		}
	}

	// extract Python occupancy info
	bool has_occ1, has_occ2;
	std::vector<double> occ1, occ2;
	if (occupancies_val1 == Py_None) {
		has_occ1 = false;
	} else {
		has_occ1 = true;
		try {
			pysupport::pylist_of_float_to_cvec(occupancies_val1, occ1, "seq1 occupancy");
		} catch (pysupport::PySupportError& e) {
			PyErr_SetString(PyExc_ValueError, e.what());
			delete eval;
			return nullptr;
		}
	}
	if (occupancies_val2 == Py_None) {
		has_occ2 = false;
	} else {
		has_occ2 = true;
		try {
			pysupport::pylist_of_float_to_cvec(occupancies_val2, occ2, "seq2 occupancy");
		} catch (pysupport::PySupportError& e) {
			PyErr_SetString(PyExc_ValueError, e.what());
			delete eval;
			return nullptr;
		}
	}

	//
	// Allocate space for the score matrix and the backtracking matrix
	//
	double **m = new double *[rows];
	int **bt = new int *[rows];
	for (size_t i = 0; i < rows; ++i) {
		m[i] = new double[cols];
		bt[i] = new int[cols];
		bt[i][0] = 1;
		if (ends_are_gaps && i > 0)
			m[i][0] = gap_open + i * gap_extend;
		else
			m[i][0] = 0.0;
	}
	for (size_t j = 0; j < cols; ++j) {
		bt[0][j] = 2;
		if (ends_are_gaps && j > 0)
			m[0][j] = gap_open + j * gap_extend;
		else
			m[0][j] = 0.0;
	}

	// fill matrix [dynamic programming]
	std::vector<size_t> col_gap_starts(cols-1, 0); // don't care about column zero
	double base_col_gap_val, base_row_gap_val, skip;
	for (size_t i1 = 0; i1 < rows-1; ++i1) {
		size_t row_gap_pos = 0;
		size_t i2_end = cols-1;
		for (size_t i2 = 0; i2 < i2_end; ++i2) {
			auto best = m[i1][i2] + eval->score(i1, i2);
			int bt_type = 0, skip_size;
			double col_skip_val, row_skip_val;
			if (i2 + 1 < cols-1 || ends_are_gaps) {
				auto col_gap_pos = col_gap_starts[i2];
				skip_size = i1 + 1 - col_gap_pos;
				if (has_occ1) {
					double tot_occ = 0.0;
					for (auto i = col_gap_pos; i < i1+1; ++i) {
						tot_occ += occ1[i];
					}
					col_skip_val = tot_occ * gap_extend;
				} else {
					col_skip_val = skip_size * gap_extend;
				}
				base_col_gap_val = m[col_gap_pos][i2+1] + col_skip_val;
				skip = base_col_gap_val + gap_open_2[i2+1];
			} else {
				skip_size = 1;
				col_skip_val = 0.0;
				skip = m[i1][i2+1];
			}
			if (skip > best) {
				best = skip;
				bt_type = skip_size;
			}
			if (i1 + 1 < rows-1 || ends_are_gaps) {
				skip_size = i2 + 1 - row_gap_pos;
				if (has_occ2) {
					double tot_occ = 0.0;
					for (auto i = row_gap_pos; i < i2+1; ++i) {
						tot_occ += occ2[i];
					}
					row_skip_val = tot_occ * gap_extend;
				} else {
					row_skip_val = skip_size * gap_extend;
				}
				base_row_gap_val = m[i1+1][row_gap_pos] + row_skip_val;
				skip = base_row_gap_val + gap_open_1[i1+1];
			} else {
				skip_size = 1;
				row_skip_val = 0.0;
				skip = m[i1+1][i2];
			}
			if (skip > best) {
				best = skip;
				bt_type = 0 - skip_size;
			}

			m[i1+1][i2+1] = best;
			bt[i1+1][i2+1] = bt_type;
			if (bt_type >= 0) {
				// not gapping the row
				if (best > base_row_gap_val)
					row_gap_pos = i2 + 1;
			}
			if (bt_type <= 0) {
				// not gapping the column
				if (best > base_col_gap_val)
					col_gap_starts[i2] = i1 + 1;
			}
		}
	}

	// create match list
	bool py_error_happened = false;
	PyObject* match_list = PyList_New(0);
	if (match_list == nullptr) {
		py_error_happened = true;
	} else {
		auto i1 = rows - 1;
		auto i2 = cols - 1;
		while (i1 > 0 && i2 > 0) {
			auto bt_type = bt[i1][i2];
			if (bt_type == 0) {
				PyObject* tuple = PyTuple_New(2);
				if (tuple == nullptr) {
					py_error_happened = true;
					Py_DECREF(match_list);
					break;
				}
				if (PyList_Append(match_list, tuple) < 0) {
					py_error_happened = true;
					Py_DECREF(tuple);
					Py_DECREF(match_list);
					break;
				}
				PyObject* py_i1 = PyLong_FromSize_t(i1-1);
				if (py_i1 == nullptr) {
					py_error_happened = true;
					Py_DECREF(match_list);
					break;
				}
				PyTuple_SET_ITEM(tuple, 0, py_i1);
				PyObject* py_i2 = PyLong_FromSize_t(i2-1);
				if (py_i2 == nullptr) {
					py_error_happened = true;
					Py_DECREF(match_list);
					break;
				}
				PyTuple_SET_ITEM(tuple, 1, py_i2);
				i1--;
				i2--;
			} else if (bt_type > 0) {
				i1 -= bt_type;
			} else {
				i2 += bt_type;
			}
		}
	}

	auto best_score = m[rows-1][cols-1];

	//
	// Release the score and backtrack matrix memory
	//
	for (size_t i = 0; i < rows; ++i) {
		delete [] m[i];
		delete [] bt[i];
	}
	delete [] m;
	delete [] bt;
	delete eval;

	if (py_error_happened)
		return nullptr;

	return Py_BuildValue(PY_STUPID "fO", best_score, match_list);
}

}

static const char* docstr_match =
"Private function for computing Needleman-Wunsch score and matching.\n"
"Use the chimerax.seqalign.align_algs.NeedlemanWunsch.nw method, which uses this as a backend.";

static PyMethodDef nw_methods[] = {
	{ PY_STUPID "match", match,	METH_VARARGS, PY_STUPID docstr_match	},
	{ nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef nw_def =
{
	PyModuleDef_HEAD_INIT,
	"_nw",
	"Needleman-Wunsch alignment method",
	-1,
	nw_methods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

PyMODINIT_FUNC
PyInit__nw()
{
	return PyModule_Create(&nw_def);
}
