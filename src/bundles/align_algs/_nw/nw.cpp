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

#if 0
const char *MissingKey = "no score for '%c' vs. '%c'";
const char *MissingSSKey = "no score for gap open between '%c' and '%c'";
const char *SeqLenMismatch = "sequence lengths don't match their secondary structure strings";
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
		for (char* ss_ptr = ss_types; *ss_ptr != '\0'; ++ss_ptr, ++i)
			freq_dicts[i][*ss_ptr] = 1.0;
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
	if (!PyArg_ParseTuple(args, PY_STUPID "ssddddpOOOOOpdddOOssOO", &seq1, &seq2,
			&score_match, &score_mismatch, &gap_open, &gap_extend, &ends_are_gaps,
			&sim_m, &score_m, &freq_m, &ss_m, &ss_fract,
			&ss_specific_gaps, &gap_open_helix, &gap_open_strand, &gap_open_other,
			&gap_freqs1, &gap_freqs2,
			&ss_types1, &ss_types2, &ss_freqs_val1, &ss_freqs_val2))
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
	std::vector<size_t> col_gap_starts(0, cols-1); // don't care about column zero
	for (size_t i1 = 0; i1 < rows-1; ++i1) {
		size_t row_gap_pos = 0;
		size_t i2_end = cols-1;
		for (size_t i2 = 0; i2 < i2_end; ++i2) {
			auto best = m[i1][i2] + eval->score(i1, i2);
			int bt_type = 0;
			if (i2 + 1 < cols-1 || ends_are_gaps) {
				auto col_gap_pos = col_gap_starts[i2];
				int skip_size = i1 + 1 - col_gap_pos;
				//TODO: occupancy
			} //TODO
		}
	}

	//TODO: dynamic programming; create match list; complete module; test; add doc strings to Python layer

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
#if 0
    //# if 'score_matrix', 'similarity_matrix', or 'frequency_matrix' is
    //# provided, then 'score_match' and 'score_mismatch' are ignored and
    //# the matrix is used to evaluate matching between the sequences.
    //# 'score_matrix' should be a two-dimensional array of size
    //# len(s1) x len(s2).  'similarity_matrix' should be a dictionary
    //# keyed with two-tuples of residue types.  'frequency_matrix' should
    //# be a list of length s2 of dictionaries, keyed by residue type.
    //#
    //# if 'ss_fraction' is not None/False, then 'ss_matrix' should be a 3x3
    //# matrix keyed with 2-tuples of secondary structure types ('H': helix,
    //# 'S': strand, 'O': other).  The score will be a mixture of the
    //# ss/similarity matrix scores weighted by the ss_fraction
    //# [ss_fraction * ss score + (1 - ss_fraction) * similarity score]
    //#
    //# if 'gap_open_helix/Strand/Other' is not None and 'ss_fraction' is not
    //# None/False, then score_gap_open is ignored when an intra-helix/
    //# intra-strand/other gap is opened and the appropriate penalty
    //# is applied instead
    //#
    //# if 'return_seqs' is True, then instead of returning a match list
    //# (a list of two-tuples) as the second value, a two-tuple of gapped
    //# Sequences will be returned.  In both cases, the first return value
    //# is the match score.
    m = []
    bt = []
    for i1 in range(len(s1) + 1):
        m.append((len(s2) + 1) * [ 0 ])
        bt.append((len(s2) + 1) * [None])
        bt[i1][0] = 1
        if ends_are_gaps and i1 > 0:
            m[i1][0] = score_gap_open + i1 * score_gap
    for i2 in range(len(s2) + 1):
        bt[0][i2] = 2
        if ends_are_gaps and i2 > 0:
            m[0][i2] = score_gap_open * i2 * score_gap

    if similarity_matrix is not None:
        evaluate = lambda i1, i2: similarity_matrix[(s1[i1], s2[i2])]
    elif score_matrix is not None:
        evaluate = lambda i1, i2: score_matrix[i1][i2]
    elif frequency_matrix is not None:
        evaluate = lambda i1, i2: frequency_matrix[i2][s1[i1]]
    else:
        def evaluate(i1, i2):
            if s1[i1] == s2[i2]:
                return score_match
            return score_mismatch
    doing_ss =  ss_fraction is not None and ss_fraction is not False and ss_matrix is not None
    if doing_ss:
        prev_eval = evaluate
        sim_fraction = 1.0 - ss_fraction
        # prevent slow ss_type() call in inner loop...
        ss_types1 = [ s1.ss_type(i) for i in range(len(s1))]
        ss_types2 = [ s2.ss_type(i) for i in range(len(s2))]
        def ss_eval(i1, i2):
            if hasattr(s1, 'ss_freqs'):
                freqs1 = s1.ss_freqs[i1]
            else:
                freqs1 = {ss_types1[i1]: 1.0}
            if hasattr(s2, 'ss_freqs'):
                freqs2 = s2.ss_freqs[i2]
            else:
                freqs2 = {ss_types2[i2]: 1.0}
            val = 0.0
            for ss1, freq1 in freqs1.items():
                if ss1 == None:
                    continue
                for ss2, freq2 in freqs2.items():
                    if ss2 == None:
                        continue
                    val += freq1 * freq2 * ss_matrix[(ss1, ss2)]
            return val
        evaluate = lambda i1, i2: ss_fraction * ss_eval(i1, i2) + sim_fraction * prev_eval(i1, i2)

    # precompute appropriate gap-open penalties
    gap_open_1 = [score_gap_open] * (len(s1)+1)
    gap_open_2 = [score_gap_open] * (len(s2)+1)
    if ends_are_gaps:
        if gap_open_other is not None:
            gap_open_1[0] = gap_open_2[0] = gap_open_other
    else:
            gap_open_1[0] = gap_open_2[0] = 0
    if doing_ss and gap_open_other != None:
        for seq, gap_opens in [(s1, gap_open_1), (s2, gap_open_2)]:
            if hasattr(seq, 'gap_freqs'):
                for i, gap_freq in enumerate(seq.gap_freqs):
                    gap_opens[i+1] = \
                        gap_freq['H'] * gap_open_helix + \
                        gap_freq['S'] * gap_open_strand + \
                        gap_freq['O'] * gap_open_other
            else:
                ss_type = [seq.ss_type(i)
                        for i in range(len(seq))]
                for i, ss in enumerate(ss_type[:-1]):
                    nextSS = ss_type[i+1]
                    if ss == nextSS and ss == 'H':
                        gap_opens[i+1] = gap_open_helix
                    elif ss == nextSS and ss == 'S':
                        gap_opens[i+1] = gap_open_strand
                    else:
                        gap_opens[i+1] = gap_open_other

    col_gap_starts = [0] * len(s2) # don't care about column zero
    for i1 in range(len(s1)):
        row_gap_pos = 0
        for i2 in range(len(s2)):
            best = m[i1][i2] + evaluate(i1, i2)
            bt_type = 0
            if i2 + 1 < len(s2) or ends_are_gaps:
                col_gap_pos = col_gap_starts[i2]
                skip_size = i1 + 1 - col_gap_pos
                if hasattr(s1, "occupancy"):
                    tot_occ = 0.0
                    for i in range(col_gap_pos, i1+1):
                        tot_occ += s1.occupancy[i]
                    col_skip_val = tot_occ * score_gap
                else:
                    col_skip_val = skip_size * score_gap
                base_col_gap_val = m[col_gap_pos][i2+1] + col_skip_val
                skip = base_col_gap_val + gap_open_2[i2+1]
            else:
                skip_size = 1
                col_skip_val = 0
                skip = m[i1][i2+1]
            if skip > best:
                best = skip
                bt_type = skip_size
            if i1 + 1 < len(s1) or ends_are_gaps:
                skip_size = i2 + 1 - row_gap_pos
                if hasattr(s2, "occupancy"):
                    tot_occ = 0.0
                    for i in range(row_gap_pos, i2+1):
                        tot_occ += s2.occupancy[i]
                    row_skip_val = tot_occ * score_gap
                else:
                    row_skip_val = skip_size * score_gap
                base_row_gap_val = m[i1+1][row_gap_pos] + row_skip_val
                skip = base_row_gap_val + gap_open_1[i1+1]
            else:
                skip_size = 1
                row_skip_val = 0
                skip = m[i1+1][i2]
            if skip > best:
                best = skip
                bt_type = 0 - skip_size
            m[i1+1][i2+1] = best
            bt[i1+1][i2+1] = bt_type
            if bt_type >= 0:
                # not gapping the row
                if best > base_row_gap_val:
                    row_gap_pos = i2 + 1
            if bt_type <= 0:
                # not gapping the column
                if best > base_col_gap_val:
                    col_gap_starts[i2] = i1 + 1
    """
    if debug:
        from chimera.selection import currentResidues
        cr = currentResidues(asDict=True)
        if cr:
            for fileName, matrix in [("scores", m), ("trace", bt)]:
                out = open("/home/socr/a/pett/rm/" + fileName,
                                    "w")
                print>>out, "    ",
                for i2, r2 in enumerate(s2.residues):
                    if r2 not in cr:
                        continue
                    print>>out, "%5d" % i2,
                print>>out
                print>>out, "    ",
                for i2, r2 in enumerate(s2.residues):
                    if r2 not in cr:
                        continue
                    print>>out, "%5s" % s2[i2],
                print>>out
                for i1, r1 in enumerate(s1.residues):
                    if r1 not in cr:
                        continue
                    print>>out, "%3d" % i1, s1[i1],
                    for i2, r2 in enumerate(s2.residues):
                        if r2 not in cr:
                            continue
                        print>>out, "%5g" % (
                            matrix[i1+1][i2+1]),
                    print>>out
                out.close()
    """
    i1 = len(s1)
    i2 = len(s2)
    match_list = []
    while i1 > 0 and i2 > 0:
        bt_type = bt[i1][i2]
        if bt_type == 0:
            match_list.append((i1-1, i2-1))
            i1 = i1 - 1
            i2 = i2 - 1
        elif bt_type > 0:
            i1 = i1 - bt_type
        else:
            i2 = i2 + bt_type
--- end Python ---
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
#endif
}

}

#if 0
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
"useful for optimizing similarity matrices.\n\n"
SIM_MATRIX_EXPLAIN;

static const char* docstr_align =
"align\n"
"Compute the best Smith-Waterman score and alignment\n"
"\n"
"The function takes five mandatory arguments:\n"
"	seq1		first sequence\n"
"	seq2		second sequence\n"
"	matrix		similarity dictionary (see below)\n"
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
"by the ss_fraction.\n\n"
SIM_MATRIX_EXPLAIN;

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
#endif
