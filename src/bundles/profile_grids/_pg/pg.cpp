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
#include <algorithm>  // std::min
#include <arrays/pythonarray.h>	// use python_float_array
#include <cctype>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <pysupport/convert.h>
#include <set>
#include <sstream>
#include <thread>
#include <typeinfo>
#include <vector>

typedef std::vector<std::vector<char>> SequencesType;
typedef std::vector<float> WeightsType;
typedef std::vector<std::vector<float>> ProfileType;

// 29 == 26 letters, '?', gap, plus "other"
const int NUM_CATEGORIES = 29;

static void
initiate_compute_profile(SequencesType* sequences, int start, int end, WeightsType* weights,
	ProfileType* profile, std::mutex* profile_mutex)
{
	for (auto col = start; col != end; ++col) {
		profile_mutex->lock();
		auto& counts = (*profile)[col];
		profile_mutex->unlock();
		counts.resize(NUM_CATEGORIES);
		size_t num_seqs = sequences->size();
		for (size_t i = 0; i < num_seqs; ++i) {
			auto& seq = (*sequences)[i];
			auto weight = (*weights)[i];
			auto c = seq[col];
			if (std::isalpha(c))
				counts[std::toupper(c) - 'A'] += weight;
			else if (c == '?')
				counts[26] += weight;
			else if (std::ispunct(c))
				counts[27] += weight;
			else
				counts[28] += weight;
		}
	}
}

static PyObject*
make_profile(ProfileType& profile)
{
	float* data_ptr;
	PyObject* py_profile = python_float_array(profile.size(), NUM_CATEGORIES, &data_ptr);
	for (auto col_data: profile)
		for (auto item: col_data)
			*data_ptr++ = item;
	return py_profile;
}

extern "C" {

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

static
PyObject *
compute_profile(PyObject *, PyObject *args)
{
	PyObject*  py_seqs;
	PyObject*  py_weights;
	unsigned int  num_cpus;
	if (!PyArg_ParseTuple(args, PY_STUPID "OOI", &py_seqs, &py_weights, &num_cpus))
		return nullptr;
	if (!PyList_Check(py_seqs)) {
		PyErr_SetString(PyExc_TypeError, "1st argument must be a list of strings (sequences)");
		return nullptr;
	}
	if (!PyList_Check(py_weights)) {
		PyErr_SetString(PyExc_TypeError, "2nd argument must be a list of numeric sequence weights");
		return nullptr;
	}

	SequencesType sequences;
	try {
		pysupport::pylist_of_string_to_cvec_of_cvec(py_seqs, sequences, "sequences");
	} catch (pysupport::PySupportError& pse) {
		PyErr_SetString(PyExc_TypeError, pse.what());
		return nullptr;
	}
	auto seq_len = sequences[0].size();
	for (auto &seq: sequences)
		if (seq.size() != seq_len) {
			PyErr_SetString(PyExc_TypeError, "Not all sequences in alignment are same length");
			return nullptr;
		}

	WeightsType weights;
	try {
		pysupport::pylist_of_int_to_cvec(py_weights, weights, "sequence weights");
	} catch (pysupport::PySupportError& pse) {
		PyErr_SetString(PyExc_TypeError, pse.what());
		return nullptr;
	}
	if (sequences.size() != weights.size()) {
		PyErr_SetString(PyExc_TypeError, "Number of sequences and sequence weights differ");
		return nullptr;
	}

	ProfileType profile;
	profile.resize(seq_len);
	std::mutex profile_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the columns among the threads;
	// letting the threads take columns from a global pool
	// likely results in too much lock contention
	num_threads = std::min(num_threads, seq_len);
	if (num_threads > 0) {
		float per_thread = seq_len / (float) num_threads;
		int start = 0;
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)((i+1) * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = seq_len;
			threads.push_back(std::thread(initiate_compute_profile, &sequences, start, end, &weights,
				&profile, &profile_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_profile(profile);
}

}

static const char* docstr_compute_profile = "compute_profile\n"
"Compute a composition profile of a sequence alignment";

static PyMethodDef pg_methods[] = {
	{ PY_STUPID "compute_profile", compute_profile,	METH_VARARGS, PY_STUPID docstr_compute_profile	},
	{ nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef pg_def =
{
	PyModuleDef_HEAD_INIT,
	"_profile_grids",
	"Compute alignment profile",
	-1,
	pg_methods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

PyMODINIT_FUNC
PyInit__profile_grids()
{
	return PyModule_Create(&pg_def);
}
