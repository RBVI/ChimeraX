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
#include <atomic/Atom.h>
#include <pysupport/convert.h>
#include <vector>

class Atom_Condition
{
public:
	virtual  ~Atom_Condition() {}
	virtual bool  evaluate(const Atom* a) const;
};

class CG_Condition
{
public:
	Atom_Condition*  atom_cond;
	std::vector<CG_Condition*>  bonded;

	virtual  ~CG_Condition() { delete atom_cond; for (auto cond: bonded) delete cond; }
	bool  evaluate(const Atom* a, const Atom* parent = nullptr) const;
};

CG_Condition*
make_condition(PyObject* group_rep)
{
	
}

extern "C" {

//TODO: don't forget to parallize on per-atom basis

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

static
PyObject *
find_group(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	PyObject*  py_group_rep;
	PyObject*  py_group_principals;
	unsigned int  num_cpus;
	if (!PyArg_ParseTuple(args, PY_STUPID "OOOI", &py_struct_ptr, &py_group_rep,
			&py_group_principals, &num_cpus))
		return NULL;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return NULL;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));
	if (!PyList_Check(py_group_rep)) {
		PyErr_SetString(PyExc_TypeError, "group_rep must be a list!");
		return NULL;
	}
	if (!PyList_Check(py_group_principals) || PyList_Size(py_group_principals) != 2) {
		PyErr_SetString(PyExc_TypeError, "group_principals must be a two-element list!");
		return NULL;
	}

	std::vector<long>  group_principals;
	try {
		pysupport::pylist_of_int_to_cvec(py_group_principals, &group_principals, "group principal");
	} catch (pysupport::PySupportError& pse) {
		PyErr_SetString(PyExc_TypeError, pse.what());
		return NULL;
	}

	auto group_rep = make_condition(py_group_rep);
	if (group_rep == nullptr)
		return NULL;

Py_BEGIN_ALLOW_THREADS
Py_END_ALLOW_THREADS

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

}

static const char* docstr_find_group = "find_group\n"
"Find a chemical group (documented in Python layer)";

static PyMethodDef cg_methods[] = {
	{ PY_STUPID "find_group", find_group,	METH_VARARGS, PY_STUPID docstr_find_group	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cg_def =
{
	PyModuleDef_HEAD_INIT,
	"_cg",
	"Chemical group finding",
	-1,
	cg_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit__cg()
{
	return PyModule_Create(&cg_def);
}
