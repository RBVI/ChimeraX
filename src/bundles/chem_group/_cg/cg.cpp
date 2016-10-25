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
#include <atomstruct/Atom.h>
#include <atomstruct/AtomicStructure.h>
#include <functional>
#include <mutex>
#include <pysupport/convert.h>
#include <sstream>
#include <thread>
#include <vector>

using atomstruct::Atom;
using atomstruct::AtomicStructure;

class Atom_Condition
{
public:
	virtual  ~Atom_Condition() {}
	virtual bool  atom_matches(const Atom* a) const = 0;
};

class Atom_Idatm_Condition: public Atom_Condition
{
	std::string  _idatm_type;
public:
	Atom_Idatm_Condition(const char *idatm_type): _idatm_type(idatm_type) {}
	virtual  ~Atom_Idatm_Condition() {}
	bool  atom_matches(const Atom* a) const { return a->idatm_type() == _idatm_type; }
};

class CG_Condition: public Atom_Condition
{
public:
	Atom_Condition*  atom_cond;
	std::vector<Atom_Condition*>  bonded; // may actually also hold CG_Conditions

	virtual  ~CG_Condition() { delete atom_cond; for (auto cond: bonded) delete cond; }
	bool  atom_matches(const Atom* a) const { return atom_cond->atom_matches(a); }
	bool  evaluate(const Atom* a, std::vector<const Atom*>& group) {
		if (atom_cond->atom_matches(a)) {
			group.push_back(a);
		} else {
			return false;
		}
		//TODO: evaluate bonded
	}
};

Atom_Condition*
make_atom_condition(PyObject* atom_rep)
{
	if (PyUnicode_Check(atom_rep)) {
		return new Atom_Idatm_Condition(PyUnicode_AsUTF8(atom_rep));
	}
	//TODO: remainder of types
}

CG_Condition*
make_condition(PyObject* group_rep)
{
	if (!PyList_Check(group_rep) || PyList_Size(group_rep) != 2) {
		PyObject* repr = PyObject_ASCII(group_rep);
		if (repr == nullptr)
			PyErr_SetString(PyExc_ValueError,
				"Could not compute repr() of chem group representation");
		else {
			std::ostringstream err_msg;
			err_msg << "While parsing chemical group representation, ";
			err_msg << "expected two-element list but got: ";
			err_msg << PyUnicode_AsUTF8(repr);
			PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
			Py_DECREF(repr);
		}
		return nullptr;
	}
	PyObject* atom = PyList_GET_ITEM(group_rep, 0);
	PyObject* bonded = PyList_GET_ITEM(group_rep, 1);
	if (!PyList_Check(bonded)) {
		PyErr_SetString(PyExc_ValueError, "Second element of chem group list is not itself a list");
		return nullptr;
	}

	auto cond = new CG_Condition();
	cond->atom_cond = make_atom_condition(atom);
	if (cond->atom_cond == nullptr) {
		delete cond;
		return nullptr;
	}
	
	auto list_size = PyList_GET_SIZE(bonded);
	for (Py_ssize_t i = 0; i < list_size; ++i) {
		PyObject* b = PyList_GET_ITEM(bonded, i);
		Atom_Condition* bcond;
		if (PyList_Check(b))
			bcond = static_cast<Atom_Condition*>(make_condition(b));
		else
			bcond = make_atom_condition(b);
		if (bcond == nullptr) {
			delete cond;
			return nullptr;
		}
		cond->bonded.push_back(bcond);
	}
	return cond;
}

void
initiate_find_group(CG_Condition* group_rep, std::vector<long>* group_principals,
	const std::vector<Atom*>& atoms, std::mutex* atoms_mutex, size_t* atom_index,
	std::vector<std::vector<const Atom*>>* groups, std::mutex* groups_mutex)
{
	atoms_mutex->lock();
	while (*atom_index < atoms.size()) {
		auto a = atoms[*atom_index];
		(*atom_index)++;
		atoms_mutex->unlock();

		std::vector<const Atom*> group;
		if (group_rep->evaluate(a, group)) {
			//TODO: check rings, reduce to principals, and add group with locking
		}

		atoms_mutex->lock();
	}
	atoms_mutex->unlock();
}

extern "C" {

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
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));
	if (!PyList_Check(py_group_principals)) {
		PyErr_SetString(PyExc_TypeError, "group_principals must be a list!");
		return nullptr;
	}

	std::vector<long>  group_principals;
	try {
		pysupport::pylist_of_int_to_cvec(py_group_principals, group_principals, "group principal");
	} catch (pysupport::PySupportError& pse) {
		PyErr_SetString(PyExc_TypeError, pse.what());
		return nullptr;
	}

	auto group_rep = make_condition(py_group_rep);
	if (group_rep == nullptr)
		return nullptr;

	auto& atoms = s->atoms();
	std::vector<std::vector<const Atom*>> groups;
	std::mutex atoms_mtx, groups_mtx;

	int num_threads = num_cpus > 1 ? num_cpus : 1;
	size_t atom_index = 0;
	std::vector<std::thread> threads;
	for (int i = 0; i < num_threads; ++i)
		threads.push_back(std::thread(initiate_find_group, group_rep, &group_principals,
			std::ref(atoms), &atoms_mtx, &atom_index, &groups, &groups_mtx));
	for (auto& th: threads)
		th.join();

	delete group_rep;

	// return some Python form of the groups
	return Py_None;
	//return groups;

}

}

static const char* docstr_find_group = "find_group\n"
"Find a chemical group (documented in Python layer)";

static PyMethodDef cg_methods[] = {
	{ PY_STUPID "find_group", find_group,	METH_VARARGS, PY_STUPID docstr_find_group	},
	{ nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef cg_def =
{
	PyModuleDef_HEAD_INIT,
	"_cg",
	"Chemical group finding",
	-1,
	cg_methods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

PyMODINIT_FUNC
PyInit__cg()
{
	return PyModule_Create(&cg_def);
}
