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
#include <algorithm>  // std::find
#include <atomstruct/Atom.h>
#include <atomstruct/AtomicStructure.h>
#include <functional>
#include <map>
#include <mutex>
#include <pysupport/convert.h>
#include <sstream>
#include <thread>
#include <vector>

using atomstruct::Atom;
using atomstruct::AtomicStructure;
using atomstruct::AtomType;

typedef std::vector<const Atom*> Group;

class Atom_Condition
{
public:
	virtual  ~Atom_Condition() {}
	virtual bool  atom_matches(const Atom* a) const = 0;
	virtual bool  operator==(const Atom_Condition& other) const = 0;
	virtual bool  possibly_matches_H() const = 0;
	virtual std::vector<Group>  trace_group(const Atom* a, const Atom* parent = nullptr) = 0;
};

class Atom_Idatm_Condition: public Atom_Condition
{
	AtomType  _idatm_type;
public:
	Atom_Idatm_Condition(const char *idatm_type): _idatm_type(idatm_type) {}
	virtual  ~Atom_Idatm_Condition() {}
	bool  atom_matches(const Atom* a) const { return a->idatm_type() == _idatm_type; }
	bool  operator==(const Atom_Condition& other) const {
		auto casted = dynamic_cast<const Atom_Idatm_Condition*>(&other);
		if (casted == nullptr)
			return false;
		return casted->_idatm_type == _idatm_type;
	}
	bool  possibly_matches_H() const { return _idatm_type == "H" || _idatm_type == "HC"; }
	std::vector<Group>  trace_group(const Atom* a, const Atom* = nullptr) {
		std::vector<Group> traced_groups;
		if (atom_matches(a)) {
			traced_groups.emplace_back();
			traced_groups.back().push_back(a);
		}
		return traced_groups;
	}
};

class CG_Condition: public Atom_Condition
{
public:
	Atom_Condition*  atom_cond;
	std::vector<Atom_Condition*>  bonded; // may actually also hold CG_Conditions

	virtual  ~CG_Condition() { delete atom_cond; for (auto cond: bonded) delete cond; }
	bool  atom_matches(const Atom* a) const { return atom_cond->atom_matches(a); }
	bool  operator==(const Atom_Condition& other) const {
		auto casted = dynamic_cast<const CG_Condition*>(&other);
		if (casted == nullptr)
			return false;
		if (!(atom_cond == casted->atom_cond))
			return false;
		if (bonded.size() != casted->bonded.size())
			return false;
		for (unsigned int i = 0; i < bonded.size(); ++i) {
			if (!(bonded[i] == casted->bonded[i]))
				return false;
		}
		return true;
	}
	bool  possibly_matches_H() const { return false; }
	std::vector<Group>  trace_group(const Atom* a, const Atom* parent = nullptr);
};

inline unsigned int
count_possible_Hs(std::vector<Atom_Condition*>& conditions)
{
	unsigned int possible_Hs = 0;
	for (auto cond: conditions)
		if (cond->possibly_matches_H())
			possible_Hs++;
	return possible_Hs;
}

bool
condition_compare(Atom_Condition* c1, Atom_Condition* c2)
{
	//TODO
	return *c1 == *c2;
}

typedef std::map<Atom*, std::vector<Atom_Condition*>> Assignments;

std::vector<Group>
match_descendents(const Atom* a, const Atom::Neighbors& neighbors, const Atom* parent,
	std::vector<Atom_Condition*>& descendents, Assignments prev_assigned = Assignments())
{
	// prev_assigned notes what atom->condition assignments have occurred and is
	// used to try to avoid multiply matching indistinguishable fragments with the
	// same set of atoms
	std::vector<Group> matches;
	auto target = descendents[0];
	std::vector<Atom_Condition*> alternatives(descendents.begin()+1, descendents.end());
	unsigned int bonds_to_match = neighbors.size() - (parent != nullptr);

	if (descendents.size() < bonds_to_match
	|| descendents.size() - count_possible_Hs(descendents) > bonds_to_match)
		return matches;

	for (auto other_atom: neighbors) {
		if (other_atom == parent)
			continue;

		auto prev_assignments = prev_assigned.find(other_atom);
		if (prev_assignments != prev_assigned.end()) {
			bool skip_atom = false;
			for (auto assignment: (*prev_assignments).second) {
				if (condition_compare(target, assignment)) {
					skip_atom = true;
					break;
				}
			}
			if (skip_atom)
				continue;
		}
		
		auto possible_matches = target->trace_group(other_atom, a);
		if (possible_matches.size() > 0) {
			std::vector<Group> remainder_matches;
			if (alternatives.size() == 0)
				remainder_matches.emplace_back();
			else {
				auto remain_neighbors = neighbors;
				remain_neighbors.erase(
					std::find(remain_neighbors.begin(), remain_neighbors.end(), other_atom));
				remainder_matches = match_descendents(a, remain_neighbors, parent, alternatives,
					prev_assigned);
			}
			if (remainder_matches.size() > 0) {
				for (auto match1: possible_matches) {
					for (auto match2: remainder_matches) {
						matches.push_back(match1);
						matches.back().insert(matches.back().end(), match2.begin(), match2.end());
					}
				}
			}
			// don't modify the value of prev_assigned in place, since it may be in use elsewhere
			std::vector<Atom_Condition*> new_assigned;
			if (prev_assigned.find(other_atom) != prev_assigned.end())
				new_assigned = prev_assigned[other_atom];
			new_assigned.push_back(target);
			prev_assigned[other_atom] = new_assigned;
		}
	}

	if (target->possibly_matches_H() && alternatives.size() >= bonds_to_match
	&& alternatives.size() - count_possible_Hs(alternatives) <= bonds_to_match) {
		// since 'R'/None may be hydrogen, and hydrogen can be missing from the
		// structure, check if the group matches while omitting the 'R'/None (or H)
		if (alternatives.size() == 0) // and bonds_to_match == 0 due to preceding test
			matches.emplace_back();
		else {
			auto remainder_matches = match_descendents(a, neighbors, parent, alternatives,
				prev_assigned);
			for (auto match: remainder_matches)
				matches.push_back(match);
		}
	}
	return matches;
}

std::vector<Group>
CG_Condition::trace_group(const Atom* a, const Atom* parent)
{
	std::vector<Group> traced_groups;
	if (!atom_matches(a))
		return traced_groups;

	// for efficiency, don't check the bonded atoms in detail if they can't
	// possibly match because the number is wrong (accounting for hydrogens
	// being allowed to match nothing)
	unsigned int bonds_to_match = a->bonds().size() - (parent != nullptr);
	if (bonded.size() < bonds_to_match
	|| bonded.size() - count_possible_Hs(bonded) > bonds_to_match)
		return traced_groups;

	if (bonded.size() == 0) {
		// due to preceeding test, bonds_to_match must also be 0
		traced_groups.emplace_back();
		traced_groups.back().push_back(a);
		return traced_groups;
	}

	//TODO: evaluate bonded
	auto matches = match_descendents(a, a->neighbors(), parent, bonded);
}

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
	std::vector<Group>* groups, std::mutex* groups_mutex)
{
	atoms_mutex->lock();
	while (*atom_index < atoms.size()) {
		auto a = atoms[*atom_index];
		(*atom_index)++;
		atoms_mutex->unlock();

		for (auto raw_group: group_rep->trace_group(a)) {
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
	std::vector<Group> groups;
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
