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
#include <algorithm>  // std::find, std::min
#include <arrays/pythonarray.h>
#include <atomstruct/Atom.h>
#include <atomstruct/AtomicStructure.h>
#include <cstring>
#include <element/Element.h>
#include <functional>
#include <map>
#include <mutex>
#include <pysupport/convert.h>
#include <set>
#include <sstream>
#include <thread>
#include <typeinfo>
#include <vector>

using atomstruct::Atom;
using atomstruct::AtomicStructure;
using atomstruct::AtomType;
using element::Element;

typedef std::vector<const Atom*> Group;

static PyObject*  py_ring_atom_class;

class AtomCondition
{
public:
	virtual  ~AtomCondition() {}
	virtual bool  atom_matches(const Atom* a) const = 0;
	virtual bool  operator==(const AtomCondition& other) const = 0;
	bool  operator!=(const AtomCondition& other) const { return !(*this == other); }
	virtual bool  possibly_matches_H() const = 0;
	virtual std::vector<Group>  trace_group(const Atom* a, const Atom* = nullptr) {
		std::vector<Group> traced_groups;
		if (atom_matches(a)) {
			traced_groups.emplace_back();
			traced_groups.back().push_back(a);
		}
		return traced_groups;
	}
};

static std::vector<AtomType> hydrogen_types = { "H", "HC", "D", "DC" };

class AtomIdatmCondition: public AtomCondition
// Python equivalent:  string
{
	AtomType  _idatm_type;
public:
	AtomIdatmCondition(const char *idatm_type): _idatm_type(idatm_type) {}
	AtomIdatmCondition(const AtomType& idatm_type): _idatm_type(idatm_type) {}
	virtual  ~AtomIdatmCondition() {}
	bool  atom_matches(const Atom* a) const {
		return a != nullptr && a->idatm_type() == _idatm_type;
	}
	bool  atom_matches(const AtomType& idatm_type) const { return idatm_type == _idatm_type; }
	bool  is_heavy_atom_type() const {
		return std::find(hydrogen_types.begin(), hydrogen_types.end(), _idatm_type)
			== hydrogen_types.end();
	}
	bool  operator==(const AtomCondition& other) const {
		auto casted = dynamic_cast<const AtomIdatmCondition*>(&other);
		if (casted == nullptr)
			return false;
		return casted->_idatm_type == _idatm_type;
	}
	bool  possibly_matches_H() const { return _idatm_type == "H" || _idatm_type == "HC"; }
};

class AtomElementCondition: public AtomCondition
// Python equivalent:  int
{
	int  _element_num;
public:
	AtomElementCondition(int element_num): _element_num(element_num) {}
	virtual  ~AtomElementCondition() {}
	bool  atom_matches(const Atom* a) const {
		return a != nullptr && a->element().number() == _element_num;
	}
	bool  operator==(const AtomCondition& other) const {
		auto casted = dynamic_cast<const AtomElementCondition*>(&other);
		if (casted == nullptr)
			return false;
		return casted->_element_num == _element_num;
	}
	bool  possibly_matches_H() const { return _element_num == 1; }
};

class AtomAlternativesCondition: public AtomCondition
// Python equivalent:  tuple
{
public:
	std::vector<AtomCondition*>  alternatives;

	virtual  ~AtomAlternativesCondition() { for (auto cond: alternatives) delete cond; }
	bool  atom_matches(const Atom* a) const {
		for (auto cond: alternatives)
			if (cond->atom_matches(a)) return true;
		return false;
	}
	bool  operator==(const AtomCondition& other) const {
		auto casted = dynamic_cast<const AtomAlternativesCondition*>(&other);
		if (casted == nullptr)
			return false;
		for (auto cond1: alternatives) {
			bool matched_any = false;
			for (auto cond2: casted->alternatives) {
				if (cond1 == cond2) {
					matched_any = true;
					break;
				}
			}
			if (!matched_any)
				return false;
		}
		return true;
	}
	bool  possibly_matches_H() const {
		for (auto cond: alternatives)
			if (cond->possibly_matches_H()) return true;
		return false;
	}
	std::vector<Group>  trace_group(const Atom* a, const Atom* parent = nullptr) {
		std::vector<Group> traced_groups;
		for (auto cond: alternatives) {
			traced_groups = cond->trace_group(a, parent);
			if (traced_groups.size() > 0)
				break;
		}
		return traced_groups;
	}
};

class AnyAtomCondition: public AtomCondition
// Python equivalent:  None
{
public:
	AnyAtomCondition() {}
	virtual  ~AnyAtomCondition() {}
	bool  atom_matches(const Atom*) const { return true; }
	bool  operator==(const AtomCondition& other) const {
		auto casted = dynamic_cast<const AnyAtomCondition*>(&other);
		return (casted != nullptr);
	}
	bool  possibly_matches_H() const { return true; }
};

class IdatmPropertyCondition: public AtomCondition
// Python equivalent:  dict
{
public:
	bool  has_default = false;
	bool  default_val;
	std::vector<AtomIdatmCondition*>  not_type;
	bool  has_geometry = false;
	Atom::IdatmGeometry  geometry;
	int  substituents = -1;

	virtual  ~IdatmPropertyCondition() { for (auto cond: not_type) delete cond; }
	bool  atom_matches(const AtomType& idatm_type) const;
	bool  atom_matches(const Atom* a) const {
		if (a == nullptr)
			return matches_missing_structure();
		return atom_matches(a->idatm_type());
	}
	bool  matches_missing_structure() const;
	bool  operator==(const AtomCondition& other) const {
		auto casted = dynamic_cast<const IdatmPropertyCondition*>(&other);
		if (casted == nullptr)
			return false;
		if (has_default != casted->has_default)
			return false;
		if (has_default && default_val != casted->default_val)
			return false;
		if (has_geometry != casted->has_geometry)
			return false;
		if (has_geometry && geometry != casted->geometry)
			return false;
		if (substituents != casted->substituents)
			return false;
		for (auto cond1: not_type) {
			bool matched_any = false;
			for (auto cond2: casted->not_type) {
				if (cond1 == cond2) {
					matched_any = true;
					break;
				}
			}
			if (!matched_any)
				return false;
		}
		return true;
	}
	bool  possibly_matches_H() const {
		AtomType h("H"), hc("HC");
		return atom_matches(h) || atom_matches(hc);
	}
};

bool
IdatmPropertyCondition::atom_matches(const AtomType& idatm_type) const
{
	auto& idatm_info_map = Atom::get_idatm_info_map();
	auto mi = idatm_info_map.find(idatm_type);
	if (mi == idatm_info_map.end()) {
		// uncommon type
		if (has_default)
			return default_val;
		return false;
	}
	if (not_type.size() > 0) {
		for (auto cond: not_type)
			if (cond->atom_matches(idatm_type))
				return false;;
	}
	if (has_geometry && mi->second.geometry != geometry)
		return false;
	if (substituents >= 0 && (int)mi->second.substituents != substituents)
		return false;
	return true;
}

bool
IdatmPropertyCondition::matches_missing_structure() const
{
	// only thing missing structure matches is a generic heavy atom...
	if (has_geometry || substituents >= 0)
		return false;

	for (auto cond: not_type) {
		if (cond->is_heavy_atom_type())
			return false;
	}
	return true;
}

class RingAtomCondition: public AtomCondition
// Python equivalent:  RingAtom instance
{
	AtomCondition*  _cond;
	Atom::Rings::size_type  _num_rings;
public:
	RingAtomCondition(AtomCondition* ac, int num_rings): _cond(ac), _num_rings(num_rings) {}
	virtual  ~RingAtomCondition() {}
	bool  atom_matches(const Atom* a) const {
		return _cond->atom_matches(a) && a != nullptr && a->rings().size() == _num_rings;
	}
	bool  operator==(const AtomCondition& other) const {
		auto casted = dynamic_cast<const RingAtomCondition*>(&other);
		if (casted == nullptr)
			return false;
		return casted->_num_rings == _num_rings && casted->_cond == _cond;
	}
	bool  possibly_matches_H() const { return false; }
	std::vector<Group>  trace_group(const Atom* a, const Atom* = nullptr) {
		std::vector<Group> traced_groups;
		if (atom_matches(a)) {
			traced_groups.emplace_back();
			traced_groups.back().push_back(a);
		}
		return traced_groups;
	}
};

class CG_Condition: public AtomCondition
// Python equivalent:  list
{
public:
	AtomCondition*  atom_cond;
	std::vector<AtomCondition*>  bonded; // may actually also hold CG_Conditions

	virtual  ~CG_Condition() { delete atom_cond; for (auto cond: bonded) delete cond; }
	bool  atom_matches(const Atom* a) const { return atom_cond->atom_matches(a); }
	bool  operator==(const AtomCondition& other) const {
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
count_possible_Hs(std::vector<AtomCondition*>& conditions)
{
	unsigned int possible_Hs = 0;
	for (auto cond: conditions)
		if (cond->possibly_matches_H())
			possible_Hs++;
	return possible_Hs;
}

static bool
condition_compare(AtomCondition* c1, AtomCondition* c2)
{
	if (typeid(*c1) != typeid(*c2)) {
		auto aac1 = dynamic_cast<AtomAlternativesCondition*>(c1);
		if (aac1 != nullptr) {
			for (auto alt: aac1->alternatives)
				if (alt == c2)
					return true;
		}
		auto aac2 = dynamic_cast<AtomAlternativesCondition*>(c2);
		if (aac2 != nullptr) {
			for (auto alt: aac2->alternatives)
				if (alt == c1)
					return true;
		}
		return false;
	}
	// condition types now known to be the same
	auto cgc1 = dynamic_cast<CG_Condition*>(c1);
	auto aac1 = dynamic_cast<AtomAlternativesCondition*>(c1);
	if (cgc1 != nullptr || aac1 != nullptr) {
		std::vector<AtomCondition*> conditions1, conditions2;
		if (cgc1 != nullptr) {
			auto cgc2 = dynamic_cast<CG_Condition*>(c2);
			conditions1.push_back(cgc1->atom_cond);
			conditions1.insert(conditions1.end(), cgc1->bonded.begin(), cgc1->bonded.end());
			conditions2.push_back(cgc2->atom_cond);
			conditions2.insert(conditions2.end(), cgc2->bonded.begin(), cgc2->bonded.end());
		} else {
			auto aac2 = dynamic_cast<AtomAlternativesCondition*>(c2);
			conditions1.insert(conditions1.end(),
				aac1->alternatives.begin(), aac1->alternatives.end());
			conditions2.insert(conditions2.end(),
				aac2->alternatives.begin(), aac2->alternatives.end());
		}
		if (conditions1.size() != conditions2.size())
			return false;
		for (decltype(conditions1)::size_type i = 0; i < conditions1.size(); ++i) {
			auto cond1 = conditions1[i];
			auto cond2 = conditions2[i];
			if (dynamic_cast<AtomAlternativesCondition*>(cond1) != nullptr
			|| dynamic_cast<CG_Condition*>(cond1) != nullptr) {
				if (!condition_compare(cond1, cond2))
					return false;
			} else {
				if (*cond1 != *cond2)
					return false;
			}
		}
		return true;
	}
	return *c1 == *c2;
}

typedef std::map<Atom*, std::vector<AtomCondition*>> Assignments;

static std::vector<Group>
match_descendents(const Atom* a, const Atom::Neighbors& neighbors, const Atom* parent,
	std::vector<AtomCondition*>& descendents, Assignments prev_assigned = Assignments())
{
	// prev_assigned notes what atom->condition assignments have occurred and is
	// used to try to avoid multiply matching indistinguishable fragments with the
	// same set of atoms
	std::vector<Group> matches;
	auto target = descendents[0];
	std::vector<AtomCondition*> alternatives(descendents.begin()+1, descendents.end());
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
			std::vector<AtomCondition*> new_assigned;
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
	if (!atom_matches(a) || a == nullptr)
		return traced_groups;

	// Use nullptrs for missing-structure "neighbors"
	auto neighbors = a->neighbors();
	auto nulls_to_add = a->num_explicit_bonds() - neighbors.size();
	for (decltype(nulls_to_add) i = 0; i < nulls_to_add; ++i)
		neighbors.push_back(nullptr);

	// for efficiency, don't check the bonded atoms in detail if they can't
	// possibly match because the number is wrong (accounting for hydrogens
	// being allowed to match nothing)
	unsigned int bonds_to_match = neighbors.size() - (parent != nullptr);
	if (bonded.size() < bonds_to_match
	|| bonded.size() - count_possible_Hs(bonded) > bonds_to_match)
		return traced_groups;

	if (bonded.size() == 0) {
		// due to preceeding test, bonds_to_match must also be 0
		traced_groups.emplace_back();
		traced_groups.back().push_back(a);
	} else { 
		auto matches = match_descendents(a, neighbors, parent, bonded);
		for (auto& match: matches) {
			traced_groups.emplace_back();
			auto& back = traced_groups.back();
			back.push_back(a);
			back.insert(back.end(), match.begin(), match.end());
		}
	}
	return traced_groups;
}

static IdatmPropertyCondition*
make_idatm_property_condition(PyObject* dict)
{
	auto cond = new IdatmPropertyCondition;
	PyObject* key;
	PyObject* value;
	Py_ssize_t pos = 0;
	while (PyDict_Next(dict, &pos, &key, &value)) {
		if (!PyUnicode_Check(key)) {
			delete cond;
			PyObject* repr = PyObject_ASCII(key);
			if (repr == nullptr) {
				PyErr_SetString(PyExc_ValueError, "Could not compute repr() of"
					" chem group IDATM-property test-condition dictionary key");
				return nullptr;
			}
			std::ostringstream err_msg;
			err_msg << "Unexpected IDATM-property condition dictionary key: ";
			err_msg << PyUnicode_AsUTF8(repr);
			PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
			Py_DECREF(repr);
			return nullptr;
		}
		auto str_key = PyUnicode_AsUTF8(key);
		if (str_key == nullptr) {
			delete cond;
			PyObject* repr = PyObject_ASCII(key);
			if (repr == nullptr) {
				PyErr_SetString(PyExc_ValueError, "Could not convert"
					" chem group IDATM-property test-condition dictionary key to UTF8");
				return nullptr;
			}
			std::ostringstream err_msg;
			err_msg << "Unexpected IDATM-property condition dictionary key: ";
			err_msg << repr;
			PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
			Py_DECREF(repr);
			return nullptr;
		}
		if (strcmp(str_key, "default") == 0) {
			if (!PyBool_Check(value)) {
				delete cond;
				std::ostringstream err_msg;
				err_msg << "Value for chem group IDATM-property test-condition dictionary key";
				err_msg << " 'default' is not boolean";
				PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
				return nullptr;
			}
			cond->has_default = true;
			cond->default_val = value == Py_True;
			continue;
		}
		if (strcmp(str_key, "not type") == 0) {
			if (!PyList_Check(value)) {
				delete cond;
				std::ostringstream err_msg;
				err_msg << "Value for chem group IDATM-property test-condition dictionary key";
				err_msg << " 'notType' is not a list";
				PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
				return nullptr;
			}
			auto size = PyList_Size(value);
			for (decltype(size) i = 0; i < size; ++i) {
				auto item = PyList_GET_ITEM(value, i);
				if (!PyUnicode_Check(item)) {
					delete cond;
					std::ostringstream err_msg;
					err_msg << "Item in chem group IDATM-property test-condition dictionary";
					err_msg << " 'not type' list is not a string";
					PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
					return nullptr;
				}
				cond->not_type.push_back(new AtomIdatmCondition(PyUnicode_AsUTF8(item)));
			}
			continue;
		}
		if (strcmp(str_key, "geometry") == 0) {
			if (!PyLong_Check(value)) {
				delete cond;
				std::ostringstream err_msg;
				err_msg << "Value for chem group IDATM-property test-condition dictionary key";
				err_msg << " 'geometry' is not an integer";
				PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
				return nullptr;
			}
			cond->has_geometry = true;
			cond->geometry = static_cast<Atom::IdatmGeometry>(PyLong_AsLong(value));
			continue;
		}
		if (strcmp(str_key, "substituents") == 0) {
			if (!PyLong_Check(value)) {
				delete cond;
				std::ostringstream err_msg;
				err_msg << "Value for chem group IDATM-property test-condition dictionary key";
				err_msg << " 'substituents' is not an integer";
				PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
				return nullptr;
			}
			cond->substituents = static_cast<int>(PyLong_AsLong(value));
			continue;
		}
		delete cond;
		PyObject* repr = PyObject_ASCII(key);
		if (repr == nullptr) {
			PyErr_SetString(PyExc_ValueError, "Could not compute repr() of"
				" chem group IDATM-property test-condition dictionary key");
			return nullptr;
		}
		std::ostringstream err_msg;
		err_msg << "Unexpected IDATM-property condition dictionary key: ";
		err_msg << PyUnicode_AsUTF8(repr);
		PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
		Py_DECREF(repr);
		return nullptr;
	}
	return cond;
}

static AtomCondition*
make_simple_atom_condition(PyObject* atom_rep)
{
	if (PyUnicode_Check(atom_rep))
		return new AtomIdatmCondition(PyUnicode_AsUTF8(atom_rep));
	if (PyLong_Check(atom_rep)) 
		return new AtomElementCondition((int)PyLong_AsLong(atom_rep));
	if (PyTuple_Check(atom_rep)) {
		auto cond = new AtomAlternativesCondition;
		auto num_conds = PyTuple_GET_SIZE(atom_rep);
		for (decltype(num_conds) i = 0; i < num_conds; ++i) {
			auto sub_cond = make_simple_atom_condition(PyTuple_GET_ITEM(atom_rep, i));
			if (sub_cond == nullptr) {
				for (decltype(i) j = 0; j < i; ++j)
					delete cond->alternatives[j];
				delete cond;
				return nullptr;
			}
			cond->alternatives.push_back(make_simple_atom_condition(PyTuple_GET_ITEM(atom_rep, i)));
		}
		return cond;
	}
	if (PyDict_Check(atom_rep))
		return make_idatm_property_condition(atom_rep);
	if (atom_rep == Py_None)
		return new AnyAtomCondition();

	auto py_type = PyObject_Type(atom_rep);
	if (py_type == nullptr) {
		PyErr_SetString(PyExc_ValueError, "Could not get type() of chem group fragment");
		return nullptr;
	}
	auto py_type_string = PyObject_ASCII(py_type);
	if (py_type_string == nullptr) {
		PyErr_SetString(PyExc_ValueError,
			"Could not convert type to ASCII string for chem group fragment");
		Py_DECREF(py_type);
		return nullptr;
	}
	PyObject* repr = PyObject_ASCII(atom_rep);
	if (repr == nullptr) {
		PyErr_SetString(PyExc_ValueError,
			"Could not compute repr() of chem group test-condition representation");
		Py_DECREF(py_type);
		Py_DECREF(py_type_string);
		return nullptr;
	}
	std::ostringstream err_msg;
	err_msg << "Unexpected type (";
	err_msg << PyUnicode_AsUTF8(py_type_string);
	err_msg << ") for chem group component: ";
	err_msg << PyUnicode_AsUTF8(repr);
	PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
	Py_DECREF(py_type);
	Py_DECREF(py_type_string);
	Py_DECREF(repr);
	return nullptr;
}

static AtomCondition*
make_atom_condition(PyObject* atom_rep)
{
	if (PyObject_IsInstance(atom_rep, py_ring_atom_class)) {
		auto atom_desc = PyObject_GetAttrString(atom_rep, "atom_desc");
		if (atom_desc == nullptr) {
			PyErr_SetString(PyExc_AttributeError, "RingAtom instance has no 'atom_desc' attribute");
			return nullptr;
		}
		auto cond = make_simple_atom_condition(atom_desc);
		if (cond == nullptr) {
			Py_DECREF(atom_desc);
			return nullptr;
		}
		auto num_rings = PyObject_GetAttrString(atom_rep, "num_rings");
		if (atom_desc == nullptr) {
			delete cond;
			Py_DECREF(atom_desc);
			PyErr_SetString(PyExc_AttributeError, "RingAtom instance has no 'num_rings' attribute");
			return nullptr;
		}
		if (!PyLong_Check(num_rings)) {
			delete cond;
			Py_DECREF(atom_desc);
			Py_DECREF(num_rings);
			PyErr_SetString(PyExc_AttributeError, "RingAtom 'num_rings' attribute is not an int");
			return nullptr;
		}
		return new RingAtomCondition(cond, static_cast<int>(PyLong_AsLong(num_rings)));
	}
	return make_simple_atom_condition(atom_rep);
}

static CG_Condition*
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
		AtomCondition* bcond;
		if (PyList_Check(b))
			bcond = static_cast<AtomCondition*>(make_condition(b));
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

static void
initiate_find_group(CG_Condition* group_rep, std::vector<long>* group_principals,
	AtomicStructure::Atoms::const_iterator start, AtomicStructure::Atoms::const_iterator end,
	std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;

		for (auto raw_group: group_rep->trace_group(a)) {
			// check rings, reduce to principals, and add group with locking
			std::map<long, const Atom*> ring_atom_map;
			Group pruned;
			bool rings_okay = true;
			auto group_size = raw_group.size();
			for (Group::size_type i = 0; i < group_size; ++i) {
				auto principal = (*group_principals)[i];
				auto group_atom = raw_group[i];
				if (principal != 0) {
					if (group_atom == nullptr) {
						// missing-structure can't actually be part of returned
						// group; disallow/abort by setting rings_okay false
						rings_okay = false;
						break;
					}
					if (ring_atom_map.find(principal) != ring_atom_map.end()) {
						if (ring_atom_map[principal] != group_atom) {
							rings_okay = false;
							break;
						}
						// don't add a second instance of this atom
						continue;
					}
					if (principal != 1)
						ring_atom_map[principal] = group_atom;
					pruned.push_back(group_atom);
				}
			}
			if (rings_okay) {
				groups_mutex->lock();
				groups->emplace_back();
				auto& back = groups->back();
				back.swap(pruned);
				groups_mutex->unlock();
			}
		}
	}
}

static Group
find_aro_amine(const Atom* a, unsigned int order)
{
	// already checked to be an Npl not in an aromatic ring
	if (a->has_missing_structure_pseudobond())
		return Group();
	Group amine = { a };
	unsigned int bound_carbons = 0;
	bool bound_aro = false;
	for (auto nb: a->neighbors()) {
		if (nb->element().number() == 1) { // H
			amine.push_back(nb);
			continue;
		}
		if (nb->element().number() == 6) { // C
			++bound_carbons;
			if (nb->idatm_type() == "Car") {
				bound_aro = true;
				continue;
			}
			if (nb->idatm_type() == "C3")
				continue;
		}
		// something non-amine-like is bonded...
		bound_aro = false;
		break;
	}
	if (!bound_aro)
		return Group();
	if (order > 0 && bound_carbons != order)
		return Group();
	return amine;
}

static void
initiate_find_aro_amines(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end,
	unsigned int order, std::set<const Atom*>* aro_ring_npls,
	std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (a->idatm_type() != "Npl" || aro_ring_npls->find(a) != aro_ring_npls->end())
			continue;

		auto amine = find_aro_amine(a, order);
		if (amine.size() > 0) {
			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			back.swap(amine);
			groups_mutex->unlock();
		}
	}
}

static Group
find_ring_planar_NHR2(const Atom* a, unsigned int ring_size, bool aromatic_only)
{
	// already checked to be an Npl or N2+ participating in one ring
	if (a->has_missing_structure_pseudobond())
		return Group();
	Group nhr2 = { a };
	auto& neighbors = a->neighbors();
	if (neighbors.size() == 3) {
		bool h_bound = false;
		for (auto nb: neighbors) {
			if (nb->element().number() == 1) {
				h_bound = true;
				break;
			}
		}
		if (!h_bound)
			return Group();
	}

	auto ring = a->rings()[0];
	if (ring->size() != ring_size)
		return Group();

	if (aromatic_only && !ring->aromatic())
		return Group();

	auto& ring_atoms = ring->atoms();
	for (auto nb: neighbors) {
		if (ring_atoms.find(nb) != ring_atoms.end())
			nhr2.push_back(nb);
	}
	return nhr2;
}

static Group
find_5ring_planar_NR2(const Atom* a, bool symmetric)
{
	// already checked to be an N2 or two-bond Npl or N2+ participating in one 5-member ring
	if (a->has_missing_structure_pseudobond())
		return Group();
	Group nr2 = { a };

	auto& N = Element::get_element("N");
	auto ring = a->rings()[0];
	auto& ring_atoms = ring->atoms();
	// only consider Npl/N2+ ambiguous if multiple nitrogens in ring
	if (a->idatm_type() != "N2") {
		bool other_N = false;
		for (auto ra: ring_atoms) {
			if (ra != a && ra->element() == N) {
				other_N = true;
				break;
			}
		}
		if (!other_N)
			return Group();
	}

	// check symmetry
	bool asymmetric = false;
	auto& neighbors = a->neighbors();
	auto bonded1 = neighbors[0];
	auto bonded2 = neighbors[1];
	nr2.push_back(bonded1);
	nr2.push_back(bonded2);
	if (symmetric) {
		if (bonded1->element() != bonded2->element())
			return Group();
	} else {
		if (bonded1->element() != bonded2->element())
			asymmetric = true;
	}
	Atom* rem1 = nullptr;
	Atom* rem2 = nullptr;
	for (auto ra: ring_atoms) {
		if (ra != a && ra != bonded1 && ra != bonded2) {
			if (rem1) {
				rem2 = ra;
				break;
			}
			rem1 = ra;
		}
	}
	if (symmetric) {
		if (rem1->element() != rem2->element())
			return Group();
	} else if (!asymmetric && rem1->element() == rem2->element())
		return Group();
	return nr2;
}

static Group
find_6ring_planar_NR2(const Atom* a, bool symmetric)
{
	// already checked to be an N2 participating in one 6-member ring
	if (a->has_missing_structure_pseudobond())
		return Group();
	Group nr2 = { a };

	auto ring = a->rings()[0];
	auto& ring_atoms = ring->atoms();

	// check symmetry
	bool asymmetric = false;
	auto& neighbors = a->neighbors();
	auto bonded1 = neighbors[0];
	auto bonded2 = neighbors[1];
	nr2.push_back(bonded1);
	nr2.push_back(bonded2);
	if (symmetric) {
		if (bonded1->element() != bonded2->element())
			return Group();
	} else {
		if (bonded1->element() != bonded2->element())
			asymmetric = true;
	}
	Atom* next_to_1 = nullptr;
	Atom* next_to_2 = nullptr;
	auto& nbs1 = bonded1->neighbors();
	auto& nbs2 = bonded2->neighbors();
	for (auto ra: ring_atoms) {
		if (ra != a && ra != bonded1 && ra != bonded2) {
			if (!next_to_1
			&& std::find(nbs1.begin(), nbs1.end(), ra) != nbs1.end())
				next_to_1 = ra;
			else if (!next_to_2
			&& std::find(nbs2.begin(), nbs2.end(), ra) != nbs2.end())
				next_to_2 = ra;
		}
	}
	if (symmetric) {
		if (next_to_1->element() != next_to_2->element())
			return Group();
	} else if (!asymmetric && next_to_1->element() == next_to_2->element())
		return Group();
	return nr2;
}

static Group
find_nonring_NR2(const Atom* a)
{
	// already checked to be an N2 not in a ring
	if (a->has_missing_structure_pseudobond())
		return Group();
	Group nr2 = { a };

	auto& bonded = a->neighbors();
	if (bonded.size() != 2)
		return Group();

	auto& C = Element::get_element("C");
	if (bonded[0]->element() != C || bonded[1]->element() != C)
		return Group();
	nr2.push_back(bonded[0]);
	nr2.push_back(bonded[1]);
	return nr2;
}

static Group
find_nonring_ether(const Atom* a)
{
	// already checked to be an O3 not in a ring
	if (a->has_missing_structure_pseudobond())
		return Group();
	Group ether = { a };

	auto& bonded = a->neighbors();
	if (bonded.size() != 2)
		return Group();
	std::vector<const Atom*> bonded_carbons;
	auto& C = Element::get_element("C");
	for (auto nb: a->neighbors()) {
		if (nb->element() == C)
			bonded_carbons.push_back(nb);
	}
	if (bonded_carbons.size() != 2)
		return Group();

	ether.push_back(bonded[0]);
	ether.push_back(bonded[1]);
	return ether;
}

static void
initiate_find_ring_planar_NHR2(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end, unsigned int ring_size,
	bool aromatic_only, std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (a->idatm_type() != "Npl" && a->idatm_type() != "N2+")
			continue;

		// caller already computed rings
		if (a->rings().size() != 1)
			continue;

		auto nhr2 = find_ring_planar_NHR2(a, ring_size, aromatic_only);
		if (nhr2.size() > 0) {
			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			back.swap(nhr2);
			groups_mutex->unlock();

			// if ring has multiple nitrogens and implicit hydrogens,
			// then even the N2 nitrogens may be donors...
			for (auto ra: a->rings()[0]->atoms()) {
				if (ra->idatm_type() == "N2") {
					// add group with locking
					groups_mutex->lock();
					groups->emplace_back();
					auto& back = groups->back();
					groups_mutex->unlock();
					back.push_back(ra);
					back.push_back(ra->neighbors()[0]);
					back.push_back(ra->neighbors()[1]);
				}
			}
		}
	}
}

static void
initiate_find_5ring_planar_NR2(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end,
	bool symmetric, std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (!(a->idatm_type() == "N2" || ((a->idatm_type() == "Npl" || a->idatm_type() == "N2+")
				&& a->neighbors().size() == 2)))
			continue;

		// caller already computed rings
		if (a->rings().size() != 1 || a->rings()[0]->size() != 5)
			continue;

		auto nr2 = find_5ring_planar_NR2(a, symmetric);
		if (nr2.size() > 0) {
			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			back.swap(nr2);
			groups_mutex->unlock();
		}
	}
}

static void
initiate_find_6ring_planar_NR2(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end,
	bool symmetric, std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (a->idatm_type() != "N2")
			continue;

		// caller already computed rings
		if (a->rings().size() != 1 || a->rings()[0]->size() != 6)
			continue;

		auto nr2 = find_6ring_planar_NR2(a, symmetric);
		if (nr2.size() > 0) {
			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			back.swap(nr2);
			groups_mutex->unlock();
		}
	}
}

static void
initiate_find_5ring_OR2(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end,
	std::vector<Group>* groups, std::mutex* groups_mutex)
{
	auto& O = Element::get_element("O");
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (a->element() != O)
			continue;

		if (a->has_missing_structure_pseudobond())
			continue;

		// caller already computed rings
		for (auto ring: a->rings()) {
			if (ring->size() != 5)
				continue;

			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			groups_mutex->unlock();
			back.push_back(a);
			for (auto nb: a->neighbors())
				back.push_back(nb);
		}
	}
}

static void
initiate_find_nonring_ether(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end,
	std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (a->idatm_type() != "O3")
			continue;

		// caller already computed rings
		if (a->rings().size() > 0)
			continue;

		auto ether = find_nonring_ether(a);
		if (ether.size() > 0) {
			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			back.swap(ether);
			groups_mutex->unlock();
		}
	}
}

static void
initiate_find_nonring_NR2(AtomicStructure::Atoms::const_iterator start,
	AtomicStructure::Atoms::const_iterator end,
	std::vector<Group>* groups, std::mutex* groups_mutex)
{
	for (auto i = start; i != end; ++i) {
		auto a = *i;
		if (a->idatm_type() != "N2")
			continue;

		// caller already computed rings
		if (a->rings().size() > 0)
			continue;

		auto nr2 = find_nonring_NR2(a);
		if (nr2.size() > 0) {
			// add group with locking
			groups_mutex->lock();
			groups->emplace_back();
			auto& back = groups->back();
			back.swap(nr2);
			groups_mutex->unlock();
		}
	}
}

static PyObject*
make_group_list(std::vector<Group>& groups, bool return_collection)
{
	PyObject* py_grp_list;
	try {
		if (return_collection) {
			// just return a simple list of pointers that will be turned into
			// a single Collection on the Python side

			// first, convert the vector-of-vectors into a simple vector
			std::vector<const Atom*> all_group_atoms;
			for (auto grp: groups)
				all_group_atoms.insert(all_group_atoms.end(), grp.begin(), grp.end());
			// put into numpy array
			void** data_ptr;
			auto num_atoms = all_group_atoms.size();
			py_grp_list = python_voidp_array(num_atoms, &data_ptr);
			if (py_grp_list == nullptr)
				throw pysupport::PySupportError("Cannot create overall group list");
			std::memcpy(data_ptr, all_group_atoms.data(), sizeof(void*) * num_atoms);
		} else {
			// return a list of lists of individual Atom pointers
			auto num_groups = groups.size();
			py_grp_list = PyList_New(num_groups);
			if (py_grp_list == nullptr)
				throw pysupport::PySupportError("Cannot create overall group list");
			for (decltype(num_groups) i = 0; i < num_groups; ++i) {
				auto& grp = groups[i];
				auto num_atoms = grp.size();
				PyObject* py_grp = PyList_New(num_atoms);
				if (py_grp == nullptr)
					throw pysupport::PySupportError("Cannot create group atom list");
				for (decltype(num_atoms) j = 0; j < num_atoms; ++j) {
					PyObject* py_ptr =  PyLong_FromVoidPtr(
						const_cast<void*>(static_cast<const void*>(grp[j])));
					if (py_ptr == nullptr)
						throw pysupport::PySupportError("Cannot create group atom ptr");
					PyList_SET_ITEM(py_grp, j, py_ptr);
				}
				PyList_SET_ITEM(py_grp_list, i, py_grp);
			}
		}
	} catch (pysupport::PySupportError& pse) {
		PyErr_SetString(PyExc_TypeError, pse.what());
		return nullptr;
	}
	return py_grp_list;
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
	int	return_collection;
	if (!PyArg_ParseTuple(args, PY_STUPID "OOOOIp", &py_struct_ptr, &py_group_rep,
			&py_group_principals, &py_ring_atom_class, &num_cpus, &return_collection))
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
	if (!PyType_Check(py_ring_atom_class)) {
		PyErr_SetString(PyExc_TypeError, "4th argument must be a class (RingAtom)");
		return nullptr;
	}
	auto type_name = ((PyTypeObject*)py_ring_atom_class)->tp_name;
	auto subloc = strstr(type_name, "RingAtom");
	if (subloc == nullptr || (strlen(type_name) - (subloc - type_name) != strlen("RingAtom"))) {
		PyErr_SetString(PyExc_TypeError, "4th argument is not the RingAtom class");
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
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_group, group_rep, &group_principals,
				start, end, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	delete group_rep;

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_aro_amines(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus, order;
	int	return_collection;
	if (!PyArg_ParseTuple(args, PY_STUPID "OIIp", &py_struct_ptr, &order,
			&num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// Compute the set of Npls in aromatic rings, which will be needed for elimination
	// purposes in the aromatic amine code
	std::set<const Atom*> aro_ring_npls;
	for (auto& ring: s->rings()) {
		if (!ring.aromatic())
			continue;
		for (auto a: ring.atoms()) {
			if (a->idatm_type() == "Npl")
				aro_ring_npls.insert(a);
		}
	}

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_aro_amines, start, end,
				order, &aro_ring_npls, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_aromatics(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus;
	int	return_collection;
	if (!PyArg_ParseTuple(args, PY_STUPID "OIp", &py_struct_ptr, &num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	std::vector<Group> groups;
	for (auto& ring: s->rings()) {
		if (ring.aromatic())
			groups.emplace_back(ring.atoms().begin(), ring.atoms().end());
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_ring_planar_NHR2(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus, ring_size;
	int	return_collection, aromatic_only;
	if (!PyArg_ParseTuple(args, PY_STUPID "OIpIp", &py_struct_ptr, &ring_size, &aromatic_only,
			&num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// ensure the rings are computed (once) here, rather than possibly multiple
	// times in the threads (besides, computation is not thread safe)
	(void)s->rings();

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_ring_planar_NHR2, start, end,
				ring_size, (bool)aromatic_only, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_5ring_planar_NR2(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus;
	int	return_collection;
	int symmetric = (int)false;
	if (!PyArg_ParseTuple(args, PY_STUPID "OpIp", &py_struct_ptr, &symmetric,
			&num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// ensure the rings are computed (once) here, rather than possibly multiple
	// times in the threads (besides, computation is not thread safe)
	(void)s->rings();

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_5ring_planar_NR2,
				start, end, (bool)symmetric, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_6ring_planar_NR2(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus;
	int	return_collection;
	int symmetric = (int)false;
	if (!PyArg_ParseTuple(args, PY_STUPID "OpIp", &py_struct_ptr, &symmetric,
			&num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// ensure the rings are computed (once) here, rather than possibly multiple
	// times in the threads (besides, computation is not thread safe)
	(void)s->rings();

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_6ring_planar_NR2,
				start, end, (bool)symmetric, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_5ring_OR2(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus;
	int	return_collection;
	if (!PyArg_ParseTuple(args, PY_STUPID "OIp", &py_struct_ptr, &num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// ensure the rings are computed (once) here, rather than possibly multiple
	// times in the threads (besides, computation is not thread safe)
	(void)s->rings();

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_5ring_OR2,
				start, end, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_nonring_NR2(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus;
	int	return_collection;
	if (!PyArg_ParseTuple(args, PY_STUPID "OIp", &py_struct_ptr, &num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// ensure the rings are computed (once) here, rather than possibly multiple
	// times in the threads (besides, computation is not thread safe)
	(void)s->rings();

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_nonring_NR2,
				start, end, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

static
PyObject *
find_nonring_ether(PyObject *, PyObject *args)
{
	PyObject*  py_struct_ptr;
	unsigned int  num_cpus;
	int	return_collection;
	if (!PyArg_ParseTuple(args, PY_STUPID "OIp", &py_struct_ptr, &num_cpus, &return_collection))
		return nullptr;
	if (!PyLong_Check(py_struct_ptr)) {
		PyErr_SetString(PyExc_TypeError, "Structure pointer value must be int!");
		return nullptr;
	}
	auto s = static_cast<AtomicStructure*>(PyLong_AsVoidPtr(py_struct_ptr));

	// ensure the rings are computed (once) here, rather than possibly multiple
	// times in the threads (besides, computation is not thread safe)
	(void)s->rings();

	auto& atoms = s->atoms();
	std::vector<Group> groups;
	std::mutex groups_mtx;

	size_t num_threads = num_cpus > 1 ? num_cpus : 1;
	// divvy up the atoms among the threads;
	// letting the threads take atoms from a global pool
	// results in too much lock contention since many
	// of the atoms fail to form a group quickly
	num_threads = std::min(num_threads, atoms.size());
	if (num_threads > 0) {
		float per_thread = atoms.size() / (float) num_threads;
		auto start = atoms.begin();
		std::vector<std::thread> threads;
		for (size_t i = 0; i < num_threads; ++i) {
			decltype(start) end = start + (int)(i * per_thread + 0.5);
			if (i == num_threads - 1) // an overabundance of caution
				end = atoms.end();
			threads.push_back(std::thread(initiate_find_nonring_ether,
				start, end, &groups, &groups_mtx));
			start = end;
		}
		for (auto& th: threads)
			th.join();
	}

	return make_group_list(groups, return_collection);
}

}

static const char* docstr_find_group = "find_group\n"
"Find a chemical group (documented in Python layer)";

static const char* docstr_find_aro_amines = "find_aro_amines\n"
"Find aromatic amines; used internally by find_group";

static const char* docstr_find_aromatics = "find_aromatics\n"
"Find atoms in aromatic rings; used internally by find_group";

static const char* docstr_find_ring_planar_NHR2 = "find_ring_planar_NHR2\n"
"Find planar ring nitrogens that have a hydrogen bound (and the two non-H bond partners)";

static const char* docstr_find_5ring_planar_NR2 = "find_5ring_planar_NR2\n"
"Find planar nitrogens in a 5-member ring (and the two non-H bond partners)\nIf optional 'symmetric' keyword is True (default False) then the ring must be symmetric.";

static const char* docstr_find_6ring_planar_NR2 = "find_6ring_planar_NR2\n"
"Find planar nitrogens in a 6-member ring (and the two non-H bond partners)\nIf optional 'symmetric' keyword is True (default False) then the ring must be symmetric.";

static const char* docstr_find_5ring_OR2 = "find_5ring_OR2\n"
"Find oxygens in a 5-member ring (and their two non-H bond partners)";

static const char* docstr_find_nonring_NR2 = "find_nonring_NR2\n"
"Find non-ring nitrogens with two bonds, both to carbons (which are included in the returned group)";

static const char* docstr_find_nonring_ether = "find_nonring_ether\n"
"Find ether not in a ring system; returns the oxygen and both bonded carbons";

static PyMethodDef cg_methods[] = {
	{ PY_STUPID "find_group", find_group,	METH_VARARGS, PY_STUPID docstr_find_group	},
	{ PY_STUPID "find_aro_amines", find_aro_amines,	METH_VARARGS, PY_STUPID docstr_find_aro_amines	},
	{ PY_STUPID "find_aromatics", find_aromatics,	METH_VARARGS, PY_STUPID docstr_find_aromatics	},
	{ PY_STUPID "find_ring_planar_NHR2", find_ring_planar_NHR2,	METH_VARARGS, PY_STUPID docstr_find_ring_planar_NHR2	},
	{ PY_STUPID "find_5ring_planar_NR2", find_5ring_planar_NR2,	METH_VARARGS, PY_STUPID docstr_find_5ring_planar_NR2	},
	{ PY_STUPID "find_6ring_planar_NR2", find_6ring_planar_NR2,	METH_VARARGS, PY_STUPID docstr_find_6ring_planar_NR2	},
	{ PY_STUPID "find_5ring_OR2", find_5ring_OR2,	METH_VARARGS, PY_STUPID docstr_find_5ring_OR2	},
	{ PY_STUPID "find_nonring_NR2", find_nonring_NR2,	METH_VARARGS, PY_STUPID docstr_find_nonring_NR2	},
	{ PY_STUPID "find_nonring_ether", find_nonring_ether,	METH_VARARGS, PY_STUPID docstr_find_nonring_ether	},
	{ nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef cg_def =
{
	PyModuleDef_HEAD_INIT,
	"_chem_group",
	"Chemical group finding",
	-1,
	cg_methods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

PyMODINIT_FUNC
PyInit__chem_group()
{
	return PyModule_Create(&cg_def);
}
