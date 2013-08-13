#include "Residue.h"
#include "Atom.h"
#include <utility>  // for pair
#include <sstream>

Residue::Residue(Molecule *m, std::string &name, std::string &chain,
	int pos, char insert): _molecule(m), _name(name), _position(pos),
	_chain_id(chain), _insertion_code(insert), _is_helix(false),
	_is_sheet(false), _is_het(false), _ss_id(-1)
{
}

void
Residue::add_atom(Atom *a)
{
	a->_residue = this;
	_atoms.push_back(a);
}

Residue::AtomsMap
Residue::atoms_map() const
{
	AtomsMap map;
	for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
		Atom *a = *ai;
		map.insert(AtomsMap::value_type(a->name(), a));
	}
	return map;
}

int
Residue::count_atom(const std::string &name) const
{
	int count = 0;
	for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
		Atom *a = *ai;
		if (a->name() == name)
			++count;
	}
	return count;
}

int
Residue::count_atom(const char *name) const
{
	int count = 0;
	for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
		Atom *a = *ai;
		if (a->name() == name)
			++count;
	}
	return count;
}

Atom *
Residue::find_atom(const std::string &name) const
{
	
	for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
		Atom *a = *ai;
		if (a->name() == name)
			return a;
	}
	return NULL;
}

Atom *
Residue::find_atom(const char *name) const
{
	
	for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
		Atom *a = *ai;
		if (a->name() == name)
			return a;
	}
	return NULL;
}

std::string
Residue::str() const
{
	std::stringstream pos_string;
	std::string ret = _name;
	ret += " ";
	pos_string << _position;
	ret += pos_string.str();
	if (_insertion_code != ' ') {
		ret += ".";
		ret += _insertion_code;
	}
	return ret;
}
