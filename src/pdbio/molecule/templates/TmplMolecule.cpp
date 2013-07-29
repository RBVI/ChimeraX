#include "restmpl.h"

#include <algorithm> // for std::find

void
TmplMolecule::set_active_coord_set(TmplCoordSet *cs)
{
	TmplCoordSet *new_active;
	if (cs == NULL) {
		if (_coord_sets.empty())
			return;
		new_active = _coord_sets.front();
	} else {
		CoordSets::iterator csi = std::find(_coord_sets.begin(), _coord_sets.end(), cs);
		if (csi == _coord_sets.end()) {
			throw std::out_of_range("Request active template coord set not in list");
		} else {
			new_active = cs;
		}
	}
	_active_cs = new_active;
}

TmplAtom *
TmplMolecule::new_atom(std::string n, Element e)
{
	TmplAtom *_inst_ = new TmplAtom(this, n, e);
	_atoms.insert(_inst_);
	return _inst_;
}
#ifdef UNPORTED
void
TmplMolecule::deleteAtom(TmplAtom *element)
{
	for (TmplAtom::BondsMap::iterator i = element->_bonds.begin(); i != element->_bonds.end(); i = element->_bonds.begin()) {
		TmplBond *e = (*i).second;
		// element->_bonds.erase(i); -- done by deleteBond(e) below
		deleteBond(e);
	}
	element->residue()->removeAtom(element);
	_atoms.erase(element);
	delete element;
}
#endif  // UNPORTED
TmplBond *
TmplMolecule::new_bond(TmplAtom *a0, TmplAtom *a1)
{
	TmplBond *_inst_ = new TmplBond(this, a0, a1);
	_bonds.insert(_inst_);
	return _inst_;
}
#ifdef UNPORTED
TmplBond *
TmplMolecule::newBond(TmplAtom *a[2])
{
	TmplBond *_inst_ = new TmplBond(this, a);
	_bonds.insert(_inst_);
	return _inst_;
}
void
TmplMolecule::deleteBond(TmplBond *element)
{
	for (TmplBond::Atoms::const_iterator i = element->atoms().begin(); i != element->atoms().end(); ++i)
		(*i)->removeBond(element);
	_bonds.erase(element);
	delete element;
}
#endif  // UNPORTED
TmplCoordSet *
TmplMolecule::new_coord_set(int k)
{
	TmplCoordSet *_inst_ = new TmplCoordSet(this, k);
	if ((int)_coord_sets.size() <= _inst_->id())
		_coord_sets.resize(_inst_->id() + 1, NULL);
	_coord_sets[_inst_->id()] = _inst_;
	return _inst_;
}
#ifdef UNPORTED
TmplCoordSet *
TmplMolecule::newCoordSet(int k, int size)
{
	TmplCoordSet *_inst_ = new TmplCoordSet(this, k, size);
	_coord_sets[_inst_->id()] = _inst_;
	return _inst_;
}
void
TmplMolecule::deleteCoordSet(TmplCoordSet *element)
{
	_coord_sets.erase(element->id());
	delete element;
}
#endif  // UNPORTED
TmplCoordSet *
TmplMolecule::find_coord_set(int index) const
{
	for (CoordSets::const_iterator csi = _coord_sets.begin(); csi != _coord_sets.end();
	++csi) {
		if ((*csi)->id() == index)
			return *csi;
	}
	return NULL;
}
TmplResidue *
TmplMolecule::new_residue(const char *t)
{
	TmplResidue *_inst_ = new TmplResidue(this, t);
	_residues[_inst_->name()] = _inst_;
	return _inst_;
}
#ifdef UNPORTED
TmplResidue *
TmplMolecule::new_residue(Symbol t, Symbol chain, int pos, char insert)
{
	TmplResidue *_inst_ = new TmplResidue(this, t, chain, pos, insert);
	_residues[_inst_->type()] = _inst_;
	return _inst_;
}
void
TmplMolecule::deleteResidue(TmplResidue *element)
{
	for (TmplResidue::AtomsMap::iterator i = element->_atoms.begin(); i != element->_atoms.end(); i = element->_atoms.begin()) {
		TmplAtom *e = (*i).second;
		// element->_atoms.erase(i); -- done by deleteAtom(e) below
		deleteAtom(e);
	}
	_residues.erase(element->type());
	delete element;
}
#endif  // UNPORTED
TmplResidue *
TmplMolecule::find_residue(const std::string &index) const
{
	Residues::const_iterator i = _residues.find(index);
	if (i == _residues.end())
		return NULL;
	return i->second;
}

TmplMolecule::TmplMolecule(): _active_cs(NULL)
{
}

TmplMolecule::~TmplMolecule()
{
	for (Residues::iterator j = _residues.begin(); j != _residues.end(); ++j) {
		delete (*j).second;
	}
	for (CoordSets::iterator j = _coord_sets.begin(); j != _coord_sets.end(); ++j) {
		delete (*j);
	}
	for (Bonds::iterator j = _bonds.begin(); j != _bonds.end(); ++j) {
		delete (*j);
	}
	for (Atoms::iterator j = _atoms.begin(); j != _atoms.end(); ++j) {
		delete (*j);
	}
}

