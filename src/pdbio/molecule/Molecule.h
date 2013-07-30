#ifndef molecule_Molecule
#define molecule_Molecule

#include <vector>
#include <string>
#include <map>

class Atom;
class CoordSet;
class Bond;
class Element;
class Residue;

class Molecule {
public:
	typedef std::vector<Atom *>  Atoms;
	typedef std::vector<Bond *>  Bonds;
	typedef std::vector<CoordSet *>  CoordSets;
	typedef std::vector<Residue *>  Residues;
private:
	CoordSet *  _active_coord_set;
	Atoms  _atoms;
	Bonds  _bonds;
	CoordSets  _coord_sets;
	Residues  _residues;
public:
	Molecule();
	const Atoms &	atoms() const { return _atoms; }
	CoordSet *  active_coord_set() const { return _active_coord_set; };
	bool  asterisks_translated;
	const Bonds &	bonds() const { return _bonds; }
	const CoordSets &  coord_sets() const { return _coord_sets; }
	void  delete_bond(Bond *);
	CoordSet *  find_coord_set(int) const;
	Residue *  find_residue(std::string &chain_id, int pos, char insert) const;
	Residue *  find_residue(std::string &chain_id, int pos, char insert,
		std::string &name) const;
	bool  lower_case_chains;
	Atom *  new_atom(std::string &name, Element e);
	Bond *  new_bond(Atom *, Atom *);
	CoordSet *  new_coord_set();
	CoordSet *  new_coord_set(int index);
	CoordSet *  new_coord_set(int index, int size);
	Residue *  new_residue(std::string &name, std::string &chain, int pos, char insert,
		Residue *neighbor=NULL, bool after=true);
	std::map<std::string, std::vector<std::string> > pdb_headers;
	int  pdb_version;
	const Residues &  residues() const { return _residues; }
	void  set_active_coord_set(CoordSet *cs);
};

#endif  // molecule_Molecule
