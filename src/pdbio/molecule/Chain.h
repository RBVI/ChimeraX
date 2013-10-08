#ifndef molecule_chain
#define molecule_chain

#include "Sequence.h"
#include <string>
#include <vector>

class Residue;

class Chain: public Sequence {
public:
	typedef std::vector<Residue *>  Residues;
private:
	std::string  _chain_id;
	Residues  _residues;

public:
	const std::string &  chain_id() const { return _chain_id; }
	const Residues&  residues() const { return _residues; }
	Residue *  get(unsigned i) const { return _residues[i]; }
	void  set(unsigned i, Residue *r, char character = -1);
};

#endif  // molecule_chain
