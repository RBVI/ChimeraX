// vim: set expandtab ts=4 sw=4:
#ifndef atomic_chain
#define atomic_chain

#include "Sequence.h"
#include <string>
#include <vector>
#include "imex.h"

class Residue;

class ATOMSTRUCT_IMEX Chain: public Sequence {
public:
    typedef std::vector<Residue *>  Residues;
private:
    std::string  _chain_id;
    Residues  _residues;

public:
    Chain(const std::string& chain_id): _chain_id(chain_id) {}
    const std::string&  chain_id() const { return _chain_id; }
    const Residues&  residues() const { return _residues; }
    Residue*  get(unsigned i) const { return _residues[i]; }
    void  set(unsigned i, Residue* r, char character = -1);
    void  bulk_set(Residues& residues,
            Sequence::Contents* chars = nullptr);
};

#endif  // atomic_chain
