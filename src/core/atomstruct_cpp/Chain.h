// vim: set expandtab ts=4 sw=4:
#ifndef atomstruct_chain
#define atomstruct_chain

#include "Sequence.h"
#include <string>
#include <vector>
#include "imex.h"

namespace atomstruct {

class Residue;

class ATOMSTRUCT_IMEX Chain: public Sequence {
public:
    typedef std::vector<Residue *>  Residues;
private:
    std::string  _chain_id;
    bool  _from_seqres;
    Residues  _residues;

public:
    Chain(const std::string& chain_id):
        _chain_id(chain_id), _from_seqres(false) {}
    const std::string&  chain_id() const { return _chain_id; }
    // is character sequence derived from SEQRES records (or equivalent)?
    bool  from_seqres() const { return _from_seqres; }
    const Residues&  residues() const { return _residues; }
    Residue*  get(unsigned i) const { return _residues[i]; }
    void  set(unsigned i, Residue* r, char character = -1);
    void  set_from_seqres(bool fs);
    void  bulk_set(Residues& residues,
            Sequence::Contents* chars = nullptr);
};

}  // namespace atomstruct

#endif  // atomstruct_chain
