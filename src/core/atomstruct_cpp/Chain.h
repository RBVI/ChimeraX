// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_chain
#define atomstruct_chain

#include <map>
#include <string>
#include <vector>

#include "imex.h"
#include "Sequence.h"

namespace atomstruct {

class AtomicStructure;
class Residue;

class ATOMSTRUCT_IMEX Chain: public Sequence {
    friend class AtomicStructure;
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
    typedef std::vector<Residue *>  Residues;
private:
    std::string  _chain_id;
    bool  _from_seqres;
    std::map<Residue*, SeqPos>  _res_map;
    Residues  _residues;

public:
    Chain(const std::string& chain_id): Sequence(),
        _chain_id(chain_id), _from_seqres(false) {}
    const std::string&  chain_id() const { return _chain_id; }
    // is character sequence derived from SEQRES records (or equivalent)?
    bool  from_seqres() const { return _from_seqres; }
    const Residues&  residues() const { return _residues; }
    Residue*  get(unsigned i) const { return _residues[i]; }
    void  pop_back() {
        Chain::pop_back();
        if (_residues.back() != nullptr) _res_map.erase(_residues.back());
        _residues.pop_back();
    }
    void  pop_front() {
        Chain::pop_front();
        if (_residues.front() != nullptr) _res_map.erase(_residues.front());
        _residues.erase(_residues.begin());
    }
    void  set(unsigned i, Residue* r, char character = -1);
    void  set_from_seqres(bool fs);
    AtomicStructure*  structure() const;
    void  bulk_set(Residues& residues,
            std::vector<unsigned char>* chars = nullptr);
};

}  // namespace atomstruct

#include "Residue.h"
inline atomstruct::AtomicStructure*
atomstruct::Chain::structure() const { return _residues.front()->structure(); }

#endif  // atomstruct_chain
