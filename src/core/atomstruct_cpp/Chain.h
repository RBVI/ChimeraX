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
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
    typedef std::vector<Residue *>  Residues;

private:
    friend class Residue;
    void  remove_residue(Residue* r);

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
    void  pop_back();
    void  pop_front();
    void  set(unsigned i, Residue* r, char character = -1);
    void  set_from_seqres(bool fs);
    AtomicStructure*  structure() const;
    void  bulk_set(const Residues& residues,
            const Sequence::Contents* chars = nullptr);
};

}  // namespace atomstruct

#include "Residue.h"
inline void
atomstruct::Chain::pop_back()
{
    atomstruct::Sequence::pop_back();
    auto back = _residues.back();
    if (back != nullptr) {
        _res_map.erase(back);
        back->set_chain(nullptr);
    }
    _residues.pop_back();
}

inline void
atomstruct::Chain::pop_front()
{
    atomstruct::Sequence::pop_front();
    auto front = _residues.front();
    if (front != nullptr) {
        _res_map.erase(front);
        front->set_chain(nullptr);
    }
    _residues.erase(_residues.begin());
}

inline atomstruct::AtomicStructure*
atomstruct::Chain::structure() const {
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        if (*ri != nullptr)
            return (*ri)->structure();
    }
    throw std::logic_error("No actual residues in chain?!?");
}

#endif  // atomstruct_chain
