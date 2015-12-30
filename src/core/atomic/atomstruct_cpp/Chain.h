// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_chain
#define atomstruct_chain

#include <map>
#include <string>
#include <vector>

#include "imex.h"
#include "Sequence.h"
#include "string_types.h"

namespace atomstruct {

class AtomicStructure;
class Residue;

class ATOMSTRUCT_IMEX Chain: public Sequence {
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
    typedef std::vector<Residue *>  Residues;

    const int  SESSION_NUM_INTS = 3;
    const int  SESSION_NUM_FLOATS = 0;
private:
    friend class Residue;
    void  remove_residue(Residue* r);

    ChainID  _chain_id;
    bool  _from_seqres;
    typedef std::map<Residue*, SeqPos>  ResMap;
    ResMap  _res_map;
    Residues  _residues;
    AtomicStructure*  _structure;

    bool  no_structure_left() const {
        for (auto r: _residues) {
            if (r != nullptr)
                return false;
        }
        return true;
    }

public:
    Chain(const ChainID& chain_id, AtomicStructure* as);
    virtual ~Chain();

    const ChainID&  chain_id() const { return _chain_id; }
    // is character sequence derived from SEQRES records (or equivalent)?
    bool  from_seqres() const { return _from_seqres; }
    const Residues&  residues() const { return _residues; }
    Residue*  get(unsigned i) const { return _residues[i]; }
    bool  is_sequence() const { return _structure == nullptr; }
    Chain&  operator+=(Chain&);
    void  pop_back();
    void  pop_front();
    void  push_back(Residue* r);
    void  push_front(Residue* r);
    int  session_num_floats() const { return Sequence::session_num_floats() + SESSION_NUM_FLOATS; }
    int  session_num_ints() const {
        return Sequence::session_num_ints() + SESSION_NUM_INTS
            + 2 * _res_map.size() + _residues.size();
    }
    void  session_restore(int**, float**);
    void  session_save(int**, float**) const;
    void  set(unsigned i, Residue* r, char character = -1);
    void  set_from_seqres(bool fs);
    AtomicStructure*  structure() const { return _structure; }
    void  bulk_set(const Residues& residues,
            const Sequence::Contents* chars = nullptr);
};

}  // namespace atomstruct

#endif  // atomstruct_chain
