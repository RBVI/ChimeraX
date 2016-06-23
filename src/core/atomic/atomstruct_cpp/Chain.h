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

class Structure;
class Residue;

class ATOMSTRUCT_IMEX Chain: private Sequence {
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
    typedef std::vector<Residue *>  Residues;
private:
    friend class Residue;
    void  remove_residue(Residue* r);

    friend class Structure;
    void  clear_residues();

    ChainID  _chain_id;
    bool  _from_seqres;
    typedef std::map<Residue*, SeqPos>  ResMap;
    ResMap  _res_map;
    Residues  _residues;
    Structure*  _structure;

    static int  SESSION_NUM_INTS(int /*version*/=0) { return 3; }
    static int  SESSION_NUM_FLOATS(int /*version*/=0) { return 0; }

    bool  no_structure_left() const {
        for (auto r: _residues) {
            if (r != nullptr)
                return false;
        }
        return true;
    }

public:
    Chain(const ChainID& chain_id, Structure* as);
    virtual ~Chain();

    Contents::const_reference  back() const { return Sequence::back(); }
    Contents::const_iterator  begin() const { return Sequence::begin(); }
    const ChainID&  chain_id() const { return _chain_id; }
    Contents::const_iterator  end() const { return Sequence::end(); }
    // is character sequence derived from SEQRES records (or equivalent)?
    bool  from_seqres() const { return _from_seqres; }
    Contents::const_reference  front() const { return Sequence::front(); }
    const Residues&  residues() const { return _residues; }
    Residue*  get(unsigned i) const { return _residues[i]; }
    bool  is_sequence() const { return _structure == nullptr; }
    Chain&  operator+=(Chain&);
    void  pop_back();
    void  pop_front();
    void  push_back(Residue* r);
    void  push_front(Residue* r);
    int  session_num_floats(int version=0) const {
        return Sequence::session_num_floats(version) + SESSION_NUM_FLOATS(version);
    }
    int  session_num_ints(int version=0) const {
        return Sequence::session_num_ints(version) + SESSION_NUM_INTS(version)
            + 2 * _res_map.size() + _residues.size();
    }
    void  session_restore(int, int**, float**);
    void  session_save(int**, float**) const;
    void  set(unsigned i, Residue* r, char character = -1);
    void  set_from_seqres(bool fs);
    Contents::size_type  size() const { return Sequence::size(); }
    Structure*  structure() const { return _structure; }
    void  bulk_set(const Residues& residues,
            const Sequence::Contents* chars = nullptr);
};

}  // namespace atomstruct

#endif  // atomstruct_chain
