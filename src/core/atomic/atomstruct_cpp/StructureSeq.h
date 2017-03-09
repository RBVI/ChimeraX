// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_structureseq
#define atomstruct_structureseq

#include <map>
#include <string>
#include <vector>

#include "imex.h"
#include "Sequence.h"
#include "session.h"
#include "string_types.h"

namespace atomstruct {

class Structure;
class Residue;

class ATOMSTRUCT_IMEX StructureSeq: private Sequence {
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
    typedef std::vector<Residue *>  Residues;
    typedef std::map<Residue*, SeqPos>  ResMap;

protected:
    friend class Residue;
    void  remove_residue(Residue* r);

    friend class Structure;
    void  clear_residues();

    ChainID  _chain_id;
    bool  _from_seqres;
    ResMap  _res_map;
    Residues  _residues;
    Structure*  _structure;

    static int  SESSION_NUM_INTS(int /*version*/=CURRENT_SESSION_VERSION) { return 3; }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }

    void  demote_to_sequence();
    bool  no_structure_left() const {
        for (auto r: _residues) {
            if (r != nullptr)
                return false;
        }
        return true;
    }

public:
    StructureSeq(const ChainID& chain_id, Structure* as);
    virtual ~StructureSeq() { }

    Contents::const_reference  back() const { return Sequence::back(); }
    Contents::const_iterator  begin() const { return Sequence::begin(); }
    void  bulk_set(const Residues& residues,
            const Sequence::Contents* chars = nullptr);
    StructureSeq*  copy() const;
    const ChainID&  chain_id() const { return _chain_id; }
    Contents::const_iterator  end() const { return Sequence::end(); }
    // is character sequence derived from SEQRES records (or equivalent)?
    bool  from_seqres() const { return _from_seqres; }
    Contents::const_reference  front() const { return Sequence::front(); }
    Residue*  get(unsigned i) const { return _residues[i]; }
    virtual bool  is_chain() const { return false; }
    bool  is_sequence() const { return _structure == nullptr; }
    const std::string&  name() const { return Sequence::name(); }
    StructureSeq&  operator+=(StructureSeq&);
    void  pop_back();
    void  pop_front();
    void  push_back(Residue* r);
    void  push_front(Residue* r);
    const ResMap&  res_map() const { return _res_map; }
    const Residues&  residues() const { return _residues; }
    int  session_num_floats(int version=CURRENT_SESSION_VERSION) const {
        return Sequence::session_num_floats(version) + SESSION_NUM_FLOATS(version);
    }
    int  session_num_ints(int version=CURRENT_SESSION_VERSION) const {
        return Sequence::session_num_ints(version) + SESSION_NUM_INTS(version)
            + 2 * _res_map.size() + _residues.size();
    }
    void  session_restore(int, int**, float**);
    void  session_save(int**, float**) const;
    void  set(unsigned i, Residue* r, char character = -1);
    void  set_from_seqres(bool fs);
    Contents::size_type  size() const { return Sequence::size(); }
    Structure*  structure() const { return _structure; }
};

}  // namespace atomstruct

#endif  // atomstruct_structureseq
