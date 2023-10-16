// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_structureseq
#define atomstruct_structureseq

#include <map>
#include <set>
#include <string>
#include <vector>

#include "imex.h"
#include "destruct.h"
#include "polymer.h"
#include "Sequence.h"
#include "session.h"
#include "string_types.h"

namespace atomstruct {

class Structure;
class Residue;

class ATOMSTRUCT_IMEX StructureSeq: private Sequence, public DestructionObserver {
public:
    typedef std::vector<unsigned char>::size_type  SeqPos;
    typedef std::vector<Residue *>  Residues;
    typedef std::map<Residue*, SeqPos>  ResMap;

protected:
    friend class Residue;
    void  remove_residues(std::set<Residue*>& residues);
    void  remove_residue(Residue* residue);

    friend class Structure;
    void  clear_residues();

    ChainID  _chain_id;
    std::string  _description;
    bool  _from_seqres;
    bool  _is_chain;
    PolymerType  _polymer_type;
    ResMap  _res_map;
    Residues  _residues;
    Structure*  _structure;

    static int  SESSION_NUM_INTS(int version=CURRENT_SESSION_VERSION) {
        return version < 10 ? 3 : (version < 18 ? 4 : 5);
    }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 0; }

    void  demote_to_sequence();
    void  demote_to_structure_sequence();
    bool  no_structure_left() const {
        for (auto r: _residues) {
            if (r != nullptr)
                return false;
        }
        return true;
    }

public:
    StructureSeq(const ChainID& chain_id, Structure* as, PolymerType pt = PT_NONE);
    virtual ~StructureSeq() { }

    Contents::const_reference  back() const { return Sequence::back(); }
    Contents::const_iterator  begin() const { return Sequence::begin(); }
    void  bulk_set(const Residues& residues,
            const Sequence::Contents* chars = nullptr);
    StructureSeq*  copy() const;
    const ChainID&  chain_id() const { return _chain_id; }
    const Contents&  characters() const { return _contents; }
    Contents::const_iterator  end() const { return Sequence::end(); }
    const std::string&  description() const { return _description; }
    virtual void  destructors_done(const std::set<void*>& destroyed);
    // is character sequence derived from SEQRES records (or equivalent)?
    bool  from_seqres() const { return _from_seqres; }
    Contents::const_reference  front() const { return Sequence::front(); }
    Residue*  get(unsigned i) const { return _residues[i]; }
    void  insert(Residue* follower, Residue* insertion);
    virtual bool  is_chain() const { return false; }
    bool  is_sequence() const { return _structure == nullptr; }
    const std::string&  name() const { return Sequence::name(); }
    StructureSeq&  operator+=(StructureSeq&);
    PolymerType  polymer_type() const { return _polymer_type; }
    void  pop_back();
    void  pop_front();
    void  push_back(Residue* r);
    void  push_front(Residue* r);
    PyObject*  py_instance(bool create) { return Sequence::py_instance(create); }
    void  python_destroyed() { if (!is_chain()) delete this; }
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
    void  set_chain_id(ChainID chain_id);
    void  set_description(const std::string& d) { _description = d; }
    void  set_from_seqres(bool fs);
    Contents::size_type  size() const { return Sequence::size(); }
    Structure*  structure() const { return _structure; }
};

}  // namespace atomstruct

#endif  // atomstruct_structureseq
