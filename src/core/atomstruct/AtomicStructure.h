// vim: set expandtab ts=4 sw=4:
#ifndef atomstruct_AtomicStructure
#define atomstruct_AtomicStructure

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <set>

#include "Chain.h"
#include "Pseudobond.h"
#include "Ring.h"
#include "basegeom/Graph.h"

namespace atomstruct {

class Atom;
class Bond;
class CoordSet;
class Element;
class Residue;

class ATOMSTRUCT_IMEX AtomicStructure: public basegeom::Graph<Atom, Bond> {
    friend class Atom; // for IDATM stuff
public:
    typedef Vertices  Atoms;
    typedef Edges  Bonds;
    typedef std::vector<std::unique_ptr<Chain>>  Chains;
    typedef std::vector<std::unique_ptr<CoordSet>>  CoordSets;
    typedef std::map<std::string, std::vector<std::string>>  InputSeqInfo;
    static const char*  PBG_METAL_COORDINATION;
    static const char*  PBG_MISSING_STRUCTURE;
    typedef std::vector<std::unique_ptr<Residue>>  Residues;
    typedef std::set<Ring> Rings;
private:
    CoordSet *  _active_coord_set;
    bool  _being_destroyed;
    void  _calculate_rings(bool cross_residue, unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    mutable Chains *  _chains;
    void  _compute_atom_types();
    void  _compute_idatm_types() { _idatm_valid = true; _compute_atom_types(); }
    CoordSets  _coord_sets;
    bool  _idatm_valid;
    InputSeqInfo  _input_seq_info;
    AS_PBManager  _pb_mgr;
    mutable bool  _recompute_rings;
    Residues  _residues;
    mutable Rings  _rings;
    mutable unsigned int  _rings_last_all_size_threshold;
    mutable bool  _rings_last_cross_residues;
    mutable std::set<const Residue *>*  _rings_last_ignore;
public:
    AtomicStructure();
    virtual  ~AtomicStructure() { _being_destroyed = true; }
    const Atoms &    atoms() const { return vertices(); }
    CoordSet *  active_coord_set() const { return _active_coord_set; };
    bool  asterisks_translated;
    bool  being_destroyed() const { return _being_destroyed; }
    std::unordered_map<Residue *, char>  best_alt_locs() const;
    const Bonds &    bonds() const { return edges(); }
    const Chains &  chains() const { if (_chains == nullptr) make_chains(); return *_chains; }
    const CoordSets &  coord_sets() const { return _coord_sets; }
    void  delete_atom(Atom* a);
    void  delete_bond(Bond* b);
    void  extend_input_seq_info(std::string& chain_id, std::string& res_name) {
        _input_seq_info[chain_id].push_back(res_name);
    }
    CoordSet *  find_coord_set(int) const;
    Residue *  find_residue(std::string &chain_id, int pos, char insert) const;
    Residue *  find_residue(std::string &chain_id, int pos, char insert,
        std::string &name) const;
    const InputSeqInfo&  input_seq_info() const { return _input_seq_info; }
    bool  is_traj;
    bool  lower_case_chains;
    void  make_chains() const;
    Atom *  new_atom(const std::string &name, Element e);
    Bond *  new_bond(Atom *, Atom *);
    CoordSet *  new_coord_set();
    CoordSet *  new_coord_set(int index);
    CoordSet *  new_coord_set(int index, int size);
    Residue *  new_residue(const std::string &name, const std::string &chain,
        int pos, char insert, Residue *neighbor=NULL, bool after=true);
    int  num_atoms() const { return atoms().size(); }
    int  num_bonds() const { return bonds().size(); }
    AS_PBManager&  pb_mgr() { return _pb_mgr; }
    std::unordered_map<std::string, std::vector<std::string>> pdb_headers;
    int  pdb_version;
    std::vector<Chain::Residues>  polymers() const;
    const Residues &  residues() const { return _residues; }
    const Rings&  rings(bool cross_residues = false,
        unsigned int all_size_threshold = 0,
        std::set<const Residue *>* ignore = nullptr) const;
    void  set_active_coord_set(CoordSet *cs);
    void  set_input_seq_info(std::string& chain_id, std::vector<std::string>& res_names) { _input_seq_info[chain_id] = res_names; }
    void  use_best_alt_locs();
};

}  // namespace atomstruct

#include "Atom.h"
inline void
atomstruct::AtomicStructure::delete_atom(atomstruct::Atom* a) {
    for (auto b: a->bonds()) delete_bond(b);
    delete_vertex(a);
}

#include "Bond.h"
inline void
atomstruct::AtomicStructure::delete_bond(atomstruct::Bond* b) {
    for (auto a: b->atoms()) a->remove_bond(b);
    delete_edge(b);
}

// for unique_ptr template expansion
#include "CoordSet.h"
#include "Residue.h"

#endif  // atomstruct_AtomicStructure
