// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_AtomicStructure
#define atomstruct_AtomicStructure

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <basegeom/Graph.h>
#include <basegeom/destruct.h>
#include <element/Element.h>
#include "Chain.h"
#include "Pseudobond.h"
#include "Ring.h"
#include "string_types.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace atomstruct {

class Atom;
class Bond;
class Chain;
class CoordSet;
class Residue;

using element::Element;

class ATOMSTRUCT_IMEX AtomicStructure: public basegeom::Graph<Atom, Bond> {
    friend class Atom; // for IDATM stuff and _polymers_computed
    friend class Bond; // for checking if make_chains() has been run yet
public:
    typedef Vertices  Atoms;
    typedef Edges  Bonds;
    typedef std::vector<Chain*>  Chains;
    typedef std::vector<CoordSet*>  CoordSets;
    typedef std::map<ChainID, std::vector<ResName>>  InputSeqInfo;
    static const char*  PBG_METAL_COORDINATION;
    static const char*  PBG_MISSING_STRUCTURE;
    static const char*  PBG_HYDROGEN_BONDS;
    typedef std::vector<Residue*>  Residues;
    // The MSR-finding step of ring perception depends on the iteration
    // being in ascending order (which std::set guarantees), so use std::set
    typedef std::set<Ring> Rings;
private:
    friend class Chain;
    void  remove_chain(Chain* chain);

    CoordSet *  _active_coord_set;
    void  _calculate_rings(bool cross_residue, unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    mutable Chains*  _chains;
    void  _compute_atom_types();
    void  _compute_idatm_types() { _idatm_valid = true; _compute_atom_types(); }
    CoordSets  _coord_sets;
    void  _delete_atom(Atom* a);
    void  _delete_residue(Residue* r, const Residues::iterator& ri);
    void  _fast_calculate_rings(std::set<const Residue *>* ignore) const;
    bool  _fast_ring_calc_available(bool cross_residue,
            unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    bool  _idatm_valid;
    InputSeqInfo  _input_seq_info;
    PyObject*  _logger;
    std::string  _name;
    Chain*  _new_chain(const ChainID& chain_id) const {
        auto chain = new Chain(chain_id, (AtomicStructure*)this);
        _chains->emplace_back(chain);
        return chain;
    }
    int  _num_hyds = 0;
    AS_PBManager  _pb_mgr;
    mutable bool  _polymers_computed;
    mutable bool  _recompute_rings;
    Residues  _residues;
    mutable Rings  _rings;
    bool  _rings_cached (bool cross_residues, unsigned int all_size_threshold,
        std::set<const Residue *>* ignore = nullptr) const;
    mutable unsigned int  _rings_last_all_size_threshold;
    mutable bool  _rings_last_cross_residues;
    mutable std::set<const Residue *>*  _rings_last_ignore;
    bool  _gc_ribbon = false;
public:
    AtomicStructure(PyObject* logger = nullptr);
    virtual  ~AtomicStructure();

    CoordSet *  active_coord_set() const { return _active_coord_set; };
    bool  asterisks_translated;
    const Atoms &    atoms() const { return vertices(); }
    // ball_scale() inherited from Graph
    std::map<Residue *, char>  best_alt_locs() const;
    const Bonds &    bonds() const { return edges(); }
    const Chains &  chains() const { if (_chains == nullptr) make_chains(); return *_chains; }
    const CoordSets &  coord_sets() const { return _coord_sets; }
    AtomicStructure*  copy() const;
    void  delete_atom(Atom* a);
    void  delete_atoms(std::vector<Atom*> atoms);
    void  delete_bond(Bond* b) { delete_edge(b); }
    void  delete_residue(Residue* r);
    // display() inherited from Graph
    void  extend_input_seq_info(ChainID& chain_id, ResName& res_name) {
        _input_seq_info[chain_id].push_back(res_name);
    }
    CoordSet *  find_coord_set(int) const;
    Residue *  find_residue(const ChainID& chain_id, int pos, char insert) const;
    Residue *  find_residue(const ChainID& chain_id, int pos, char insert,
        ResName& name) const;
    const InputSeqInfo&  input_seq_info() const { return _input_seq_info; }
    std::string  input_seq_source;
    bool  is_traj;
    PyObject*  logger() const { return _logger; }
    bool  lower_case_chains;
    void  make_chains() const;
    std::map<std::string, std::vector<std::string>> metadata;
    const std::string&  name() const { return _name; }
    Atom *  new_atom(const char* name, const Element& e);
    Bond *  new_bond(Atom *, Atom *);
    CoordSet *  new_coord_set();
    CoordSet *  new_coord_set(int index);
    CoordSet *  new_coord_set(int index, int size);
    Residue *  new_residue(const ResName& name, const ChainID& chain,
        int pos, char insert, Residue *neighbor=NULL, bool after=true);
    size_t  num_atoms() const { return atoms().size(); }
    size_t  num_bonds() const { return bonds().size(); }
    size_t  num_hyds() const { return _num_hyds; }
    size_t  num_residues() const { return residues().size(); }
    size_t  num_chains() const { return chains().size(); }
    size_t  num_coord_sets() const { return coord_sets().size(); }
    AS_PBManager&  pb_mgr() { return _pb_mgr; }
    int  pdb_version;
    std::vector<Chain::Residues>  polymers(
        bool consider_missing_structure = true,
        bool consider_chain_ids = true) const;
    const Residues &  residues() const { return _residues; }
    const Rings&  rings(bool cross_residues = false,
        unsigned int all_size_threshold = 0,
        std::set<const Residue *>* ignore = nullptr) const;
    int  session_info(PyObject* ints, PyObject* floats, PyObject* strings) const;
    void  set_active_coord_set(CoordSet *cs);
    // set_ball_scale() inherited from Graph
    // set_display() inherited from Graph
    void  set_input_seq_info(const ChainID& chain_id, const std::vector<ResName>& res_names) { _input_seq_info[chain_id] = res_names; }
    void  set_name(const std::string& name) { _name = name; }
    void  use_best_alt_locs();
    bool  get_gc_ribbon() const { return _gc_ribbon; }
    void  set_gc_ribbon(bool gc = true) { _gc_ribbon = gc; }
};

}  // namespace atomstruct

#include "Atom.h"
inline void
atomstruct::AtomicStructure::_delete_atom(atomstruct::Atom* a)
{
    if (a->element().number() == 1)
        --_num_hyds;
    delete_vertex(a);
}

inline void
atomstruct::AtomicStructure::remove_chain(Chain* chain)
{
    _chains->erase(std::find(_chains->begin(), _chains->end(), chain));
}

#endif  // atomstruct_AtomicStructure
