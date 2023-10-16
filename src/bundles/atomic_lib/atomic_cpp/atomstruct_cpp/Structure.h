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

#ifndef atomstruct_Structure
#define atomstruct_Structure

#include <algorithm>
#include <element/Element.h>
#include <map>
#include <pyinstance/PythonInstance.declare.h>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Chain.h"
#include "ChangeTracker.h"
#include "CompSS.h"
#include "destruct.h"
#include "PBManager.h"
#include "polymer.h"
#include "Real.h"
#include "res_numbering.h"
#include "Rgba.h"
#include "Ring.h"
#include "session.h"
#include "string_types.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
using element::Element;

namespace atomstruct {

class Atom;
class Bond;
class Chain;
class CoordSet;
class Residue;
class Rgba;

    
class ATOMSTRUCT_IMEX GraphicsChanges {
private:
    int _gc_changes;
    
public:
    enum ChangeType {
      SHAPE_CHANGE = (1 << 0),
      COLOR_CHANGE = (1 << 1),
      SELECT_CHANGE = (1 << 2),
      RIBBON_CHANGE = (1 << 3),
      ADDDEL_CHANGE = (1 << 4),
      DISPLAY_CHANGE = (1 << 5),
      RING_CHANGE = (1 << 6),
    };

    GraphicsChanges(): _gc_changes(0) {}
    virtual  ~GraphicsChanges() {}
    virtual void  gc_clear() { _gc_changes = 0; }
    virtual int   get_graphics_changes() const { return _gc_changes; }
    virtual void  set_gc_color() { set_graphics_change(COLOR_CHANGE); }
    virtual void  set_gc_select() { set_graphics_change(SELECT_CHANGE); }
    virtual void  set_gc_shape() { set_graphics_change(SHAPE_CHANGE); }
    virtual void  set_gc_ribbon() { set_graphics_change(RIBBON_CHANGE); }
    virtual void  set_gc_ring() { set_graphics_change(RING_CHANGE); }
    virtual void  set_gc_adddel() { set_graphics_change(ADDDEL_CHANGE); }
    virtual void  set_gc_display() { set_graphics_change(DISPLAY_CHANGE); }
    virtual void  set_graphics_changes(int change) { _gc_changes = change; }
    virtual void  set_graphics_change(ChangeType type) { _gc_changes |= type; }
    virtual void  clear_graphics_change(ChangeType type) { _gc_changes &= ~type; }
};

// Structure and AtomicStructure have all the methods and typedefs (i.e.
// AtomicStructure simply inherits everything from Structure and doesn't
// add any) so that they can be treated identically in the Python
// layer.  Some atomic-structure-specific methods will have no-op
// implementations in Structure and real implementations in AtomicStructure.
class ATOMSTRUCT_IMEX Structure: public GraphicsChanges,
        public pyinstance::PythonInstance<Structure> {
    friend class Atom; // for IDATM stuff and structure categories
    friend class Bond; // for _form_chain_check, struct categories
    friend class Residue; // for _polymers_computed
    friend class StructurePBGroup; // for _form_chain_check
    friend class StructureSeq; // for remove_chain()
public:
    typedef std::vector<Atom*>  Atoms;
    typedef std::vector<Bond*>  Bonds;
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
    enum RibbonOrientation { RIBBON_ORIENT_GUIDES = 1,
                             RIBBON_ORIENT_ATOMS = 2,
                             RIBBON_ORIENT_CURVATURE = 3,
                             RIBBON_ORIENT_PEPTIDE = 4 };
    enum RibbonMode { RIBBON_MODE_DEFAULT = 0,
                      RIBBON_MODE_ARC = 1,
                      RIBBON_MODE_WRAP = 2 };
    enum TetherShape { RIBBON_TETHER_CONE = 0,
                       RIBBON_TETHER_REVERSE_CONE = 1,
                       RIBBON_TETHER_CYLINDER = 2 };
protected:
    bool  _active_coord_set_change_notify = true;
    CoordSet *  _active_coord_set;
    mutable bool  _alt_loc_change_notify = true;
    mutable bool  _ss_change_notify = true;
    bool  _atom_types_notify = true;
    Atoms  _atoms;
    float  _ball_scale = 0.25;
    Bonds  _bonds;
    mutable Chains*  _chains;
    ChangeTracker*  _change_tracker;
    CoordSets  _coord_sets;
    bool  _display = true;
    bool  _idatm_failed;
    bool  _idatm_valid;
    InputSeqInfo  _input_seq_info;
    PyObject*  _logger;
    int  _num_hyds = 0;
    AS_PBManager  _pb_mgr;
    mutable bool  _polymers_computed;
    PositionMatrix  _position;
    mutable bool  _recompute_rings;
    Residues  _residues;
    ResNumbering  _res_numbering = RN_AUTHOR;
    std::vector<bool>  _res_numbering_valid = { true, false, false };
    int _ribbon_display_count = 0;
    RibbonOrientation _ribbon_orientation = RIBBON_ORIENT_PEPTIDE;
    RibbonMode  _ribbon_mode_helix = RIBBON_MODE_DEFAULT;
    RibbonMode  _ribbon_mode_strand = RIBBON_MODE_DEFAULT;
    bool  _ribbon_show_spine = false;
    float  _ribbon_tether_opacity = 0.5;
    float  _ribbon_tether_scale = 1.0;
    TetherShape  _ribbon_tether_shape = RIBBON_TETHER_CONE;
    int  _ribbon_tether_sides = 4;
    int _ring_display_count = 0;
    mutable Rings  _rings;
    mutable unsigned int  _rings_last_all_size_threshold;
    mutable bool  _rings_last_cross_residues;
    mutable std::set<const Residue *>*  _rings_last_ignore;
    bool  _ss_assigned;
    mutable bool  _structure_cats_dirty;

    void  add_bond(Bond* b) { _bonds.emplace_back(b); set_gc_shape(); set_gc_adddel(); }
    void  add_atom(Atom* a) { _atoms.emplace_back(a); set_gc_shape(); set_gc_adddel(); }
    void  _calculate_rings(bool cross_residue, unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    virtual void  _compute_atom_types() {}
    void  _compute_idatm_types() { _idatm_valid = true; _compute_atom_types(); }
    virtual void  _compute_structure_cats() const {}
    void _coord_set_insert(CoordSets &coord_sets, CoordSet* cs, int index);
    void  _copy(Structure* s, PositionMatrix coord_adjust = nullptr,
        std::map<ChainID, ChainID>* chain_id_map = nullptr) const;
    void  _delete_atom(Atom* a);
    void  _delete_atoms(const std::set<Atom*>& atoms, bool verify=false);
    void  _delete_residue(Residue* r);
    void  _fast_calculate_rings(std::set<const Residue *>* ignore) const;
    bool  _fast_ring_calc_available(bool cross_residue,
            unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    void  _form_chain_check(Atom* a1, Atom* a2, Bond* b=nullptr);
    void  _get_interres_connectivity(std::map<Residue*, int>& res_lookup,
            std::map<int, Residue*>& index_lookup,
            std::map<Residue*, bool>& res_connects_to_next,
            std::set<Atom*>& left_missing_structure_atoms,
            std::set<Atom*>& right_missing_structure_atoms,
            const std::set<Atom*>* deleted_atoms = nullptr) const;
    Bond*  _new_bond(Atom* a1, Atom* a2, bool bond_only);
    Chain*  _new_chain(const ChainID& chain_id, PolymerType pt = PT_NONE) const {
        auto chain = new Chain(chain_id, const_cast<Structure*>(this), pt);
        _chains->emplace_back(chain);
        return chain;
    }
    void  _per_residue_rings(unsigned int all_size_threshold, std::set<const Residue *>* ignore,
        Rings* rings = nullptr) const;
    void  _per_structure_rings(unsigned int all_size_threshold, std::set<const Residue *>* ignore) const;
    void  remove_chain(Chain* chain) {
        _chains->erase(std::find(_chains->begin(), _chains->end(), chain));
    }
    bool  _rings_cached (bool cross_residues, unsigned int all_size_threshold,
        std::set<const Residue *>* ignore = nullptr) const;
    static int  SESSION_NUM_FLOATS(int version=CURRENT_SESSION_VERSION) {
        return version < 5 ? 1 : (version < 13 ? 3: 15);
    }
    static int  SESSION_NUM_INTS(int version=CURRENT_SESSION_VERSION) {
        return version == 1 ? 9 : (version < 5 ? 10 : (version < 12 ? 16 : (version < 16 ? 17 : 18)));
    }
    static int  SESSION_NUM_MISC(int version=CURRENT_SESSION_VERSION) {
        return version > 7 ? 3 : 4;
    }
    void  _temporary_per_residue_rings(Rings& rings, unsigned int all_size_threshold,
        std::set<const Residue *>* ignore) const;

public:
    Structure(PyObject* logger = nullptr);
    virtual  ~Structure();

    bool  active_coord_set_change_notify() const { return _active_coord_set_change_notify; }
    CoordSet*  active_coord_set() const { return _active_coord_set; };
    bool  alt_loc_change_notify() const { return _alt_loc_change_notify; }
    bool  ss_change_notify() const { return _ss_change_notify; }
    bool  asterisks_translated;
    const Atoms&  atoms() const { return _atoms; }
    float  ball_scale() const { return _ball_scale; }
    std::map<Residue *, char>  best_alt_locs() const;
    void  bonded_groups(std::vector<std::vector<Atom*>>* groups,
        bool consider_missing_structure) const;
    const Bonds&  bonds() const { return _bonds; }
    const Chains&  chains() const { if (_chains == nullptr) make_chains(); return *_chains; }
    void  change_chain_ids(const std::vector<StructureSeq*>, const std::vector<ChainID>,
        bool /*non-polymeric*/=true);
    ChangeTracker*  change_tracker() { return _change_tracker; }
    void  clear_coord_sets();
    void  combine(Structure* s, std::map<ChainID, ChainID>* chain_id_map,
        PositionMatrix coord_adjust=nullptr) const { _copy(s, coord_adjust, chain_id_map); }
    void  combine_sym_atoms();
    virtual void  compute_secondary_structure(float = -0.5, int = 3, int = 3,
        bool = false, CompSSInfo* = nullptr) {}
    const CoordSets&  coord_sets() const { return _coord_sets; }
    virtual Structure*  copy() const;
    void  delete_alt_locs();
    void  delete_atom(Atom* a);
    void  delete_atoms(const std::set<Atom*>& atoms) { _delete_atoms(atoms); }
    void  delete_atoms(const std::vector<Atom*>& atoms);
    void  delete_bond(Bond* b);
    void  delete_residue(Residue* r);
    bool  display() const { return _display; }
    void  extend_input_seq_info(ChainID& chain_id, ResName& res_name) {
        _input_seq_info[chain_id].push_back(res_name);
    }
    CoordSet*  find_coord_set(int) const;
    Residue*  find_residue(const ChainID& chain_id, int pos, char insert) const;
    Residue*  find_residue(const ChainID& chain_id, int pos, char insert,
        ResName& name) const;
    bool  idatm_failed() { ready_idatm_types(); return _idatm_failed; }
    bool  idatm_valid() const { return _idatm_valid; }
    const InputSeqInfo&  input_seq_info() const { return _input_seq_info; }
    std::string  input_seq_source;
    bool  is_traj;
    PyObject*  logger() const { return _logger; }
    bool  lower_case_chains;
    virtual void  make_chains() const;
    std::map<std::string, std::vector<std::string>> metadata;
    Atom*  new_atom(const char* name, const Element& e);
    Bond*  new_bond(Atom* a1, Atom* a2) { return _new_bond(a1, a2, false); }
    CoordSet*  new_coord_set();
    CoordSet*  new_coord_set(int index);
    CoordSet*  new_coord_set(int index, int size);
    Residue*  new_residue(const ResName& name, const ChainID& chain,
        int pos, char insert=' ', Residue *neighbor=NULL, bool after=true);
    std::set<ResName>  nonstd_res_names() const;
    virtual void  normalize_ss_ids() {}
    size_t  num_atoms() const { return atoms().size(); }
    size_t  num_bonds() const { return bonds().size(); }
    size_t  num_hyds() const { return _num_hyds; }
    size_t  num_residues() const { return residues().size(); }
    size_t  num_ribbon_residues() const;
    size_t  num_chains() const { return chains().size(); }
    size_t  num_coord_sets() const { return coord_sets().size(); }
    AS_PBManager&  pb_mgr() { return _pb_mgr; }
    const AS_PBManager&  pb_mgr() const { return _pb_mgr; }
    int  pdb_version;
    enum PolymerMissingStructure {
        PMS_ALWAYS_CONNECTS = 0,
        PMS_NEVER_CONNECTS = 1,
        PMS_TRACE_CONNECTS = 2
    };
    virtual std::vector<std::pair<Chain::Residues,PolymerType>>  polymers(
        PolymerMissingStructure /*missing_structure_treatment*/ = PMS_ALWAYS_CONNECTS,
        bool /*consider_chain_ids*/ = true) const {
            return std::vector<std::pair<Chain::Residues,PolymerType>>();
        }
    const PositionMatrix&  position() const { return _position; }
    void  ready_idatm_types() { if (!_idatm_valid) _compute_idatm_types(); }
    void  renumber_residues(const std::vector<Residue*>& res_list, int start);
    void  reorder_residues(const Residues&); 
    ResNumbering  res_numbering() const { return _res_numbering; }
    bool  res_numbering_valid(ResNumbering rn) const { return _res_numbering_valid[rn]; }
    const Residues&  residues() const { return _residues; }
    const Rings&  rings(bool cross_residues = false,
        unsigned int all_size_threshold = 0,
        std::set<const Residue *>* ignore = nullptr) const;
    int  session_info(PyObject* ints, PyObject* floats, PyObject* misc) const;
    void  session_restore(int version, PyObject* ints, PyObject* floats, PyObject* misc);
    void  session_restore_setup() const { _pb_mgr.session_restore_setup();}
    void  session_restore_teardown() const { _pb_mgr.session_restore_teardown();}
    mutable std::unordered_map<const Atom*, size_t>  *session_save_atoms;
    mutable std::unordered_map<const Bond*, size_t>  *session_save_bonds;
    mutable std::unordered_map<const Chain*, size_t>  *session_save_chains;
    mutable std::unordered_map<const CoordSet*, size_t>  *session_save_crdsets;
    mutable std::unordered_map<const Residue*, size_t>  *session_save_residues;
    void  session_save_setup() const;
    void  session_save_teardown() const;
    void  set_active_coord_set_change_notify(bool cn) { _active_coord_set_change_notify = cn; }
    void  set_active_coord_set(CoordSet *cs);
    void  set_alt_loc_change_notify(bool cn) const { _alt_loc_change_notify = cn; }
    void  set_ss_change_notify(bool cn) const { _ss_change_notify = cn; }
    void  set_ball_scale(float bs) {
        if (bs == _ball_scale) return;
        set_gc_shape(); _ball_scale = bs;
        change_tracker()->add_modified(this, this, ChangeTracker::REASON_BALL_SCALE);
    }
    void  set_color(const Rgba& rgba);
    void  set_display(bool d) {
        if (d == _display) return;
        set_gc_shape(); _display = d;
        change_tracker()->add_modified(this, this, ChangeTracker::REASON_DISPLAY);
    }
    void  set_idatm_valid(bool valid) { _idatm_valid = valid; }
    void  set_input_seq_info(const ChainID& chain_id, const std::vector<ResName>& res_names,
        const std::vector<Residue*>* correspondences = nullptr, PolymerType pt = PT_NONE,
        bool one_letter_names = false);
    void  set_position_matrix(double* pos);
    void  set_res_numbering(ResNumbering rn);
    void  set_res_numbering_valid(ResNumbering rn, bool valid) { _res_numbering_valid[rn] = valid; }
    void  set_ss_assigned(bool sa) { _ss_assigned = sa; }
    bool  ss_assigned() const { return _ss_assigned; }
    bool  ss_ids_normalized;
    void  start_change_tracking(ChangeTracker* ct) {
        _change_tracker = ct;
        ct->add_created(this, this);
        pb_mgr().start_change_tracking(ct);
    }
    void  use_best_alt_locs();
    void  use_default_atom_radii();

    // ribbon stuff
    float  ribbon_tether_scale() const { return _ribbon_tether_scale; }
    TetherShape  ribbon_tether_shape() const { return _ribbon_tether_shape; }
    int  ribbon_tether_sides() const { return _ribbon_tether_sides; }
    float  ribbon_tether_opacity() const { return _ribbon_tether_opacity; }
    bool  ribbon_show_spine() const { return _ribbon_show_spine; }
    int  ribbon_display_count() const { return _ribbon_display_count; }
    RibbonOrientation  ribbon_orientation() const { return _ribbon_orientation; }
    RibbonOrientation  ribbon_orient(const Residue *r) const;
    RibbonMode  ribbon_mode_helix() const { return _ribbon_mode_helix; }
    RibbonMode  ribbon_mode_strand() const { return _ribbon_mode_strand; }
    void  set_ribbon_tether_scale(float s);
    void  set_ribbon_tether_shape(TetherShape ts);
    void  set_ribbon_tether_sides(int s);
    void  set_ribbon_tether_opacity(float o);
    void  set_ribbon_show_spine(bool ss);
    void  set_ribbon_orientation(RibbonOrientation o);
    void  set_ribbon_mode_helix(RibbonMode m);
    void  set_ribbon_mode_strand(RibbonMode m);

    // filled ring stuff
    int  ring_display_count() const { return _ring_display_count; }

    // graphics changes including pseudobond group changes.
    int   get_all_graphics_changes() const;
    void  set_all_graphics_changes(int changes);
};

inline void
Structure::set_ribbon_tether_scale(float s) {
    if (s == _ribbon_tether_scale)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_scale = s;
}

inline void
Structure::set_ribbon_tether_shape(TetherShape ts) {
    if (ts == _ribbon_tether_shape)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_shape = ts;
}

inline void
Structure::set_ribbon_tether_sides(int s) {
    if (s == _ribbon_tether_sides)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_sides = s;
}

inline void
Structure::set_ribbon_tether_opacity(float o) {
    if (o == _ribbon_tether_opacity)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_opacity = o;
}

inline void
Structure::set_ribbon_show_spine(bool ss) {
    if (ss == _ribbon_show_spine)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_DISPLAY);
    set_gc_ribbon();
    _ribbon_show_spine = ss;
}

inline void
Structure::set_ribbon_orientation(RibbonOrientation o) {
    if (o == _ribbon_orientation)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_ORIENTATION);
    set_gc_ribbon();
    _ribbon_orientation = o;
}

inline void
Structure::set_ribbon_mode_helix(RibbonMode m) {
    if (m == _ribbon_mode_helix)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_MODE);
    set_gc_ribbon();
    _ribbon_mode_helix = m;
}

inline void
Structure::set_ribbon_mode_strand(RibbonMode m) {
    if (m == _ribbon_mode_strand)
        return;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_RIBBON_MODE);
    set_gc_ribbon();
    _ribbon_mode_strand = m;
}

} //  namespace atomstruct

#include "CoordSet.h"

namespace atomstruct {

inline void
Structure::clear_coord_sets() {
    for (auto cs: _coord_sets)
        delete cs;
    _coord_sets.clear();
    _active_coord_set = nullptr;
}

} //  namespace atomstruct

#endif  // atomstruct_Structure
