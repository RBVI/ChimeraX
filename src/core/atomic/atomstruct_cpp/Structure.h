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

#ifndef atomstruct_Structure
#define atomstruct_Structure

#include <algorithm>
#include <element/Element.h>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Chain.h"
#include "ChangeTracker.h"
#include "destruct.h"
#include "PBManager.h"
#include "Rgba.h"
#include "Ring.h"
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

    
class ATOMSTRUCT_IMEX GraphicsContainer {
private:
    int _gc_changes;
    
public:
    enum ChangeType {
      SHAPE_CHANGE = (1 << 0),
      COLOR_CHANGE = (1 << 1),
      SELECT_CHANGE = (1 << 2),
      RIBBON_CHANGE = (1 << 3),
    };

    GraphicsContainer(): _gc_changes(0) {}
    virtual  ~GraphicsContainer() {}
    virtual void  gc_clear() { _gc_changes = 0; }
    virtual bool  get_gc_color() const { return _gc_changes & COLOR_CHANGE; }
    virtual bool  get_gc_select() const { return _gc_changes & SELECT_CHANGE; }
    virtual bool  get_gc_shape() const { return _gc_changes & SHAPE_CHANGE; }
    virtual bool  get_gc_ribbon() const { return _gc_changes & RIBBON_CHANGE; }
    virtual int   get_graphics_changes() const { return _gc_changes; }
    virtual void  set_gc_color() { set_graphics_change(COLOR_CHANGE); }
    virtual void  set_gc_select() { set_graphics_change(SELECT_CHANGE); }
    virtual void  set_gc_shape() { set_graphics_change(SHAPE_CHANGE); }
    virtual void  set_gc_ribbon() { set_graphics_change(RIBBON_CHANGE); }
    virtual void  set_graphics_changes(int change) { _gc_changes = change; }
    virtual void  set_graphics_change(ChangeType type) { _gc_changes |= type; }
    virtual void  clear_graphics_change(ChangeType type) { _gc_changes &= ~type; }
};

// Structure and AtomicStructure have all the methods and typedefs (i.e.
// AtomicStructure simply inherits everything from Structure and doesn't
// add any) so that they can be treated identically in the Python
// layer.  Some atomic-structure-specific methods will have no-op
// implementations in Structure and real implementations in AtomicStructure.
class ATOMSTRUCT_IMEX Structure: public GraphicsContainer {
    friend class Atom; // for IDATM stuff and structure categories
    friend class Bond; // for checking if make_chains() has been run yet, struct categories
    friend class Residue; // for _polymers_computed
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
    enum TetherShape { RIBBON_TETHER_CONE = 0,
                       RIBBON_TETHER_REVERSE_CONE = 1,
                       RIBBON_TETHER_CYLINDER = 2 };
    enum RibbonOrientation { RIBBON_ORIENT_DEFAULT = 0,
                             RIBBON_ORIENT_GUIDES = 1,
                             RIBBON_ORIENT_ATOMS = 2,
                             RIBBON_ORIENT_CURVATURE = 3,
                             RIBBON_ORIENT_PEPTIDE = 4 };
protected:
    const int  CURRENT_SESSION_VERSION = 1;

    CoordSet *  _active_coord_set;
    Atoms  _atoms;
    float  _ball_scale = 0.25;
    Bonds  _bonds;
    mutable Chains*  _chains;
    ChangeTracker*  _change_tracker;
    CoordSets  _coord_sets;
    bool  _display = true;
    bool  _idatm_valid;
    InputSeqInfo  _input_seq_info;
    PyObject*  _logger;
    std::string  _name;
    int  _num_hyds = 0;
    AS_PBManager  _pb_mgr;
    mutable bool  _polymers_computed;
    mutable bool  _recompute_rings;
    Residues  _residues;
    int _ribbon_display_count = 0;
    RibbonOrientation _ribbon_orientation = RIBBON_ORIENT_DEFAULT;
    bool  _ribbon_show_spine = false;
    float  _ribbon_tether_scale = 1.0;
    TetherShape  _ribbon_tether_shape = RIBBON_TETHER_CONE;
    int  _ribbon_tether_sides = 4;
    float  _ribbon_tether_opacity = 0.5;
    mutable Rings  _rings;
    mutable unsigned int  _rings_last_all_size_threshold;
    mutable bool  _rings_last_cross_residues;
    mutable std::set<const Residue *>*  _rings_last_ignore;
    mutable bool  _structure_cats_dirty;

    void  add_bond(Bond* b) { _bonds.emplace_back(b); }
    void  add_atom(Atom* a) { _atoms.emplace_back(a); }
    void  _calculate_rings(bool cross_residue, unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    virtual void  _compute_atom_types() {}
    void  _compute_idatm_types() { _idatm_valid = true; _compute_atom_types(); }
    virtual void  _compute_structure_cats() const {}
    void  _copy(Structure*) const;
    void  _delete_atom(Atom* a);
    void  _delete_atoms(const std::set<Atom*>& atoms);
    void  _delete_residue(Residue* r, const Residues::iterator& ri);
    void  _fast_calculate_rings(std::set<const Residue *>* ignore) const;
    bool  _fast_ring_calc_available(bool cross_residue,
            unsigned int all_size_threshold,
            std::set<const Residue *>* ignore) const;
    Chain*  _new_chain(const ChainID& chain_id) const {
        auto chain = new Chain(chain_id, const_cast<Structure*>(this));
        _chains->emplace_back(chain);
        return chain;
    }
    void  remove_chain(Chain* chain) {
        _chains->erase(std::find(_chains->begin(), _chains->end(), chain));
    }
    bool  _rings_cached (bool cross_residues, unsigned int all_size_threshold,
        std::set<const Residue *>* ignore = nullptr) const;
    // in the SESSION* functions, a version of "0" means the latest version
    static int  SESSION_NUM_FLOATS(int /*version*/=0) { return 1; }
    static int  SESSION_NUM_INTS(int /*version*/=0) { return 9; }
    static int  SESSION_NUM_MISC(int /*version*/=0) { return 4; }

public:
    Structure(PyObject* logger = nullptr);
    virtual  ~Structure();

    CoordSet*  active_coord_set() const { return _active_coord_set; };
    bool  asterisks_translated;
    const Atoms&  atoms() const { return _atoms; }
    float  ball_scale() const { return _ball_scale; }
    std::map<Residue *, char>  best_alt_locs() const;
    void  bonded_groups(std::vector<std::vector<Atom*>>* groups,
        bool consider_missing_structure) const;
    const Bonds&  bonds() const { return _bonds; }
    const Chains&  chains() const { if (_chains == nullptr) make_chains(); return *_chains; }
    ChangeTracker*  change_tracker() { return _change_tracker; }
    const CoordSets&  coord_sets() const { return _coord_sets; }
    virtual Structure*  copy() const;
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
    const InputSeqInfo&  input_seq_info() const { return _input_seq_info; }
    std::string  input_seq_source;
    bool  is_traj;
    PyObject*  logger() const { return _logger; }
    bool  lower_case_chains;
    virtual void  make_chains() const;
    std::map<std::string, std::vector<std::string>> metadata;
    const std::string&  name() const { return _name; }
    Atom*  new_atom(const char* name, const Element& e);
    Bond*  new_bond(Atom *, Atom *);
    CoordSet*  new_coord_set();
    CoordSet*  new_coord_set(int index);
    CoordSet*  new_coord_set(int index, int size);
    Residue*  new_residue(const ResName& name, const ChainID& chain,
        int pos, char insert, Residue *neighbor=NULL, bool after=true);
    size_t  num_atoms() const { return atoms().size(); }
    size_t  num_bonds() const { return bonds().size(); }
    size_t  num_hyds() const { return _num_hyds; }
    size_t  num_residues() const { return residues().size(); }
    size_t  num_chains() const { return chains().size(); }
    size_t  num_coord_sets() const { return coord_sets().size(); }
    AS_PBManager&  pb_mgr() { return _pb_mgr; }
    const AS_PBManager&  pb_mgr() const { return _pb_mgr; }
    int  pdb_version;
    virtual std::vector<Chain::Residues>  polymers(
        bool /*consider_missing_structure*/ = true,
        bool /*consider_chain_ids*/ = true) const { return std::vector<Chain::Residues>(); }
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
    void  set_active_coord_set(CoordSet *cs);
    void  set_ball_scale(float bs) {
        if (bs == _ball_scale) return;
        set_gc_shape(); _ball_scale = bs;
        change_tracker()->add_modified(this, ChangeTracker::REASON_BALL_SCALE);
    }
    void  set_color(const Rgba& rgba);
    void  set_display(bool d) {
        if (d == _display) return;
        set_gc_shape(); _display = d;
        change_tracker()->add_modified(this, ChangeTracker::REASON_DISPLAY);
    }
    void  set_input_seq_info(const ChainID& chain_id, const std::vector<ResName>& res_names) { _input_seq_info[chain_id] = res_names; }
    void  set_name(const std::string& name) { _name = name; }
    void  start_change_tracking(ChangeTracker* ct) { _change_tracker = ct; ct->add_created(this); }
    void  use_best_alt_locs();

    // ribbon stuff
    float  ribbon_tether_scale() const { return _ribbon_tether_scale; }
    TetherShape  ribbon_tether_shape() const { return _ribbon_tether_shape; }
    int  ribbon_tether_sides() const { return _ribbon_tether_sides; }
    float  ribbon_tether_opacity() const { return _ribbon_tether_opacity; }
    bool  ribbon_show_spine() const { return _ribbon_show_spine; }
    int  ribbon_display_count() const { return _ribbon_display_count; }
    RibbonOrientation  ribbon_orientation() const { return _ribbon_orientation; }
    void  set_ribbon_tether_scale(float s);
    void  set_ribbon_tether_shape(TetherShape ts);
    void  set_ribbon_tether_sides(int s);
    void  set_ribbon_tether_opacity(float o);
    void  set_ribbon_show_spine(bool ss);
    void  set_ribbon_orientation(RibbonOrientation o);

    // graphics changes including pseudobond group changes.
    int   get_all_graphics_changes() const;
    void  set_all_graphics_changes(int changes);
};

inline void
Structure::set_ribbon_tether_scale(float s) {
    if (s == _ribbon_tether_scale)
        return;
    change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_scale = s;
}

inline void
Structure::set_ribbon_tether_shape(TetherShape ts) {
    if (ts == _ribbon_tether_shape)
        return;
    change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_shape = ts;
}

inline void
Structure::set_ribbon_tether_sides(int s) {
    if (s == _ribbon_tether_sides)
        return;
    change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_sides = s;
}

inline void
Structure::set_ribbon_tether_opacity(float o) {
    if (o == _ribbon_tether_opacity)
        return;
    change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_TETHER);
    set_gc_ribbon();
    _ribbon_tether_opacity = o;
}

inline void
Structure::set_ribbon_show_spine(bool ss) {
    if (ss == _ribbon_show_spine)
        return;
    change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_STYLE);
    set_gc_ribbon();
    _ribbon_show_spine = ss;
}

inline void
Structure::set_ribbon_orientation(RibbonOrientation o) {
    if (o == _ribbon_orientation)
        return;
    change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_ORIENTATION);
    set_gc_ribbon();
    _ribbon_orientation = o;
}

} //  namespace atomstruct

#endif  // atomstruct_Structure
