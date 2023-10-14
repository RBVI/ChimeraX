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

#ifndef atomstruct_Residue
#define atomstruct_Residue

#include <map>
#include <pyinstance/PythonInstance.declare.h>
#include <set>
#include <string>
#include <vector>

#include "backbone.h"
#include "ChangeTracker.h"
#include "imex.h"
#include "polymer.h"
#include "Real.h"
#include "res_numbering.h"
#include "Rgba.h"
#include "session.h"
#include "string_types.h"

namespace atomstruct {

class Atom;
class Bond;
class Chain;
class Structure;

class ATOMSTRUCT_IMEX Residue: public pyinstance::PythonInstance<Residue> {
public:
    typedef std::vector<Atom *>  Atoms;
    typedef std::multimap<AtomName, Atom *>  AtomsMap;
    enum SSType { SS_COIL = 0, SS_HELIX = 1, SS_STRAND = 2 };
    // 1adx chain 0 has 7.158 P-P length between residues 24 and 25
    static constexpr Real TRACE_NUCLEIC_DISTSQ_CUTOFF = 51.5;
    // 3ixy chain B has 6.602 CA-CA length between residues 131 and 132
    static constexpr Real TRACE_PROTEIN_DISTSQ_CUTOFF = 45.0;
private:
    friend class Structure;
    Residue(Structure *as, const ResName& name, const ChainID& chain, int pos, char insert);
    virtual  ~Residue();

    friend class StructureSeq;
    void  set_chain(Chain* chain) {
        _chain = chain;
        if (chain == nullptr) set_ribbon_display(false);
    }
    friend class AtomicStructure;
    friend class Bond;

    static int  SESSION_NUM_INTS(int version=CURRENT_SESSION_VERSION) {
        return version < 6 ? 10 : (version < 10 ? 9 : (version < 14 ? 8 : (version < 15 ? 7 : 9)));
    }
    static int  SESSION_NUM_FLOATS(int /*version*/=CURRENT_SESSION_VERSION) { return 1; }

    char  _alt_loc;
    Atoms  _atoms;
    Chain*  _chain;
    ChainID  _chain_id;
    char  _insertion_code;
    ChainID  _mmcif_chain_id;
    ResName  _name;
    int  _number;
    int  _numberings[NUM_RES_NUMBERINGS];
    float  _ribbon_adjust;
    bool  _ribbon_display;
    bool  _ribbon_hide_backbone;
    Rgba  _ribbon_rgba;
    int  _ss_id;
    SSType _ss_type;
    Structure *  _structure;
    bool  _ring_display;
    bool  _rings_are_thin;
    Rgba  _ring_rgba;
public:
    void  add_atom(Atom*, bool copying_structure=false);
    const Atoms&  atoms() const { return _atoms; }
    AtomsMap  atoms_map() const;
    std::vector<Bond*>  bonds_between(const Residue* other_res, bool just_first=false) const;
    Chain*  chain() const;
    const ChainID&  chain_id() const;
    bool  connects_to(const Residue* other_res, bool check_pseudobonds=false) const;
    void  clean_alt_locs();
    int  count_atom(const AtomName&) const;
    void  delete_alt_loc(char al);
    Atom *  find_atom(const AtomName&) const;
    const ChainID&  mmcif_chain_id() const { return _mmcif_chain_id; }
    char  insertion_code() const { return _insertion_code; }
    bool  is_helix() const { return ss_type() == SS_HELIX; }
    bool  is_missing_heavy_template_atoms(bool no_template_okay=false) const;
    bool  is_strand() const { return ss_type() == SS_STRAND; }
    const ResName&  name() const { return _name; }
    void  set_name(const ResName &name) {
        if (name != _name) {
            _name = name;
            change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_NAME);
        }
    }
    PolymerType  polymer_type() const;
    int  number() const { return _number; }
    Atom*  principal_atom() const;
    void  remove_atom(Atom*);
    int  session_num_floats(int version=CURRENT_SESSION_VERSION) const {
        return SESSION_NUM_FLOATS(version) + Rgba::session_num_floats();
    }
    int  session_num_ints(int version=CURRENT_SESSION_VERSION) const {
        return SESSION_NUM_INTS(version) + (version < 15 ? 1 : 2) * Rgba::session_num_ints() + atoms().size();
    }
    void  session_restore(int, int**, float**);
    void  session_save(int**, float**) const;
    void  set_alt_loc(char alt_loc);
    void  set_chain_id(ChainID chain_id);
    void  set_insertion_code(char insertion_code) {
        if (insertion_code != _insertion_code) {
            _insertion_code = insertion_code;
            change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_INSERTION_CODE);
        }
    }
    void  set_is_helix(bool ih);
    void  set_is_strand(bool is);
    void  set_ss_id(int ssid);
    void  set_ss_type(SSType sst);
    void  set_mmcif_chain_id(const ChainID &cid) { _mmcif_chain_id = cid; }
    void  set_number(int number);
    void  set_number(ResNumbering rn, int number) { _numberings[rn] = number; }
    static void  set_templates_dir(const std::string&);
    static void  set_user_templates_dir(const std::string&);
    int  ss_id() const;
    SSType  ss_type() const;
    std::string  str() const;
    Structure*  structure() const { return _structure; }
    std::vector<Atom*>  template_assign(
        void (Atom::*assign_func)(const char*), const char* app,
        const char* template_dir, const char* extension) const;

    // handy
    static const std::set<AtomName>  aa_min_backbone_names;
    static const std::vector<AtomName>  aa_min_ordered_backbone_names;
    static const std::set<AtomName>  aa_max_backbone_names;
    static const std::set<AtomName>  aa_ribbon_backbone_names;
    static const std::set<AtomName>  aa_side_connector_names;
    static const std::set<AtomName>  na_min_backbone_names;
    static const std::vector<AtomName>  na_min_ordered_backbone_names;
    static const std::set<AtomName>  na_max_backbone_names;
    static const std::set<AtomName>  na_ribbon_backbone_names;
    static const std::set<AtomName>  na_side_connector_names;
    static const std::set<AtomName>  ribose_names;
    static std::set<ResName>  std_solvent_names;
    static std::set<ResName>  std_water_names;
    static std::map<ResName, std::map<AtomName, char>>  ideal_chirality; // populated by mmCIF CCDs
    const std::set<AtomName>*  backbone_atom_names(BackboneExtent bbe) const;
    const std::vector<AtomName>*  ordered_min_backbone_atom_names() const;
    const std::set<AtomName>*  ribose_atom_names() const;
    const std::set<AtomName>*  side_connector_atom_names() const;

    // change tracking
    ChangeTracker*  change_tracker() const;

    // graphics related
    float  ribbon_adjust() const;
    const Rgba&  ribbon_color() const { return _ribbon_rgba; }
    bool  ribbon_display() const { return _ribbon_display; }
    bool  ribbon_hide_backbone() const { return _ribbon_hide_backbone; }
    bool  selected() const;  // True if any atom selected
    void  clear_hide_bits(int bit_mask, bool atoms_only = false);  // clear atom and bond hide bits
    void  set_ribbon_adjust(float a);
    void  set_ribbon_color(const Rgba& rgba);
    void  set_ribbon_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b, Rgba::Channel a) {
        set_ribbon_color(Rgba({r, g, b, a}));
    }
    void  set_ribbon_display(bool d);
    void  set_ribbon_hide_backbone(bool d);
    void  ribbon_clear_hide();

    const Rgba&  ring_color() const { return _ring_rgba; }
    bool  ring_display() const { return _ring_display; }
    bool  thin_rings() const { return _rings_are_thin; }
    void  set_ring_color(const Rgba& rgba);
    void  set_ring_color(Rgba::Channel r, Rgba::Channel g, Rgba::Channel b, Rgba::Channel a) {
        set_ring_color(Rgba({r, g, b, a}));
    }
    void  set_ring_display(bool d);
    void  set_thin_rings(bool d);
};

}  // namespace atomstruct

#include "Atom.h"
#include "Chain.h"
#include "Structure.h"

namespace atomstruct {

inline ChangeTracker*
Residue::change_tracker() const { return structure()->change_tracker(); }

inline void
Residue::clean_alt_locs() { for (auto a: atoms()) a->clean_alt_locs(); }

inline const std::set<AtomName>*
Residue::backbone_atom_names(BackboneExtent bbe) const
{
    if (!structure()->_polymers_computed) structure()->polymers();
    if (polymer_type() == PT_AMINO) {
        if (bbe == BBE_RIBBON) return &aa_ribbon_backbone_names;
        if (bbe == BBE_MAX) return &aa_max_backbone_names;
        return &aa_min_backbone_names;
    }
    if (polymer_type() == PT_NUCLEIC) {
        if (bbe == BBE_RIBBON) return &na_ribbon_backbone_names;
        if (bbe == BBE_MAX) return &na_max_backbone_names;
        return &na_min_backbone_names;
    }
    return nullptr;
}

inline const std::vector<AtomName>*
Residue::ordered_min_backbone_atom_names() const
{
    if (!structure()->_polymers_computed) structure()->polymers();
    if (polymer_type() == PT_AMINO)
        return &aa_min_ordered_backbone_names;
    if (polymer_type() == PT_NUCLEIC)
        return &na_min_ordered_backbone_names;
    return nullptr;
}

inline const std::set<AtomName>*
Residue::side_connector_atom_names() const
{
    if (!structure()->_polymers_computed) structure()->polymers();
    if (polymer_type() == PT_AMINO) {
        return &aa_side_connector_names;
    }
    if (polymer_type() == PT_NUCLEIC) {
        return &na_side_connector_names;
    }
    return nullptr;
}

inline const ChainID&
Residue::chain_id() const
{
    if (_chain != nullptr)
        return _chain->chain_id();
    return _chain_id;
}

inline Chain*
Residue::chain() const {
    (void)_structure->chains();
    return _chain;
}
inline PolymerType
Residue::polymer_type() const {
    return chain() == nullptr ? PT_NONE : chain()->polymer_type();
}

inline float
Residue::ribbon_adjust() const {
    if (_ribbon_adjust >= 0)
        return _ribbon_adjust;
    else if (_ss_type == SS_STRAND)
        return 1.0;
    else if (_ss_type == SS_HELIX)
        return 0.0;
    else
        return 0.0;
}

inline const std::set<AtomName>*
Residue::ribose_atom_names() const
{
    if (!structure()->_polymers_computed) structure()->polymers();
    if (polymer_type() == PT_NUCLEIC)
        return &ribose_names;
    return nullptr;
}

inline void
Residue::set_is_helix(bool ih) {
    // old implementation had two booleans for is_helix and is_strand;
    // now sets the ss_type instead
    if (ih)
        set_ss_type(SS_HELIX);
    else
        set_ss_type(SS_COIL);
}

inline void
Residue::set_is_strand(bool is) {
    // old implementation had two booleans for is_helix and is_strand;
    // now sets the ss_type instead
    if (is)
        set_ss_type(SS_STRAND);
    else
        set_ss_type(SS_COIL);
}

inline void
Residue::set_ribbon_adjust(float a) {
    if (a == _ribbon_adjust)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RIBBON_ADJUST);
    _structure->set_gc_ribbon();
    _ribbon_adjust = a;
}

inline void
Residue::set_ribbon_color(const Rgba& rgba) {
    if (rgba == _ribbon_rgba)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RIBBON_COLOR);
    _structure->set_gc_color();
    _ribbon_rgba = rgba;
}

inline void
Residue::set_ribbon_display(bool d) {
    if (d == _ribbon_display)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RIBBON_DISPLAY);
    _structure->set_gc_ribbon();
    _ribbon_display = d;
    if (d)
        _structure->_ribbon_display_count += 1;
    else {
        _structure->_ribbon_display_count -= 1;
        ribbon_clear_hide();
    }
}

inline void
Residue::set_ribbon_hide_backbone(bool d) {
    if (d == _ribbon_hide_backbone)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RIBBON_HIDE_BACKBONE);
    _structure->set_gc_ribbon();
    _ribbon_hide_backbone = d;
}

inline void
Residue::set_ring_color(const Rgba& rgba) {
    if (rgba == _ring_rgba)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RING_COLOR);
    _structure->set_gc_ring();
    _ring_rgba = rgba;
}

inline void
Residue::set_ring_display(bool d) {
    if (d == _ring_display)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RING_DISPLAY);
    _structure->set_gc_ring();
    _ring_display = d;
    if (d)
        _structure->_ring_display_count += 1;
    else {
        _structure->_ring_display_count -= 1;
    }
}

inline void
Residue::set_thin_rings(bool thin)
{
    if (thin == _rings_are_thin)
        return;
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_RING_MODE);
    _structure->set_gc_ring();
    _rings_are_thin = thin;
}

inline int
Residue::ss_id() const {
    if (!structure()->ss_assigned())
        structure()->compute_secondary_structure();
    if (!structure()->ss_ids_normalized)
        structure()->normalize_ss_ids();
    return _ss_id;
}

inline Residue::SSType
Residue::ss_type() const {
    if (!structure()->ss_assigned())
        structure()->compute_secondary_structure();
    return _ss_type;
}

inline void
Residue::set_ss_id(int ss_id)
{
    if (ss_id == _ss_id)
        return;
    if (_structure->ss_change_notify()) {
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_SS_ID);
        _structure->set_gc_ribbon();
    }
    _ss_id = ss_id;
}

inline void
Residue::set_ss_type(SSType sst)
{
    if (sst == _ss_type)
        return;
    if (_structure->ss_change_notify()) {
        _structure->set_ss_assigned(true);
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_SS_TYPE);
        _structure->set_gc_ribbon();
    }
    _ss_type = sst;
}

}  // namespace atomstruct

#include "Atom.h"
#include "Bond.h"

namespace atomstruct {


inline void
Residue::ribbon_clear_hide() {
    clear_hide_bits(Atom::HIDE_RIBBON);
}

inline void
Residue::clear_hide_bits(int mask, bool atoms_only) {
    for (auto atom: atoms()) {
        atom->clear_hide_bits(mask);
        if (atoms_only)
            continue;
        for (auto bond: atom->bonds())
            bond->clear_hide_bits(mask);
    }
}

inline bool
Residue::selected() const {
    for (auto atom: atoms())
        if (atom->selected())
            return true;
    return false;
}

}  // namespace atomstruct

#endif  // atomstruct_Residue
