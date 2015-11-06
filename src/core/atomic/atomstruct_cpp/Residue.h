// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Residue
#define atomstruct_Residue

#include <map>
#include <set>
#include <string>
#include <vector>

#include "backbone.h"
#include <basegeom/Rgba.h>
#include "imex.h"
#include "string_types.h"

namespace atomstruct {

class Atom;
class AtomicStructure;
class Bond;
class Chain;

using basegeom::ChangeTracker;
using basegeom::Rgba;

class ATOMSTRUCT_IMEX Residue {
public:
    typedef std::vector<Atom *>  Atoms;
    typedef std::multimap<AtomName, Atom *>  AtomsMap;
    enum Style { RIBBON_RIBBON = 0,
                 RIBBON_PIPE = 1 };
    enum PolymerType { PT_NONE, PT_AMINO, PT_NUCLEIC };
private:
    friend class AtomicStructure;
    Residue(AtomicStructure *as, const ResName& name, const ChainID& chain, int pos, char insert);
    virtual  ~Residue();

    friend class Chain;
    void  set_chain(Chain* chain) { _chain = chain; }
    friend class AtomicStructure;
    friend class Bond;
    void  set_polymer_type(PolymerType pt) { _polymer_type = pt; }

    char  _alt_loc;
    Atoms  _atoms;
    Chain*  _chain;
    ChainID  _chain_id;
    char  _insertion_code;
    bool  _is_helix;
    bool  _is_het;
    bool  _is_sheet;
    ResName  _name;
    PolymerType  _polymer_type;
    int  _position;
    float  _ribbon_adjust;
    bool  _ribbon_display;
    bool  _ribbon_hide_backbone;
    Rgba  _ribbon_rgba;
    Style  _ribbon_style;
    int  _ss_id;
    AtomicStructure *  _structure;
public:
    void  add_atom(Atom*);
    const Atoms &  atoms() const { return _atoms; }
    AtomsMap  atoms_map() const;
    std::vector<Bond*>  bonds_between(const Residue* other_res,
        bool just_first=false) const;
    Chain*  chain() const { (void)_structure->chains(); return _chain; }
    const ChainID&  chain_id() const;
    int  count_atom(const AtomName&) const;
    Atom *  find_atom(const AtomName&) const;
    char  insertion_code() const { return _insertion_code; }
    bool  is_helix() const { return _is_helix; }
    bool  is_het() const { return _is_het; }
    bool  is_sheet() const { return _is_sheet; }
    const ResName&  name() const { return _name; }
    PolymerType  polymer_type() const { return _polymer_type; }
    int  position() const { return _position; }
    void  remove_atom(Atom*);
    void  set_alt_loc(char alt_loc);
    void  set_is_helix(bool ih);
    void  set_is_het(bool ih);
    void  set_is_sheet(bool is);
    void  set_ss_id(int ssid);
    int  ss_id() const { return _ss_id; }
    std::string  str() const;
    AtomicStructure*  structure() const { return _structure; }
    std::vector<Atom*>  template_assign(
        void (Atom::*assign_func)(const char*), const char* app,
        const char* template_dir, const char* extension) const;

    // handy
    static const std::set<AtomName>  aa_min_backbone_names;
    static const std::set<AtomName>  aa_max_backbone_names;
    static const std::set<AtomName>  aa_ribbon_backbone_names;
    static const std::set<AtomName>  na_min_backbone_names;
    static const std::set<AtomName>  na_max_backbone_names;
    static const std::set<AtomName>  na_ribbon_backbone_names;
    static const std::set<ResName>  std_solvent_names;
    const std::set<AtomName>*  backbone_atom_names(BackboneExtent bbe) const;

    // graphics related
    float  ribbon_adjust() const;
    const Rgba&  ribbon_color() const { return _ribbon_rgba; }
    bool  ribbon_display() const { return _ribbon_display; }
    bool  ribbon_hide_backbone() const { return _ribbon_hide_backbone; }
    Style  ribbon_style() const { return _ribbon_style; }
    void  set_ribbon_adjust(float a);
    void  set_ribbon_color(const Rgba& rgba);
    void  set_ribbon_display(bool d);
    void  set_ribbon_hide_backbone(bool d);
    void  set_ribbon_style(Style s);
};

}  // namespace atomstruct

#include "AtomicStructure.h"
#include "Chain.h"

namespace atomstruct {

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

inline const ChainID&
Residue::chain_id() const
{
    if (_chain != nullptr)
        return _chain->chain_id();
    return _chain_id;
}

inline float
Residue::ribbon_adjust() const {
    if (_ribbon_adjust >= 0)
        return _ribbon_adjust;
    else if (_is_sheet)
        return 0.7;
    else if (_is_helix)
        return 0.0;
    else
        return 0.0;
}

inline void
Residue::set_is_helix(bool ih) {
    if (ih == _is_helix)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_IS_HELIX);
    _is_helix = ih;
}

inline void
Residue::set_is_het(bool ih) {
    if (ih == _is_het)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_IS_HET);
    _is_het = ih;
}

inline void
Residue::set_is_sheet(bool is) {
    if (is == _is_sheet)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_IS_SHEET);
    _is_sheet = is;
}

inline void
Residue::set_ribbon_adjust(float a) {
    if (a == _ribbon_adjust)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_ADJUST);
    _structure->set_gc_ribbon();
    _ribbon_adjust = a;
}

inline void
Residue::set_ribbon_color(const Rgba& rgba) {
    if (rgba == _ribbon_rgba)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_COLOR);
    _structure->set_gc_ribbon();
    _ribbon_rgba = rgba;
}

inline void
Residue::set_ribbon_display(bool d) {
    if (d == _ribbon_display)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_DISPLAY);
    _structure->set_gc_ribbon();
    _ribbon_display = d;
}

inline void
Residue::set_ribbon_hide_backbone(bool d) {
    if (d == _ribbon_hide_backbone)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_HIDE_BACKBONE);
    _structure->set_gc_ribbon();
    _ribbon_hide_backbone = d;
}

inline void
Residue::set_ribbon_style(Style s) {
    if (s == _ribbon_style)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_RIBBON_STYLE);
    _structure->set_gc_ribbon();
    _ribbon_style = s;
}

inline void
Residue::set_ss_id(int ss_id)
{
    if (ss_id == _ss_id)
        return;
    _structure->change_tracker()->add_modified(this, ChangeTracker::REASON_SS_ID);
    _ss_id = ss_id;
}

}  // namespace atomstruct

#endif  // atomstruct_Residue
