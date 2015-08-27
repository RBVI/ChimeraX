// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Residue
#define atomstruct_Residue

#include <map>
#include <set>
#include <string>
#include <vector>

#include <basegeom/Rgba.h>
#include <basegeom/destruct.h>
#include "imex.h"
#include "string_types.h"

namespace atomstruct {

class Atom;
class AtomicStructure;
class Bond;
class Chain;

using basegeom::Rgba;

class ATOMSTRUCT_IMEX Residue {
public:
    typedef std::vector<Atom *>  Atoms;
    typedef std::multimap<AtomName, Atom *>  AtomsMap;
private:
    friend class AtomicStructure;
    Residue(AtomicStructure *as, const ResName& name, const ChainID& chain, int pos, char insert);
    virtual  ~Residue() {
        auto du = basegeom::DestructionUser(this);
        if (_chain != nullptr)
            _chain->remove_residue(this);
    }

    friend class Chain;
    void  set_chain(Chain* chain) { _chain = chain; }

    char  _alt_loc;
    Atoms  _atoms;
    Chain*  _chain;
    ChainID  _chain_id;
    char  _insertion_code;
    bool  _is_helix;
    bool  _is_het;
    bool  _is_sheet;
    ResName  _name;
    int  _position;
    int  _ss_id;
    bool  _ribbon_display;
    Rgba  _ribbon_rgba;
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
    int  position() const { return _position; }
    void  remove_atom(Atom*);
    void  set_alt_loc(char alt_loc);
    void  set_is_helix(bool ih) { _is_helix = ih; }
    void  set_is_het(bool ih) { _is_het = ih; }
    void  set_is_sheet(bool is) { _is_sheet = is; }
    void  set_ss_id(int ssid) { _ss_id = ssid; }
    int  ss_id() const { return _ss_id; }
    std::string  str() const;
    AtomicStructure*  structure() const { return _structure; }
    std::vector<Atom*>  template_assign(
        void (Atom::*assign_func)(const char*), const char* app,
        const char* template_dir, const char* extension) const;

    // handy
    static const std::set<AtomName> aa_min_backbone_names;
    static const std::set<AtomName> aa_max_backbone_names;
    static const std::set<AtomName> na_min_backbone_names;
    static const std::set<AtomName> na_max_backbone_names;

    // graphics related
    bool  ribbon_display() const { return _ribbon_display; }
    const Rgba&  ribbon_color() const { return _ribbon_rgba; }
    void  set_ribbon_display(bool d)
        { structure()->set_gc_ribbon(); _ribbon_display = d; }
    void  set_ribbon_color(const Rgba& rgba)
        { structure()->set_gc_ribbon(); _ribbon_rgba = rgba; }
};

#include "Chain.h"
inline const ChainID&
Residue::chain_id() const
{
    if (_chain != nullptr)
        return _chain->chain_id();
    return _chain_id;
}

}  // namespace atomstruct

#endif  // atomstruct_Residue
