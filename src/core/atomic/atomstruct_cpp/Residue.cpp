// vi: set expandtab ts=4 sw=4:

#include <algorithm>
#include <set>
#include <sstream>
#include <utility>  // for pair

#include "Atom.h"
#include "Bond.h"
#include "Residue.h"
#include <basegeom/destruct.h>
#include "tmpl/TemplateCache.h"

namespace atomstruct {

const std::set<AtomName> Residue::aa_min_backbone_names = {
    "C", "CA", "N"};
const std::set<AtomName> Residue::aa_max_backbone_names = {
    "C", "CA", "N", "O", "OXT", "OT1", "OT2"};
const std::set<AtomName> Residue::aa_ribbon_backbone_names = {
    "C", "CA", "N", "O", "OXT", "OT1", "OT2"};
const std::set<AtomName> Residue::na_min_backbone_names = {
    "O3'", "C3'", "C4'", "C5'", "O5'", "P"};
const std::set<AtomName> Residue::na_max_backbone_names = {
    "O3'", "C3'", "C4'", "C5'", "O5'", "P", "OP1", "O1P", "OP2", "O2P", "O2'",
    "C2'", "O4'", "C1'", "OP3", "O3P"};
const std::set<AtomName> Residue::na_ribbon_backbone_names = {
    "O3'", "C5'", "O5'", "P", "OP1", "O1P", "OP2", "O2P", "OP3", "O3P"};
const std::set<ResName> Residue::std_solvent_names = { "HOH", "WAT", "DOD" };

Residue::Residue(AtomicStructure *as, const ResName& name,
    const ChainID& chain, int pos, char insert):
    _alt_loc(' '), _chain(nullptr), _chain_id(chain), _insertion_code(insert),
    _is_helix(false), _is_het(false), _is_sheet(false), _name(name), _polymer_type(PT_NONE),
    _position(pos), _ribbon_adjust(-1.0), _ribbon_display(false),
    _ribbon_hide_backbone(true), _ribbon_rgba({160,160,0,255}),
    _ribbon_style(RIBBON_RIBBON), _ss_id(-1), _structure(as)
{
    _structure->change_tracker()->add_created(this);
}

Residue::~Residue() {
    auto du = basegeom::DestructionUser(this);
    if (_chain != nullptr) {
        _chain->remove_residue(this);
        _structure->set_gc_ribbon();
    }
    _structure->change_tracker()->add_deleted(this);
}

void
Residue::add_atom(Atom* a)
{
    a->_residue = this;
    _atoms.push_back(a);
}

Residue::AtomsMap
Residue::atoms_map() const
{
    AtomsMap map;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        map.insert(AtomsMap::value_type(a->name(), a));
    }
    return map;
}

std::vector<Bond*>
Residue::bonds_between(const Residue* other_res, bool just_first) const
{
    std::vector<Bond*> tweeners;
    for (auto a: _atoms) {
        for (auto b: a->bonds()) {
            if (b->other_atom(a)->residue() == other_res) {
                tweeners.push_back(b);
                if (just_first)
                    return tweeners;
            }
        }
    }
    return tweeners;
}

int
Residue::count_atom(const AtomName& name) const
{
    int count = 0;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->name() == name)
            ++count;
    }
    return count;
}

Atom *
Residue::find_atom(const AtomName& name) const
{
    
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->name() == name)
            return a;
    }
    return nullptr;
}

Atom*
Residue::principal_atom() const
{
    // Return the 'chain trace' atom of the residue, if any
    //
    // Normally returns th C4' from a nucleic acid since that is always
    // present, but in the case of a P-only trace it returns the P
    auto am = atoms_map();
    auto caf = am.find("CA");
    if (caf != am.end()) {
        auto ca = caf->second;
        if (ca->element() != Element::C)
            return nullptr;
        if (am.find("N") != am.end() && am.find("C") != am.end())
            return ca;
        return am.size() == 1 ? ca : nullptr;
    }
    auto c4f = am.find("C4'");
    if (c4f == am.end())
        return nullptr;
    auto c4 = c4f->second;
    if (am.find("C3'") != am.end() && am.find("C5'") != am.end() && am.find("O5'") != am.end())
        return c4;
    if (am.size() > 1)
        return nullptr;
    auto pf = am.find("P");
    if (pf == am.end())
        return nullptr;
    auto p = pf->second;
    return p->element() == Element::P ? p : nullptr;
}

void
Residue::remove_atom(Atom* a)
{
    a->_residue = nullptr;
    _atoms.erase(std::find(_atoms.begin(), _atoms.end(), a));
}

void
Residue::session_restore(int** ints, float** floats)
{
    _ribbon_rgba.session_restore(ints, floats);

    auto& int_ptr = *ints;
    auto& float_ptr = *floats;

    _alt_loc = int_ptr[0];
    _is_helix = int_ptr[1];
    _is_het = int_ptr[2];
    _is_sheet = int_ptr[3];
    _polymer_type = (PolymerType)int_ptr[4];
    _ribbon_display = int_ptr[5];
    _ribbon_hide_backbone = int_ptr[6];
    _ribbon_style = (Style)int_ptr[7];
    _ss_id = int_ptr[8];
    auto num_atoms = int_ptr[9];
    int_ptr += SESSION_NUM_INTS;

    _ribbon_adjust = float_ptr[0];
    float_ptr += SESSION_NUM_FLOATS;

    auto& atoms = structure()->atoms();
    for (decltype(num_atoms) i = 0; i < num_atoms; ++i) {
        add_atom(atoms[*int_ptr++]);
    }
}

void
Residue::session_save(int** ints, float** floats) const
{
    _ribbon_rgba.session_save(ints, floats);

    auto& int_ptr = *ints;
    auto& float_ptr = *floats;

    int_ptr[0] = (int)_alt_loc;
    int_ptr[1] = (int)_is_helix;
    int_ptr[2] = (int)_is_het;
    int_ptr[3] = (int)_is_sheet;
    int_ptr[4] = (int)_polymer_type;
    int_ptr[5] = (int)_ribbon_display;
    int_ptr[6] = (int)_ribbon_hide_backbone;
    int_ptr[7] = (int)_ribbon_style;
    int_ptr[8] = (int)_ss_id;
    int_ptr[9] = atoms().size();
    int_ptr += SESSION_NUM_INTS;

    float_ptr[0] = _ribbon_adjust;
    float_ptr += SESSION_NUM_FLOATS;

    auto& atom_map = *structure()->session_save_atoms;
    for (auto a: atoms()) {
        *int_ptr++ = atom_map[a];
    }

}

void
Residue::set_alt_loc(char alt_loc)
{
    if (alt_loc == _alt_loc || alt_loc == ' ') return;
    std::set<Residue *> nb_res;
    bool have_alt_loc = false;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->has_alt_loc(alt_loc)) {
            a->set_alt_loc(alt_loc, false, true);
            have_alt_loc = true;
            for (auto nb: a->neighbors()) {
                if (nb->residue() != this && nb->alt_locs() == a->alt_locs())
                    nb_res.insert(nb->residue());
            }
        }
    }
    if (!have_alt_loc) {
        std::stringstream msg;
        msg << "set_alt_loc(): residue " << str()
            << " does not have an alt loc '" << alt_loc << "'";
        throw std::invalid_argument(msg.str().c_str());
    }
    _alt_loc = alt_loc;
    for (auto nri = nb_res.begin(); nri != nb_res.end(); ++nri) {
        (*nri)->set_alt_loc(alt_loc);
    }
}

std::string
Residue::str() const
{
    std::stringstream pos_string;
    std::string ret = (const char*)_name;
    ret += " ";
    pos_string << _position;
    ret += pos_string.str();
    if (_insertion_code != ' ')
        ret += _insertion_code;
    if (_chain_id != " ") {
        ret += ".";
        ret += _chain_id;
    }
    return ret;
}

std::vector<Atom*>
Residue::template_assign(void (Atom::*assign_func)(const char*),
    const char* app, const char* template_dir, const char* extension) const
{
    // Returns atoms that received assignments.  Can throw these exceptions:
    //   TA_TemplateSyntax:  template syntax error
    //   TA_NoTemplate:  no template found
    //   std::logic_error:  internal logic error
    using tmpl::TemplateCache;
    TemplateCache* tc = TemplateCache::template_cache();
    TemplateCache::AtomMap* am = tc->res_template(name(),
            app, template_dir, extension);

    std::vector<Atom*> assigned;
    for (auto a: _atoms) {
        auto ami = am->find(a->name());
        if (ami == am->end())
            continue;

        auto& norm_type = ami->second.first;
        auto ct = ami->second.second;
        if (ct != nullptr) {
            // assign conditional type if applicable
            bool cond_assigned = false;
            for (auto& ci: ct->conditions) {
                if (ci.op == ".") {
                    // is the given atom terminal?
                    bool is_terminal = true;
                    auto opa = find_atom(ci.operand.c_str());
                    if (opa == nullptr)
                        continue;
                    for (auto bonded: opa->neighbors()) {
                        if (bonded->residue() != this) {
                            is_terminal = false;
                            break;
                        }
                    }
                    if (is_terminal) {
                        cond_assigned = true;
                        if (ci.result != "-") {
                            (a->*assign_func)(ci.result);
                            assigned.push_back(a);
                        }
                    }
                } else if (ci.op == "?") {
                    // does the given atom exist in the residue?
                    if (find_atom(ci.operand.c_str()) != nullptr) {
                        cond_assigned = true;
                        if (ci.result != "-") {
                            (a->*assign_func)(ci.result);
                            assigned.push_back(a);
                        }
                    }
                } else {
                    throw std::logic_error("Legal template condition"
                        " not implemented");
                }
                if (cond_assigned)
                    break;
            }
            if (cond_assigned)
                continue;
        }

        // assign normal type
        if (norm_type != "-") {
            (a->*assign_func)(norm_type);
            assigned.push_back(a);
        }
    }
    return assigned;
}

}  // namespace atomstruct
