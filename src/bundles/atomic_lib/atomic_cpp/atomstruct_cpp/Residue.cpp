// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <algorithm>
#include <cctype>  // for islower
#include <set>
#include <sstream>
#include <utility>  // for pair

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "Bond.h"
#include "destruct.h"
#include "PBGroup.h"
#include "Residue.h"
#include "tmpl/residues.h"
#include "tmpl/TAexcept.h"
#include "tmpl/TemplateCache.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::Residue>;

namespace atomstruct {

const std::set<AtomName> Residue::aa_min_backbone_names = {
    "C", "CA", "N"};
const std::vector<AtomName> Residue::aa_min_ordered_backbone_names = {
    "N", "CA", "C"};
const std::set<AtomName> Residue::aa_max_backbone_names = {
    "C", "CA", "N", "O", "OXT", "OT1", "OT2"};
const std::set<AtomName> Residue::aa_ribbon_backbone_names = {
    "C", "CA", "N", "O", "OXT", "OT1", "OT2"};
const std::set<AtomName> Residue::aa_side_connector_names = {
    "CA"};
const std::set<AtomName> Residue::na_min_backbone_names = {
    "O3'", "C3'", "C4'", "C5'", "O5'", "P"};
const std::vector<AtomName> Residue::na_min_ordered_backbone_names = {
    "P", "O5'", "C5'", "C4'", "C3'", "O3'"};
const std::set<AtomName> Residue::na_max_backbone_names = {
    "O3'", "C3'", "C4'", "C5'", "O5'", "P", "OP1", "O1P", "OP2", "O2P", "O2'",
    "C2'", "O4'", "C1'", "OP3", "O3P"};
const std::set<AtomName> Residue::na_ribbon_backbone_names = {
    "O3'", "C3'", "C4'", "C5'", "O5'", "P", "OP1", "O1P", "OP2", "O2P", "OP3", "O3P"};
const std::set<AtomName> Residue::ribose_names = {
    "O3'", "C3'", "C4'", "C5'", "O5'", "O2'", "C2'", "O4'", "C1'"};
const std::set<AtomName> Residue::na_side_connector_names = {
    "C3'", "C4'", "O2'", "C2'", "O4'", "C1'"};
std::set<ResName> Residue::std_water_names = { "HOH", "WAT", "DOD", "H2O", "D2O", "TIP3" };
std::set<ResName> Residue::std_solvent_names = std_water_names;
std::map<ResName, std::map<AtomName, char>>  Residue::ideal_chirality;

Residue::Residue(Structure *as, const ResName& name, const ChainID& chain, int num, char insert):
    _alt_loc(' '), _chain(nullptr), _chain_id(chain), _insertion_code(insert), _mmcif_chain_id(chain),
    _name(name), _number(num), _ribbon_adjust(-1.0), _ribbon_display(false), _ribbon_hide_backbone(true),
    _ribbon_rgba({160,160,0,255}), _ss_id(-1), _ss_type(SS_COIL), _structure(as), _ring_display(false),
    _rings_are_thin(false), _worm_radius(1.0)
{
    if (!as->lower_case_chains) {
        for (auto c: _chain_id) {
            if (std::islower(c)) {
                as->lower_case_chains = true;
                break;
            }
        }
    }
    for (int i = 0; i < NUM_RES_NUMBERINGS; ++i)
        _numberings[i] = num;
    change_tracker()->add_created(_structure, this);
}

Residue::~Residue() {
    auto du = DestructionUser(this);
    if (_ribbon_display)
        _structure->_ribbon_display_count -= 1;
    _structure->set_gc_ribbon();
    change_tracker()->add_deleted(_structure, this);
}

void
Residue::add_atom(Atom* a, bool copying_or_restoring)
{
    a->_residue = this;
    _atoms.push_back(a);
    if (copying_or_restoring)
        return;

    // If adding first atom or backbone atoms, possibly adjust missing-structure pseudobonds.
    // Try to do this work only if we are not in initial structure creation.
    if (!structure()->_polymers_computed)
        return;
    auto pbg = structure()->pb_mgr().get_group(Structure::PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg == nullptr)
        return;
    // an additional test to avoid doing the expensive computation below if possible, which can
    // severely impact addh for instance [#15840]
    if (a->element().valence() < 2)
        return;
    // Okay, make residue-index map so that we can see if missing-structure bonds are relevant to our residue
    std::map<Residue*, Structure::Residues::size_type> res_map;
    Structure::Residues::size_type i = 0;
    for (auto r: structure()->residues()) {
        res_map[r] = i++;
    }
    auto my_index = res_map[this];
    if (_atoms.size() == 1) {
        // If residue is between residues in the same chain, look for missing-structure pseudobonds
        // that "cross over" this residue and break them into two pseudobonds.
        if (structure()->residues().back() == this || structure()->residues()[0] == this)
            return;
        for (auto pb: pbg->pseudobonds()) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            auto i1 = res_map[a1->residue()];
            auto i2 = res_map[a2->residue()];
            if ((i1 < my_index && i2 > my_index) || (i2 < my_index && i1 > my_index)) {
                pbg->delete_pseudobond(pb);
                // add the "shorter" one first, so that the residue gets placed on the correct side of a gap
                int d1, d2;
                if (i1 < my_index) {
                    d1 = number() - a1->residue()->number();
                    d2 = a2->residue()->number() - number();
                } else {
                    d1 = a1->residue()->number() - number();
                    d2 = number() - a1->residue()->number();
                }
                if (d1 <= d2) {
                    pbg->new_pseudobond(a1, a);
                    pbg->new_pseudobond(a, a2);
                } else {
                    pbg->new_pseudobond(a, a2);
                    pbg->new_pseudobond(a1, a);
                }
                break;
            }
        }
    } else {
        // If this is a new backbone atom, look for missing-structure pseudobonds that go to this
        // residue and adjust their endpoint if appropriate.
        if (chain() == nullptr)
            return;
        auto chain_type = chain()->polymer_type();
        auto bb_name_set = chain_type == PT_AMINO ? aa_min_backbone_names : na_min_backbone_names;
        if (bb_name_set.find(a->name()) == bb_name_set.end())
            return;
        // hack to short-circuit _form_chain_check's expensive computations;
        _structure->_copying_or_restoring = true;
        auto bb_names = chain_type == PT_AMINO ?
            aa_min_ordered_backbone_names : na_min_ordered_backbone_names;
        auto name_index = std::find(bb_names.begin(), bb_names.end(), a->name()) - bb_names.begin();
        // make a separate list of pseudobonds to this residue, so that we are not modifying the
        // main pseudobond list as we iterate over it
        std::vector<Pseudobond*> my_pbs;
        for (auto pb: pbg->pseudobonds()) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            auto i1 = res_map[a1->residue()];
            auto i2 = res_map[a2->residue()];
            if (i1 == my_index || i2 == my_index)
                my_pbs.push_back(pb);
        }
        for (auto pb: my_pbs) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            auto i1 = res_map[a1->residue()];
            auto i2 = res_map[a2->residue()];
            decltype(a1) change_atom = nullptr;
            if (i1 == my_index && i2 == my_index) {
                // internal to the residue
                bool a1_bb = bb_name_set.find(a1->name()) != bb_name_set.end();
                bool a2_bb = bb_name_set.find(a2->name()) != bb_name_set.end();
                if (a1_bb) {
                    auto a1_index = std::find(bb_names.begin(), bb_names.end(), a1->name())
                        - bb_names.begin();
                    if (a2_bb) {
                        auto a2_index = std::find(bb_names.begin(), bb_names.end(), a2->name())
                            - bb_names.begin();
                        auto low_index = a1_index < a2_index ? a1_index : a2_index;
                        auto high_index = a1_index > a2_index ? a1_index : a2_index;
                        if (name_index > low_index && name_index < high_index) {
                            // in between; break into two pseudobonds
                            pbg->delete_pseudobond(pb);
                            pbg->new_pseudobond(a1, a);
                            pbg->new_pseudobond(a2, a);
                        }
                    } else {
                        change_atom = a2;
                    }
                } else {
                    if (a2_bb) {
                        change_atom = a1;
                    } else {
                        // both ends are non-backbone; just delete
                        pbg->delete_pseudobond(pb);
                    }
                }
            } else {
                decltype(a1) here_a;
                decltype(i1) there_i;
                if (i1 == my_index) {
                    here_a = a1;
                    there_i = i2;
                } else {
                    here_a = a2;
                    there_i = i1;
                }
                if (bb_name_set.find(here_a->name()) == bb_name_set.end()) {
                    change_atom = here_a;
                } else {
                    auto here_name_index = std::find(bb_names.begin(), bb_names.end(), here_a->name())
                        - bb_names.begin();
                    // if current pseudobond partner in this residue is to the left of the new atom
                    // (here_name_index < name_index) then change the endpoint if this residue precedes
                    // the other residue (my_index < there_i), and vice versa
                    if ((here_name_index > name_index) == (my_index > there_i)) {
                        change_atom = here_a;
                    }
                }
            }
            if (change_atom != nullptr) {
                pbg->delete_pseudobond(pb);
                decltype(a1) keep_atom = change_atom == a1 ? a2 : a1;
                pbg->new_pseudobond(keep_atom, a);
            }
        }
        _structure->_copying_or_restoring = false;
    }
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

bool
Residue::connects_to(const Residue* other_res, bool check_pseudobonds) const
{
        if (!bonds_between(other_res, true).empty())
            return true;
        if (!check_pseudobonds)
            return false;
        auto pbg = structure()->pb_mgr().get_group(Structure::PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
        if (pbg == nullptr)
            return false;
        for (auto pb: pbg->pseudobonds()) {
            int count = 0;
            for (auto a: pb->atoms()) {
                if (a->residue() == this) {
                    count += 1;
                    continue;
                }
                if (a->residue() == other_res)
                    count += 2;
                else
                    break;
            }
            if (count == 3)
                return true;
        }
        return false;
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

void
Residue::delete_alt_loc(char alt_loc)
{
    if (alt_loc == ' ')
        throw std::invalid_argument("Residue.delete_alt_loc(): cannot delete the ' ' alt loc");
    std::set<Residue *> nb_res;
    bool have_alt_loc = false;
    bool some_multiple = false;
    for (Atoms::const_iterator ai=_atoms.begin(); ai != _atoms.end(); ++ai) {
        Atom *a = *ai;
        if (a->has_alt_loc(alt_loc)) {
            have_alt_loc = true;
            for (auto nb: a->neighbors()) {
                if (nb->residue() != this && nb->alt_locs() == a->alt_locs())
                    nb_res.insert(nb->residue());
            }
            a->delete_alt_loc(alt_loc);
            if (!a->_alt_loc_map.empty())
                some_multiple = true;
        }
    }
    if (!have_alt_loc) {
        std::stringstream msg;
        msg << "delete_alt_loc(): residue " << str()
            << " does not have an alt loc '" << alt_loc << "'";
        throw std::invalid_argument(msg.str().c_str());
    }
    for (auto nri = nb_res.begin(); nri != nb_res.end(); ++nri) {
        (*nri)->delete_alt_loc(alt_loc);
    }
    if (alt_loc == _alt_loc) {
        if (some_multiple) {
            auto best_alt_locs = structure()->best_alt_locs();
            set_alt_loc(best_alt_locs[this]);
        } else
            _alt_loc = ' ';
    }
    change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_ALT_LOCS);
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

bool
Residue::is_missing_heavy_template_atoms(bool no_template_okay) const
{
    bool chain_start = chain() != nullptr && chain()->residues()[0] == this;
    bool chain_end = chain() != nullptr && chain()->residues()[chain()->residues().size()-1] == this;
    auto tmpl_res = tmpl::find_template_residue(name(), chain_start, chain_end);
    if (tmpl_res == nullptr) {
        if (no_template_okay)
            return false;
        std::ostringstream os;
        os << "No residue template found for " << name();
        throw tmpl::TA_NoTemplate(os.str());
    }
    // pretty unsophisticated check upcoming; check both have the same number of heavy atoms
    // and then check they have the same elements.  No name or connectivity checking
    std::map<int, int> res_heavys;
    std::map<int, int> tmpl_heavys;
    for (auto a: atoms()) {
        auto atomic_num = a->element().number();
        if (atomic_num > 1)
            res_heavys[atomic_num] = res_heavys[atomic_num] + 1;
    }
    for (auto ta: tmpl_res->atoms()) {
        auto atomic_num = ta->element().number();
        if (atomic_num > 1)
            tmpl_heavys[atomic_num] = tmpl_heavys[atomic_num] + 1;
    }
    if (res_heavys.size() != tmpl_heavys.size())
        return true;
    for (auto an_num: res_heavys)
        if (tmpl_heavys[an_num.first] != an_num.second)
            return true;
    return false;
}

Atom*
Residue::principal_atom() const
{
    // Return the 'chain trace' atom of the residue, if any
    //
    // Normally returns th C4' from a nucleic acid since that is always
    // present, but in the case of a P-only trace it returns the P
    auto am = atoms_map();
    if (structure()->chains_made()) {
        auto pt = polymer_type();
        if (pt == PolymerType::PT_AMINO) {
            auto caf = am.find("CA");
            if (caf != am.end())
                return caf->second;
        } else if (pt == PolymerType::PT_NUCLEIC) {
            auto c4f = am.find("C4'");
            if (c4f == am.end()) {
                if (am.size() == 1) {
                    auto pf = am.find("P");
                    if (pf != am.end())
                        return pf->second;
                }
            } else
                return c4f->second;
        }
    } else {
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
        if (c4f == am.end()) {
            if (am.size() > 1)
                return nullptr;
            auto pf = am.find("P");
            if (pf == am.end())
                return nullptr;
            auto p = pf->second;
            return p->element() == Element::P ? p : nullptr;
        }
        auto c4 = c4f->second;
        if (am.find("C3'") != am.end() && am.find("C5'") != am.end() && am.find("O5'") != am.end())
            return c4;
    }
    return nullptr;
}

void
Residue::remove_atom(Atom* a)
{
    a->_residue = nullptr;
    _atoms.erase(std::find(_atoms.begin(), _atoms.end(), a));
}

void
Residue::session_restore(int version, int** ints, float** floats)
{
    _ribbon_rgba.session_restore(ints, floats);
    if (version > 14)
        _ring_rgba.session_restore(ints, floats);

    auto& int_ptr = *ints;
    auto& float_ptr = *floats;

    int num_atoms;
    if (version < 6) {
        _alt_loc = int_ptr[0];
        if (int_ptr[1]) // is_helix
            _ss_type = SS_HELIX;
        // was is_het
        if (int_ptr[3]) // is_strand
            _ss_type = SS_STRAND;
        _ribbon_display = int_ptr[5];
        _ribbon_hide_backbone = int_ptr[6];
        // int_ptr[7] unused ribbon selected
        _ss_id = int_ptr[8];
        num_atoms = int_ptr[9];
    } else if (version < 10) {
        _alt_loc = int_ptr[0];
        // was is_het
        _ribbon_display = int_ptr[3];
        _ribbon_hide_backbone = int_ptr[4];
        // int_ptr[5] unused ribbon selected
        _ss_id = int_ptr[6];
        _ss_type = (SSType)int_ptr[7];
        num_atoms = int_ptr[8];
    } else if (version < 14) {
        _alt_loc = int_ptr[0];
        // was is_het
        _ribbon_display = int_ptr[2];
        _ribbon_hide_backbone = int_ptr[3];
        // int_ptr[4] unused ribbon selected
        _ss_id = int_ptr[5];
        _ss_type = (SSType)int_ptr[6];
        num_atoms = int_ptr[7];
    } else if (version < 15) {
        _alt_loc = int_ptr[0];
        _ribbon_display = int_ptr[1];
        _ribbon_hide_backbone = int_ptr[2];
        // int_ptr[3] unused ribbon selected
        _ss_id = int_ptr[4];
        _ss_type = (SSType)int_ptr[5];
        num_atoms = int_ptr[6];
    } else {
        _alt_loc = int_ptr[0];
        _ribbon_display = int_ptr[1];
        _ribbon_hide_backbone = int_ptr[2];
        // int_ptr[3] unused ribbon selected
        _ss_id = int_ptr[4];
        _ss_type = (SSType)int_ptr[5];
        _ring_display = int_ptr[6];
        _rings_are_thin = int_ptr[7];
        num_atoms = int_ptr[8];
    }
    int_ptr += SESSION_NUM_INTS(version);

    _ribbon_adjust = float_ptr[0];
    _worm_radius = float_ptr[1];
    float_ptr += SESSION_NUM_FLOATS(version);

    auto& atoms = structure()->atoms();
    for (decltype(num_atoms) i = 0; i < num_atoms; ++i) {
        add_atom(atoms[*int_ptr++], true);
    }
}

void
Residue::session_save(int** ints, float** floats) const
{
    _ribbon_rgba.session_save(ints, floats);
    _ring_rgba.session_save(ints, floats);

    auto& int_ptr = *ints;
    auto& float_ptr = *floats;

    int_ptr[0] = (int)_alt_loc;
    int_ptr[1] = (int)_ribbon_display;
    int_ptr[2] = (int)_ribbon_hide_backbone;
    int_ptr[3] = 0; // Unused, former ribbon selected
    int_ptr[4] = (int)_ss_id;
    int_ptr[5] = (int)_ss_type;
    int_ptr[6] = (int)_ring_display;
    int_ptr[7] = (int)_rings_are_thin;
    int_ptr[8] = atoms().size();
    int_ptr += SESSION_NUM_INTS();

    float_ptr[0] = _ribbon_adjust;
    float_ptr[1] = _worm_radius;
    float_ptr += SESSION_NUM_FLOATS();

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

void
Residue::set_chain_id(ChainID chain_id)
{
    if (chain_id != _chain_id) {
        if (_chain != nullptr)
            throw std::logic_error("Cannot set polymeric chain ID directly from Residue; must use Chain");
        _chain_id = chain_id;
        if (!structure()->lower_case_chains) {
            for (auto c: chain_id) {
                if (std::islower(c)) {
                    structure()->lower_case_chains = true;
                    break;
                }
            }
        }
        change_tracker()->add_modified(_structure, this, ChangeTracker::REASON_CHAIN_ID);
    }
}

void
Residue::set_number(int number) {
    if (number != _number) {
        _number = number;
        _numberings[structure()->res_numbering()] = number;
        change_tracker()->add_modified(structure(), this, ChangeTracker::REASON_NUMBER);
    }
}

void
Residue::set_templates_dir(const std::string& templates_dir)
{
    using tmpl::TemplateCache;
    TemplateCache::set_templates_dir(templates_dir);
}

void
Residue::set_user_templates_dir(const std::string& templates_dir)
{
    using tmpl::TemplateCache;
    TemplateCache::set_user_templates_dir(templates_dir);
}

std::string
Residue::str() const
{
    std::stringstream num_string;
    std::string ret = _name;
    ret += ' ';
    auto cid = chain_id();
    if (cid != " ") {
        ret += '/';
        ret += cid;
    }
    ret += ':';
    num_string << _number;
    ret += num_string.str();
    if (_insertion_code != ' ')
        ret += _insertion_code;
    return ret;
}

std::vector<Atom*>
Residue::template_assign(void (Atom::*assign_func)(const char*),
    const char* app, const char* template_dir, const char* extension) const
{
    // Returns atoms that did not receive assignments.  Can throw these exceptions:
    //   TA_TemplateSyntax:  template syntax error
    //   TA_NoTemplate:  no template found
    //   std::logic_error:  internal logic error
    using tmpl::TemplateCache;
    TemplateCache* tc = TemplateCache::template_cache();
    auto lookup_name = name();
    if (lookup_name == "UNK") {
        // treat as GLY if backbone atoms present
        bool treat_as_gly = true;
        for (auto bb_name: aa_min_backbone_names) {
            if (find_atom(bb_name) == nullptr) {
                treat_as_gly = false;
                break;
            }
        }
        if (treat_as_gly)
            lookup_name = "GLY";
    }
    TemplateCache::AtomMap* am = tc->res_template(lookup_name,
            app, template_dir, extension);

    std::vector<Atom*> unassigned;
    for (auto a: _atoms) {
        auto ami = am->find(a->name());
        if (ami == am->end()) {
            unassigned.push_back(a);
            continue;
        }

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
                        }
                    }
                } else if (ci.op == "?") {
                    // does the given atom exist in the residue?
                    if (find_atom(ci.operand.c_str()) != nullptr) {
                        cond_assigned = true;
                        if (ci.result != "-") {
                            (a->*assign_func)(ci.result);
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
        } else {
            unassigned.push_back(a);
        }
    }
    return unassigned;
}

}  // namespace atomstruct
