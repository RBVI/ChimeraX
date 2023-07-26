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

#include <algorithm>
#include <map>
#include <stdlib.h>

#define PDB_CONNECT_EXPORT
#include "connect.h"
#include <atomstruct/backbone.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Coord.h>
#include <atomstruct/destruct.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/tmpl/Atom.h>
#include <atomstruct/tmpl/residues.h>

namespace pdb_connect {

inline static void
add_bond(Atom* a1, Atom* a2)
{
    if (!a1->connects_to(a2))
        (void) a1->structure()->new_bond(a1, a2);
}

// bonded_dist:
//    Are given atoms close enough to bond?  If so, return bond distance,
// otherwise return zero.
static float
bonded_dist(Atom* a, Atom* b)
{
    float bond_len = Element::bond_length(a->element(), b->element());
    if (bond_len == 0.0)
        return 0.0;
    float max_bond_len_sq = bond_len + 0.4;
    max_bond_len_sq *= max_bond_len_sq;
    float dist_sq = a->coord().sqdistance(b->coord());
    if (dist_sq > max_bond_len_sq)
        return 0.0;
    return dist_sq;
}

// connect_atom_by_distance:
//    Connect an atom to a residue by distance criteria.  Don't connect a
// hydrogen or lone pair more than once, nor connect to one that's already
// bonded.
static void
connect_atom_by_distance(Atom* a, const Residue::Atoms& atoms,
    Residue::Atoms::const_iterator& a_it, std::set<Atom *>* conect_atoms)
{
    float short_dist = 0.0;
    Atom *close_atom = NULL;

    bool H_or_LP = a->element() <= Element::H;
    if (H_or_LP && !a->bonds().empty())
        return;
    Residue::Atoms::const_iterator end = atoms.end();
    for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != end; ++ai)
    {
        Atom *oa = *ai;
        if (a == oa || a->connects_to(oa)
        || (oa->element() <= Element::H && (H_or_LP || !oa->bonds().empty())))
            continue;
        if (ai < a_it && conect_atoms && conect_atoms->find(oa) == conect_atoms->end())
            // already checked
            continue;
        float dist = bonded_dist(a, oa);
        if (dist == 0.0)
            continue;
        if (H_or_LP) {
            if (short_dist != 0.0 && dist > short_dist)
                continue;
            short_dist = dist;
            close_atom = oa;
        } else {
            (void) a->structure()->new_bond(a, oa);
        }
    }
    if (H_or_LP && short_dist != 0) {
        (void) a->structure()->new_bond(a, close_atom);
    }
}

// connect_residue_by_distance:
//    Connect atoms in residue by distance.  This is an n-squared algorithm.
//    Takes into account alternate atom locations.  'conect_atoms' are
//    atoms whose connectivity is already known.
void
connect_residue_by_distance(Residue* r, std::set<Atom *>* conect_atoms)
{
    // connect up atoms in residue by distance
    const Residue::Atoms &atoms = r->atoms();
    for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != atoms.end(); ++ai) {
        Atom *a = *ai;
        if (conect_atoms && conect_atoms->find(a) != conect_atoms->end()) {
            // connectivity specified in a CONECT record, skip
            continue;
        }
        connect_atom_by_distance(a, atoms, ai, conect_atoms);
    }
}

// connect_residue_by_template:
//    Connect bonds in residue according to the given template.  Takes into
//    acount alternate atom locations.
static void
connect_residue_by_template(Residue* r, const tmpl::Residue* tr,
                        std::set<Atom *>* conect_atoms)
{
    // foreach atom in residue
    //    connect up like atom in template
    bool some_connectivity_unknown = false;
    std::set<Atom *> known_connectivity;
    const Residue::Atoms &atoms = r->atoms();
    for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != atoms.end(); ++ai) {
        Atom *a = *ai;
        if (conect_atoms->find(a) != conect_atoms->end()) {
            // connectivity specified in a CONECT record, skip
            known_connectivity.insert(a);
            continue;
        }
        tmpl::Atom *ta = tr->find_atom(a->name());
        if (ta == NULL) {
            some_connectivity_unknown = true;
            continue;
         }
        // non-template atoms will be able to connect to known atoms;
        // avoid rechecking known atoms though...
        known_connectivity.insert(a);

        for(auto tmpl_nb: ta->neighbors()) {
            Atom *b = r->find_atom(tmpl_nb->name());
            if (b == NULL)
                continue;
            if (!a->connects_to(b)) {
                (void) a->structure()->new_bond(a, b);
            }
        }
    }
    // For each atom that wasn't connected (i.e. not in template),
    // connect it by distance
    if (!some_connectivity_unknown)
        return;
    connect_residue_by_distance(r, &known_connectivity);
}

static std::map<Element::AS, unsigned long>  _saturationMap = {
    {Element::H, 1},
    {Element::O, 2}
};
static bool
saturated(Atom* a)
{
    int target = 4;
    auto info = _saturationMap.find((Element::AS)a->element().number());
    if (info != _saturationMap.end())
        target = (*info).second;
    int num_bonds = a->bonds().size();
    // metal-coordination pseudobonds not created yet; drop those bonds...
    for (auto b: a->bonds()) {
        if (b->other_atom(a)->element().is_metal())
            --num_bonds;
    }
    return num_bonds >= target;

}

// find_closest:
//    Find closest heavy atom to given heavy atom with residue that has
//    the same alternate location identifier (or none) and optionally return
Atom *
find_closest(Atom* a, Residue* r, float* ret_dist_sq, bool nonSaturated)
{
    if (a == NULL)
        return NULL;
    if (a->element().number() == 1)
        return NULL;
    const Residue::Atoms &r_atoms = r->atoms();
    Residue::Atoms::const_iterator ai = r_atoms.begin();
    if (ai == r_atoms.end())
        return NULL;
    Atom *closest = NULL;
    float dist_sq = 0.0;
    const Coord &c = a->coord();
    for (; ai != r_atoms.end(); ++ai) {
        Atom *oa = *ai;
        if (oa->element().number() == 1)
            continue;
        if (nonSaturated && saturated(oa))
            continue;
        if ((a->residue() == r && a->name() == oa->name()))
            continue;
        const Coord &c1 = oa->coord();
        float new_dist_sq = c.sqdistance(c1);
        if (closest != NULL && new_dist_sq >= dist_sq)
            continue;
        dist_sq = new_dist_sq;
        closest = oa;
    }
    if (ret_dist_sq)
        *ret_dist_sq = dist_sq;
    return closest;
}

// add_bond_nearest_pair:
//    Add a bond between two residues.
static void
add_bond_nearest_pair(Residue* from, Residue* to, bool any_length=true)
{
    Atom    *fsave, *tsave;

    find_nearest_pair(from, to, &fsave, &tsave);
    if (fsave != NULL) {
        if (!any_length && bonded_dist(fsave, tsave) == 0.0)
            return;
        add_bond(fsave, tsave);
    }
}

// find_nearest_pair:
//    Find closest atoms between two residues.
void
find_nearest_pair(Residue* from, Residue* to, Atom** ret_from_atom,
        Atom** ret_to_atom, float* ret_dist_sq)
{
    Atom    *fsave = NULL, *tsave = NULL;
    float    dist_sq = 0.0;

    const Residue::Atoms &atoms = from->atoms();
    for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != atoms.end();
    ++ai) {
        float    new_dist_sq;

        Atom *a = *ai;
        if (saturated(a))
            continue;
        Atom *b = find_closest(a, to, &new_dist_sq, true);
        if (b == NULL)
            continue;
        if (fsave == NULL || new_dist_sq < dist_sq) {
            fsave = a;
            tsave = b;
            dist_sq = new_dist_sq;
        }
    }
    if (ret_from_atom)
        *ret_from_atom = fsave;
    if (ret_to_atom)
        *ret_to_atom = tsave;
    if (ret_dist_sq)
        *ret_dist_sq = dist_sq;
}

static bool
hookup(Atom* a, Residue* res, bool definitely_connect=true)
{
    bool made_connection = false;
    Atom *b = find_closest(a, res, NULL, true);
    if (b != NULL) {
        if (!definitely_connect && b->coord().sqdistance(a->coord()) > 9.0)
            return false;
        add_bond(a, b);
        made_connection = true;
    }
    return made_connection;
}
static std::set<Bond*>
metal_coordination_bonds(Structure* as)
{
    std::set<Bond*> mc_bonds;
    std::set<Atom*> metals;
    for (auto& a: as->atoms())
        if (a->element().is_metal())
            metals.insert(a);

    for (auto metal: metals) {
        // skip large inorganic residues (that typically
        // don't distinguish metals by name)
        if (metal->residue()->atoms_map().count(metal->name()) > 1)
            continue;
        
        // bond -> pseudobond if:
        // 1) cross residue
        // 2) > 4 bonds
        // 3) neighbor is bonded to non-metal in same res
        //    unless metal has only one bond and neighbor has
        //    no lone pairs (e.g. residue EMC in 1cjx)
        std::set<Bond*> del_bonds;
        auto metal_bonds = metal->bonds();
        auto bi = metal_bonds.begin();
        for (auto nb: metal->neighbors()) {
            if (nb->residue() != metal->residue())
                del_bonds.insert(*bi);
            ++bi;
        }
        // eliminate cross-residue bond first to preserve FEO in 1av8
        if (metal->bonds().size() - del_bonds.size() > 4) {
            del_bonds.insert(metal_bonds.begin(), metal_bonds.end());
        } else {
            // metals with just one bond may be a legitimate compound
            if (metal_bonds.size() - del_bonds.size() == 1) {
                // avoid expensive atom-type computation by skipping
                // common elements we know cannot have lone pairs...

                // find the remaining bond
                Atom* nb = nullptr;
                auto nbi = metal->neighbors().begin();
                auto bi = metal->bonds().begin();
                for (; nbi != metal->neighbors().end(); ++nbi, ++bi ) {
                    if (del_bonds.find(*bi) == del_bonds.end()) {
                        nb = *nbi;
                        break;
                    }
                }
                if (nb == nullptr)
                    throw std::logic_error("All metal bonds in del_bonds");
                if (nb->element().number() == Element::C
                || nb->element().number() == Element::H) {
                    if (del_bonds.size() > 0)
                        mc_bonds.insert(del_bonds.begin(), del_bonds.end());
                    continue;
                }
                auto idatm_type = nb->idatm_type();
                auto idatm_info_map = Atom::get_idatm_info_map();
                auto info = idatm_info_map.find(idatm_type);
                if (info != idatm_info_map.end()
                && (info->second).substituents == (info->second).geometry
                && idatm_type != "Npl" && idatm_type != "N2+") {
                    // nitrogen exclusions for HEME C in 1og5
                    if (del_bonds.size() > 0)
                        mc_bonds.insert(del_bonds.begin(), del_bonds.end());
                    continue;
                }
            }
            bi = metal_bonds.begin();
            for (auto nb: metal->neighbors()) {
                if (metals.find(nb) != metals.end()) {
                    del_bonds.insert(*bi);
                } else {
                    for (auto gnb: nb->neighbors()) {
                        if (metals.find(gnb) == metals.end()
                        && gnb->residue() == nb->residue())
                            del_bonds.insert(*bi);
                    }
                }
                ++bi;
            }
        }
        if (del_bonds.size() > 0)
            mc_bonds.insert(del_bonds.begin(), del_bonds.end());
    }
    return mc_bonds;
}

void
find_and_add_metal_coordination_bonds(Structure* as)
{
    // make metal-coordination complexes
    auto notifications_off = DestructionNotificationsOff();
    auto mc_bonds = metal_coordination_bonds(as);
    if (mc_bonds.size() > 0) {
        auto pbg = as->pb_mgr().get_group(as->PBG_METAL_COORDINATION, 
            AS_PBManager::GRP_PER_CS);
        for (auto mc: mc_bonds) {
            for (auto& cs: as->coord_sets()) {
                pbg->new_pseudobond(mc->atoms(), cs);
            }
            as->delete_bond(mc);
        }
    }
}

void
find_missing_structure_bonds(Structure *as)
{
    std::vector<Bond*> long_bonds;
    for (auto& b: as->bonds()) {
        Atom* a1 = b->atoms()[0];
        Atom* a2 = b->atoms()[1];
        Residue* r1 = a1->residue();
        Residue* r2 = a2->residue();
        if (r1 == r2)
            continue;
        if (r1->chain_id() == r2->chain_id()
        && abs(r1->number() - r2->number()) < 2
        && b->polymeric_start_atom() != nullptr) // CA/P-only should use missing structure
            continue;
        auto idealBL = Element::bond_length(a1->element(), a2->element());
        if (b->sqlength() >= 3.0625 * idealBL * idealBL)
            // 3.0625 == 1.75 squared
            // (allows ASP 223.A OD2 <-> PLP 409.A N1 bond in 1aam
            // and SER 233.A OG <-> NDP 300.A O1X bond in 1a80
            // to not be classified as missing seqments)
            long_bonds.push_back(b);
        else {
            // if either end is CA/P in standard residue, then it has to be missing structure
            // except P to O3'
            if (r1->principal_atom() == a1 && (a1->name() != "P" || a2->name() != "O3'"))
                long_bonds.push_back(b);
            else if (r2->principal_atom() == a2 && (a2->name() != "P" || a1->name() != "O3'"))
                long_bonds.push_back(b);
        }
    }
    if (long_bonds.size() > 0) {
        auto pbg = as->pb_mgr().get_group(as->PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NORMAL);
        for (auto lb: long_bonds) {
            // the new "smart" missing-structure code will automatically make the pseudobond across
            // the gap when the bond is deleted _if_ it identifies it as a backbone bond, so no need
            // to explictly make a (duplicate) one in that case
            if (!lb->is_backbone())
                pbg->new_pseudobond(lb->atoms());
            as->delete_bond(lb);
        }
    }
}

// connect_structure:
//    Connect atoms in structure by template if one is found, or by distance.
//    Adjacent residues are connected if appropriate.
void
connect_structure(Structure* as, std::vector<Residue *>* start_residues,
    std::vector<Residue *>* end_residues, std::set<Atom *>* conect_atoms,
    std::set<MolResId>* mod_res, std::set<ResName>& polymeric_res_names,
    std::set<Residue*>& het_res)
{
    // walk the residues, connecting residues as appropriate and
    // connect the atoms within the residue
    Residue *link_res = NULL, *prev_res = NULL, *first_res = NULL;
    Atom *link_atom;
    AtomName link_atom_name;
    // start/end residues much more efficient to search as a map...
    std::set<Residue*> sres_map(start_residues->begin(), start_residues->end());
    std::set<Residue*> eres_map(end_residues->begin(), end_residues->end());
    for (Structure::Residues::const_iterator ri = as->residues().begin();
    ri != as->residues().end(); ++ri) {
        Residue *r = *ri;

        if (!first_res)
            first_res = r;
        // Before we add a bunch of bonds, make sure we're not already linked
        // to other residues via CONECT records [*not* just preceding
        // residue; see entry 209D, residues 5.C and 6.C].
        // For non-polymeric residues check non-metal non-disulphide linkages to
        // other residues; for polymeric just look for linkage to previous
        // residue.
        // Can't just check conect_atoms because if the previous
        // residue is non-polymeric and this one is, only the cross-residue
        // bond may be in the CONECT records and therefore this
        // residue's connected atom won't be in conect_atoms (which
        // is only for atoms whose complete connectivity is
        // specified by CONECT records)
        bool is_mod_res = mod_res->find(MolResId(r)) != mod_res->end();
        bool prelinked = false;
        if (link_res != NULL) {
            for (auto a: r->atoms()) {
                for (auto b: a->bonds()) {
                    auto other = b->other_atom(a);
                    if (other->residue() != r) {
                        if (polymeric_res_names.find(a->residue()->name())
                        == polymeric_res_names.end() && !is_mod_res) {
                            // not coordination...
                            if (!(other->element().is_metal() || a->element().is_metal())
                            // and not disulphide...
                            && !(other->element() == Element::S && a->element() == Element::S)) {
                                prelinked = true;
                                break;
                            }
                        } else {
                            // non-Het should always link to preceding...
                            // (ignore disulphide linkage)
                            if (other->residue() == link_res
                            && !(other->element() == Element::S && a->element() == Element::S)) {
                                prelinked = true;
                                break;
                            }
                        }
                    }
                }
                if (prelinked)
                    break;
            }
        }
        const tmpl::Residue *tr;
        if (is_mod_res)
            // residue in MODRES record;
            // don't try to use template connectivity
            tr = NULL;
        else
            tr = tmpl::find_template_residue(r->name(),
                sres_map.find(r) != sres_map.end(),
                eres_map.find(r) != eres_map.end());
        if (tr != NULL) {
            connect_residue_by_template(r, tr, conect_atoms);
            // if PDB uses non-standard (or old standard) hydrogen names
            // then there may be "floating" hydrogens.  Check for that.
            for (auto a: r->atoms()) {
                if (a->bonds().size() == 0 && a->element().number() == 1) {
                    connect_residue_by_distance(r, conect_atoms);
                    break;
                }
            }
        } else
            connect_residue_by_distance(r, conect_atoms);

        // connect up previous residue
        if (link_res != NULL) {
            auto pt = Sequence::rname_polymer_type(link_res->name());
            if (prelinked) {
                ; // do nothing
            } else if (pt != PT_NONE && pt == Sequence::rname_polymer_type(r->name())) {
                Atom* prev_connect = nullptr;
                auto& backbone_names = (pt == PT_AMINO) ?
                    Residue::aa_min_ordered_backbone_names :
                    Residue::na_min_ordered_backbone_names;
                bool made_connection = false;
                for (auto i = backbone_names.rbegin(); i != backbone_names.rend(); ++i) {
                    auto prev_bba = link_res->find_atom(*i);
                    if (prev_bba != nullptr) {
                        prev_connect = prev_bba;
                        break;
                    }
                }
                if (prev_connect != nullptr) {
                    for (auto bb_name: backbone_names) {
                        auto bba = r->find_atom(bb_name);
                        if (bba != nullptr) {
                            add_bond(prev_connect, bba);
                            made_connection = true;
                            break;
                        }
                    }
                }
                if (!made_connection)
                    add_bond_nearest_pair(link_res, r);
            } else if (tr == NULL || tr->chief() == NULL) {
                add_bond_nearest_pair(link_res, r);
            } else {
                bool made_connection = false;
                // don't definitely connect a leading HET residue
                bool definitely_connect = (link_res != first_res
                    || polymeric_res_names.find(first_res->name()) != polymeric_res_names.end());
                Atom *chief = r->find_atom(tr->chief()->name());
                if (chief != NULL) {
                    // 1vqn, chain 5, is a nucleic/amino acid
                    // hybrid with the na/aa connectivity in
                    // CONECT records; prevent also adding a
                    // chief-link bond
                    if (saturated(chief)) {
                        made_connection = true;
                    } else if (link_atom != NULL) {
                        if (!saturated(link_atom)) {
                            add_bond(link_atom, chief);
                        }
                        made_connection = true;
                    } else {
                        made_connection = hookup(chief, link_res, definitely_connect);
                    }
                }
                if (!made_connection && definitely_connect) {
                    add_bond_nearest_pair(link_res, r);
                }
            }
        } else if (r->atoms().size() > 1 && prev_res != NULL
                && prev_res->chain_id() == r->chain_id()
                && polymeric_res_names.find(r->name()) == polymeric_res_names.end()
                && conect_atoms->find((*r->atoms().begin())) == conect_atoms->end()) {
            // multi-atom HET residues with no CONECTs (i.e. _not_
            // a standard PDB entry) _may_ connect to previous residue...
            add_bond_nearest_pair(prev_res, r, false);
        }

        prev_res = r;
        if (std::find(end_residues->begin(), end_residues->end(), r)
        != end_residues->end()) {
            link_res = NULL;
        } else {
            link_res = r;
            if (tr == NULL || tr->link() == NULL) {
                link_atom_name = "";
                link_atom = NULL;
            } else {
                link_atom_name = tr->link()->name();
                link_atom = r->find_atom(link_atom_name);
            }
        }
    }

    // if no CONECT/MODRES records and there are non-standard residues not
    // in HETATM records (i.e. this is clearly a non-standard PDB
    // like those output by CCP4's refmac), then examine the inter-
    // residue bonds and break the non-physical ones (> 1.5 normal length)
    // involving at least one non-standard residue
    bool break_long = false;
    if (conect_atoms->empty() && mod_res->empty()) {
        for (Structure::Residues::const_iterator ri=as->residues().begin()
        ; ri != as->residues().end(); ++ri) {
            Residue *r = *ri;
            if (polymeric_res_names.find(r->name()) != polymeric_res_names.end())
                continue;
            if (het_res.find(r) == het_res.end()) {
                break_long = true;
                break;
            }
        }
    }
    auto notifications_off = DestructionNotificationsOff();
    if (break_long) {
        bool missing_structure = false;
        std::vector<Bond *> break_these;
        for (Structure::Bonds::const_iterator bi = as->bonds().begin();
        bi != as->bonds().end(); ++bi) {
            Bond *b = *bi;
            const Bond::Atoms & atoms = b->atoms();
            Residue *r1 = atoms[0]->residue();
            Residue *r2 = atoms[1]->residue();
            if (r1 == r2)
                continue;
            if (polymeric_res_names.find(r1->name()) != polymeric_res_names.end()
            && polymeric_res_names.find(r2->name()) != polymeric_res_names.end()) {
                missing_structure = true;
                continue;
            }
            // break if non-physical
            float criteria = 1.5 * Element::bond_length(atoms[0]->element(),
                atoms[1]->element());
            if (criteria * criteria < b->sqlength())
                break_these.push_back(b);
        }
        for (std::vector<Bond *>::iterator bi = break_these.begin();
        bi != break_these.end(); ++bi) {
            Bond *b = *bi;
            as->delete_bond(b);
        }
        find_and_add_metal_coordination_bonds(as);
        if (missing_structure)
            find_missing_structure_bonds(as);
    } else {
        // turn long inter-residue bonds into "missing structure" pseudobonds
        find_and_add_metal_coordination_bonds(as);
        find_missing_structure_bonds(as);
    }
}

} // namespace pdb_connect
