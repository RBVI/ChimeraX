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
#include <cctype>
#include <set>

#include <logger/logger.h>
#include <pysupport/convert.h>
#include <arrays/pythonarray.h>   // Uses python_int_array(), python_float_array()

#include "Python.h"

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "backbone.h"
#include "Bond.h"
#include "CoordSet.h"
#include "destruct.h"
#include "polymer.h"
#include "Sequence.h"
#include "Structure.h"
#include "PBGroup.h"
#include "Pseudobond.h"
#include "Residue.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::Structure>;

namespace atomstruct {

const char*  Structure::PBG_METAL_COORDINATION = "metal coordination bonds";
const char*  Structure::PBG_MISSING_STRUCTURE = "missing structure";
const char*  Structure::PBG_HYDROGEN_BONDS = "hydrogen bonds";

Structure::Structure(PyObject* logger):
    _active_coord_set(nullptr), _chains(nullptr),
    _change_tracker(DiscardingChangeTracker::discarding_change_tracker()),
    _idatm_valid(false), _logger(logger),
    _pb_mgr(this), _polymers_computed(false), _recompute_rings(true),
    _ss_assigned(false), _structure_cats_dirty(true),
    asterisks_translated(false), is_traj(false),
    lower_case_chains(false), pdb_version(0), ss_ids_normalized(false)
{
    for (int i=0; i<3; ++i) {
        for (int j=0; j<4; ++j) {
            _position[i][j] = (i == j ? 1.0 : 0.0);
        }
    }
    change_tracker()->add_created(this, this);
}

Structure::~Structure() {
    // assign to variable so that it lives to end of destructor
    auto du = DestructionUser(this);
    change_tracker()->add_deleted(this, this);
    // delete bonds before atoms since bond destructor uses info
    // from its atoms
    for (auto b: _bonds)
        delete b;
    for (auto a: _atoms)
        delete a;
    if (_chains != nullptr) {
        for (auto ch: *_chains) {
            ch->clear_residues();
            // since Python layer may be referencing Chain, only
            // delete immediately if no Python-layer references,
            // otherwise decref and let garbage collection work
            // its magic (__del__ will destroy C++ side)
            auto inst = ch->py_instance(false);

            // py_instance() returns new reference, so ...
            Py_DECREF(inst);

            // If ref count is 1 afterward, _don't_ simply decref
            // again.  That will cause the Python __del__ to 
            // execute, which will see that the C++ side is not
            // destroyed yet, and call the destructor -- which
            // due to inheriting from PyInstance, will decref
            // __again__. Instead, just destroy the chain to
            // indirectly accomplish the second decref.
            if (inst == Py_None || Py_REFCNT(inst) == 1)
                delete ch;
            else
                // decref "C++ side" reference
                Py_DECREF(inst);
        }
        delete _chains;
    }
    for (auto r: _residues)
        delete r;
    for (auto cs: _coord_sets)
        delete cs;
}

std::map<Residue *, char>
Structure::best_alt_locs() const
{
    // check the common case of all blank alt locs first...
    bool all_blank = true;
    const Atoms& as = atoms();
    for (auto ai = as.begin(); ai != as.end(); ++ai) {
        if (!(*ai)->_alt_loc_map.empty()) {
            all_blank = false;
            break;
        }
    }
    std::map<Residue *, char> best_locs;
    if (all_blank) {
        return best_locs;
    }

    // go through the residues and collate a group of residues with
    //   related alt locs
    // use the alt loc with the highest average occupancy; if tied,
    //  the lowest bfactors; if tied, first alphabetically
    std::set<Residue *> seen;
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = *ri;
        if (seen.find(r) != seen.end())
            continue;
        seen.insert(r);
        std::set<Residue *> res_group;
        std::set<char> alt_loc_set;
        for (auto ai = r->_atoms.begin(); ai != r->_atoms.end(); ++ai) {
            Atom *a = *ai;
            alt_loc_set = a->alt_locs();
            if (!alt_loc_set.empty())
                break;
        }
        // if residue has no altlocs, skip it
        if (alt_loc_set.empty())
            continue;
        // for this residue and neighbors linked through alt loc,
        // collate occupancy/bfactor info
        res_group.insert(r);
        std::vector<Residue *> todo;
        todo.push_back(r);
        std::map<char, int> occurances;
        std::map<char, float> occupancies, bfactors;
        while (!todo.empty()) {
            Residue *cr = todo.back();
            todo.pop_back();
            for (auto ai = cr->_atoms.begin(); ai != cr->_atoms.end(); ++ai) {
                Atom *a = *ai;
                bool check_neighbors = true;
                for (auto alsi = alt_loc_set.begin(); alsi != alt_loc_set.end();
                ++alsi) {
                    char alt_loc = *alsi;
                    if (!a->has_alt_loc(alt_loc)) {
                        check_neighbors = false;
                        break;
                    }
                    occurances[alt_loc] += 1;
                    Atom::_Alt_loc_info &info = a->_alt_loc_map[alt_loc];
                    occupancies[alt_loc] += info.occupancy;
                    bfactors[alt_loc] += info.bfactor;
                }
                if (check_neighbors) {
                    for (auto nb: a->neighbors()) {
                        Residue *nr = nb->residue();
                        if (nr != cr && nb->alt_locs() == alt_loc_set
                        && seen.find(nr) == seen.end()) {
                            seen.insert(nr);
                            todo.push_back(nr);
                            res_group.insert(nr);
                        }
                    }
                }
            }
        }
        // go through the occupancy/bfactor info and decide on
        // the best alt loc
        char best_loc = '\0';
        std::vector<char> alphabetic_alt_locs(alt_loc_set.begin(),
            alt_loc_set.end());
        std::sort(alphabetic_alt_locs.begin(), alphabetic_alt_locs.end());
        float best_occupancies = 0.0, best_bfactors = 0.0;
        for (auto ali = alphabetic_alt_locs.begin();
        ali != alphabetic_alt_locs.end(); ++ali) {
            char al = *ali;
            bool is_best = best_loc == '\0';
            float occ = occupancies[al] / occurances[al];
            if (!is_best) {
                if (occ > best_occupancies)
                    is_best = true;
                else if (occ < best_occupancies)
                    continue;
            }
            float bf = bfactors[al] / occurances[al];
            if (!is_best) {
                if (bf < best_bfactors)
                    is_best = true;
                else if (bf < best_bfactors)
                    continue;
            }
            if (is_best) {
                best_loc = al;
                best_occupancies = occ;
                best_bfactors = bf;
            }
        }
        // note the best alt loc for these residues in the map
        for (auto rgi = res_group.begin(); rgi != res_group.end(); ++rgi) {
            best_locs[*rgi] = best_loc;
        }
    }

    return best_locs;
}

void
Structure::bonded_groups(std::vector<std::vector<Atom*>>* groups,
    bool consider_missing_structure) const
{
    // find connected atomic structures, considering missing-structure pseudobonds
    std::map<Atom*, std::vector<Atom*>> pb_connections;
    if (consider_missing_structure) {
        auto pbg = const_cast<Structure*>(this)->_pb_mgr.get_group(PBG_MISSING_STRUCTURE,
            AS_PBManager::GRP_NONE);
        if (pbg != nullptr) {
            for (auto& pb: pbg->pseudobonds()) {
                auto a1 = pb->atoms()[0];
                auto a2 = pb->atoms()[1];
                pb_connections[a1].push_back(a2);
                pb_connections[a2].push_back(a1);
            }
        }
    }
    std::set<Atom*> seen;
    for (auto a: atoms()) {
        if (seen.find(a) != seen.end())
            continue;
        groups->emplace_back();
        std::vector<Atom*>& bonded = groups->back();
        std::set<Atom*> pending;
        pending.insert(a);
        while (pending.size() > 0) {
            Atom* pa = *(pending.begin());
            pending.erase(pa);
            if (seen.find(pa) != seen.end())
                continue;
            seen.insert(pa);
            bonded.push_back(pa);
            if (pb_connections.find(pa) != pb_connections.end()) {
                for (auto conn: pb_connections[pa]) {
                    pending.insert(conn);
                }
            }
            for (auto nb: pa->neighbors())
                pending.insert(nb);
        }
    }
}

void
Structure::change_chain_ids(const std::vector<StructureSeq*> changing_chains,
    const std::vector<ChainID> new_ids, bool non_polymeric)
{
    if (changing_chains.size() != new_ids.size())
        throw std::logic_error("Number of chains to change IDs for must match number of IDs");

    std::set<StructureSeq*> unique_chains(changing_chains.begin(), changing_chains.end());
    if (unique_chains.size() < changing_chains.size())
        throw std::logic_error("List of chains to change IDs for must not contain duplicates");

    std::set<ChainID> unique_ids(new_ids.begin(), new_ids.end());
    if (unique_ids.size() < new_ids.size())
        throw std::logic_error("List of chain IDs to change to must not contain duplicates");

    // can change StructureSeq chain IDs freely; Chain chain IDs cannot conflict with existing
    std::set<ChainID> other_ids;
    for (auto chain: chains())
        if (unique_chains.find(chain) == unique_chains.end())
            other_ids.insert(chain->chain_id());
    std::map<StructureSeq*,ChainID> chain_remapping;
    std::map<ChainID,ChainID> id_remapping;
    auto new_ids_i = new_ids.begin();
    for (auto chain: changing_chains) {
        if (chain->is_chain() && other_ids.find(*new_ids_i) != other_ids.end()) {
            std::stringstream err_msg;
            err_msg << "New chain ID '" << *new_ids_i << "' already assigned to another chain";
            throw std::logic_error(err_msg.str().c_str());
        }
        chain_remapping[chain] = *new_ids_i;
        id_remapping[chain->chain_id()] = *new_ids_i++;
    }

    for (auto chain: changing_chains)
        chain->set_chain_id(chain_remapping[chain]);

    if (non_polymeric)
        for (auto r: residues())
            if (r->chain() == nullptr && id_remapping.find(r->chain_id()) != id_remapping.end())
                r->set_chain_id(id_remapping[r->chain_id()]);
}

static void
_copy_pseudobonds(Proxy_PBGroup* pbgc, const Proxy_PBGroup::Pseudobonds& pbs,
    std::map<Atom*, Atom*>& amap, CoordSet* cs = nullptr)
{
    for (auto pb: pbs) {
        const Connection::Atoms &a = pb->atoms();
        Pseudobond *pbc;
        if (cs == nullptr)
            pbc = pbgc->new_pseudobond(amap[a[0]], amap[a[1]]);
        else
            pbc = pbgc->new_pseudobond(amap[a[0]], amap[a[1]], cs);
        pbc->set_display(pb->display());
        pbc->set_hide(pb->hide());
        pbc->set_color(pb->color());
        pbc->set_halfbond(pb->halfbond());
        pbc->set_radius(pb->radius());
        pbc->set_shown_when_atoms_hidden(pb->shown_when_atoms_hidden());
    }
}

void Structure::_copy(Structure* s, PositionMatrix coord_adjust,
    std::map<ChainID, ChainID>* chain_id_map) const
{
    // if chain_id_map is not nullptr, then we are combining this structure into existing
    // structure s
    for (auto h = metadata.begin() ; h != metadata.end() ; ++h)
        s->metadata[h->first] = h->second;
    s->pdb_version = pdb_version;
    s->_polymers_computed = true;
    if (chain_id_map == nullptr) {
        s->lower_case_chains = lower_case_chains;
        s->set_ss_assigned(ss_assigned());
        s->set_ribbon_tether_scale(ribbon_tether_scale());
        s->set_ribbon_tether_shape(ribbon_tether_shape());
        s->set_ribbon_tether_sides(ribbon_tether_sides());
        s->set_ribbon_tether_opacity(ribbon_tether_opacity());
        s->set_ribbon_show_spine(ribbon_show_spine());
        s->set_ribbon_orientation(ribbon_orientation());
        s->set_ribbon_mode_helix(ribbon_mode_helix());
        s->set_ribbon_mode_strand(ribbon_mode_strand());
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                s->_position[i][j] = _position[i][j];
    } else {
        if (s->ss_assigned())
            s->set_ss_assigned(ss_assigned());
    }

    if (chain_id_map == nullptr) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                s->_position[i][j] = _position[i][j];
    }

    std::map<Residue*, Residue*> rmap;
    for (auto r: residues()) {
        ChainID cid;
        if (chain_id_map == nullptr)
            cid = r->chain_id();
        else {
            auto cid_i = chain_id_map->find(r->chain_id());
            if (cid_i == chain_id_map->end())
                cid = r->chain_id();
            else
                cid = cid_i->second;
            if (s->lower_case_chains) {
                for (auto c: cid) {
                    if (isupper(c)) {
                        s->lower_case_chains = false;
                        break;
                    }
                }
            }
        }
        Residue* cr = s->new_residue(r->name(), cid, r->number(), r->insertion_code());
        cr->set_mmcif_chain_id(r->mmcif_chain_id());
        cr->set_ribbon_display(r->ribbon_display());
        cr->set_ribbon_color(r->ribbon_color());
        cr->set_ring_display(r->ring_display());
        cr->set_ring_color(r->ring_color());
        cr->set_thin_rings(r->thin_rings());
        cr->set_ss_id(r->ss_id());
        cr->set_ss_type(r->ss_type());
        cr->_alt_loc = r->_alt_loc;
        cr->_ribbon_hide_backbone = r->_ribbon_hide_backbone;
        cr->_ribbon_adjust = r->_ribbon_adjust;
        rmap[r] = cr;
    }
    std::map<CoordSet*, CoordSet*> cs_map;
    unsigned int coord_base = 0;
    if (chain_id_map == nullptr) {
        for (auto cs: coord_sets()) {
            auto new_cs = s->new_coord_set(cs->id());
            *new_cs = *cs;
            if (coord_adjust != nullptr)
                new_cs->xform(coord_adjust);
            cs_map[cs] = new_cs;
        }
        s->set_active_coord_set(cs_map[active_coord_set()]);
    } else {
        coord_base = s->active_coord_set()->coords().size();
        if (s->coord_sets().size() != coord_sets().size()) {
            // copy just the current coord set onto the current one, and prune others from combination
            auto active = s->active_coord_set();
            auto add_cs = active_coord_set();
            if (coord_adjust != nullptr)
                add_cs->xform(coord_adjust);
            active->add_coords(add_cs);
            s->_coord_sets.clear();
            s->_coord_sets.push_back(active);
            cs_map[active_coord_set()] = active;
        } else {
            for (decltype(_coord_sets)::size_type i = 0; i < coord_sets().size(); ++i) {
                auto s_cs = s->coord_sets()[i];
                auto c_cs = coord_sets()[i];
                if (coord_adjust != nullptr)
                    c_cs->xform(coord_adjust);
                s_cs->add_coords(c_cs);
                cs_map[c_cs] = s_cs;
            }
        }
    }

    int serial_base = 0;
    if (chain_id_map != nullptr) {
        for (auto a: s->atoms())
            if (a->serial_number() > serial_base)
                serial_base = a->serial_number();
    }

    set_alt_loc_change_notify(false);
    std::map<Atom*, Atom*> amap;
    for (auto a: atoms()) {
        Atom* ca = s->new_atom(a->name().c_str(), a->element());
        Residue *cr = rmap[a->residue()];
        cr->add_atom(ca);	// Must set residue before setting alt locs
        ca->_coord_index = coord_base + a->coord_index();
        std::set<char> alocs = a->alt_locs();
        if (!alocs.empty()) {
            char aloc = a->alt_loc();	// Remember original alt loc.
            for (auto ali = alocs.begin() ; ali != alocs.end() ; ++ali) {
                char al = *ali;
                a->set_alt_loc(al);
                ca->set_alt_loc(al, true);
                auto crd = a->coord();
                if (coord_adjust != nullptr)
                    crd = crd.mat_mul(coord_adjust);
                ca->set_coord(crd);
                ca->set_bfactor(a->bfactor());
                ca->set_occupancy(a->occupancy());
            }
            a->set_alt_loc(aloc);	// Restore original alt loc.
            ca->set_alt_loc(aloc);
        } else {
            ca->set_bfactor(a->bfactor());
            ca->set_occupancy(a->occupancy());
        }
        ca->set_serial_number(serial_base + a->serial_number());
        ca->set_draw_mode(a->draw_mode());
        ca->set_radius(a->radius());
        ca->set_color(a->color());
        ca->set_display(a->display());
        amap[a] = ca;
    }
    set_alt_loc_change_notify(true);
    
    for (auto b: bonds()) {
        const Bond::Atoms& a = b->atoms();
        Bond* cb = s->_new_bond(amap[a[0]], amap[a[1]], true);
        cb->set_display(b->display());
        cb->set_color(b->color());
        cb->set_halfbond(b->halfbond());
        cb->set_radius(b->radius());
    }

    if (_chains != nullptr) {
        if (chain_id_map == nullptr)
            s->_chains = new Chains();
        for (auto c: chains()) {
            ChainID cid;
            if (chain_id_map == nullptr)
                cid = c->chain_id();
            else {
                auto cid_i = chain_id_map->find(c->chain_id());
                if (cid_i == chain_id_map->end())
                    cid = c->chain_id();
                else
                    cid = cid_i->second;
            }
            auto cc = s->_new_chain(cid, c->polymer_type());
            StructureSeq::Residues bulk_residues;
            for (auto r: c->residues()) {
                if (r == nullptr)
                    bulk_residues.push_back(nullptr);
                else
                    bulk_residues.push_back(rmap[r]);
            }
            cc->bulk_set(bulk_residues, &c->contents());
            cc->set_circular(c->circular());
            cc->set_from_seqres(c->from_seqres());
            cc->set_description(c->description());
        }
    }

    // Copy pseudobond groups.
    const AS_PBManager::GroupMap &gm = pb_mgr().group_map();
    for (auto gi = gm.begin() ; gi != gm.end() ; ++gi) {
        Proxy_PBGroup *pbg = gi->second;
        auto group_type = pbg->group_type();
        Proxy_PBGroup *pbgc = s->pb_mgr().get_group(gi->first, group_type);
        if (group_type == AS_PBManager::GRP_NORMAL) {
            _copy_pseudobonds(pbgc, pbg->pseudobonds(), amap);
        } else {
            // per coordinate set pseudobond groups
            for (auto cs: coord_sets()) {
                if (chain_id_map != nullptr && s->coord_sets().size() == 1) {
                    // coord sets may have been pruned; only copy current
                    if (cs != active_coord_set())
                        continue;
                }
                _copy_pseudobonds(pbgc, pbg->pseudobonds(cs), amap, cs_map[cs]);
            }
        }
    }
}

void
Structure::combine_sym_atoms()
{
    std::map<std::pair<Coord, int>, std::vector<Atom*>> sym_info;
    for (auto a: atoms()) {
        if (a->neighbors().size() == 0)
            sym_info[std::make_pair(a->coord(), a->element().number())].push_back(a);
    }
    std::vector<Atom*> extras;
    for (auto key_atoms: sym_info) {
        auto& sym_atoms = key_atoms.second;
        if (sym_atoms.size() == 1)
            continue;
        logger::info(_logger,
                "Combining ", sym_atoms.size(), " symmetry atoms into ", sym_atoms.front()->str());
        extras.insert(extras.end(), sym_atoms.begin()+1, sym_atoms.end());
    }
    if (extras.size() > 0)
        delete_atoms(extras);
}

Structure*
Structure::copy() const
{
    Structure* s = new Structure(_logger);
    _copy(s);
    return s;
}

void
Structure::delete_alt_locs()
{
    // make current alt locs into "regular" atoms and remove other alt locs
    for (auto a: _atoms) {
        if (a->alt_loc() == ' ')
            continue;
        auto aniso_u = a->aniso_u();
        auto bfactor = a->bfactor();
        auto coord = a->coord();
        auto occupancy = a->occupancy();
        auto serial_number = a->serial_number();
        a->_alt_loc = ' ';
        change_tracker()->add_modified(this, a, ChangeTracker::REASON_ALT_LOC);
        if (aniso_u != nullptr)
            a->set_aniso_u((*aniso_u)[0], (*aniso_u)[1], (*aniso_u)[2],
                (*aniso_u)[3], (*aniso_u)[4], (*aniso_u)[5]);
        // don't clear map until we've possibly used the aniso values
        a->_alt_loc_map.clear();
        a->set_bfactor(bfactor);
        a->set_coord(coord);
        a->set_occupancy(occupancy);
        a->set_serial_number(serial_number);
    }
}

void
Structure::_delete_atom(Atom* a)
{
    auto db = DestructionBatcher(this);
    // if we're a backbone atom connecting to a backbone atom in an "adjacent" residue,
    // need to insert missing-structure pseudobond. "adjacent" considers missing-structure
    // pseudobonds
    if (a->_rings.size() > 0)
        _recompute_rings = true;
    if (a->is_backbone(BBE_MIN)) {
        std::vector<Atom*> missing_partners;
        auto pbg = _pb_mgr.get_group(PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
        if (pbg != nullptr) {
            for (auto& pb: pbg->pseudobonds()) {
                Atom* a1 = pb->atoms()[0];
                Atom* a2 = pb->atoms()[1];
                if (a1 == a)
                    missing_partners.push_back(a2);
                else if (a2 == a)
                    missing_partners.push_back(a1);
            }
        }
        if (missing_partners.size()  == 2) {
            // Just connect those bad boys!
            pbg->new_pseudobond(*missing_partners.begin(), *(missing_partners.begin()+1));
        } else if (missing_partners.size() == 1) {
            // connect to other backbone atom in this residue
            for (auto nb: a->neighbors())
                if (nb->is_backbone(BBE_MIN)) {
                    pbg->new_pseudobond(*missing_partners.begin(), nb);
                    break;
                }
        } else {
            // connect neighboring backbone atoms, if at least one in adjacent residue
            Atom* bb_in_res = nullptr;
            Atom* bb_other_res = nullptr;
            for (auto nb: a->neighbors()) {
                if (!nb->is_backbone(BBE_MIN))
                    continue;
                if (nb->residue() == a->residue())
                    bb_in_res = nb;
                else
                    bb_other_res = nb;
            }
            if (bb_in_res != nullptr && bb_other_res != nullptr) {
                auto ms_pbg = _pb_mgr.get_group(PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NORMAL);
                ms_pbg->new_pseudobond(bb_in_res, bb_other_res);
            }
        }
    }
    if (a->element().number() == 1)
        --_num_hyds;
    for (auto b: a->bonds()) {
        b->other_atom(a)->remove_bond(b);
        typename Bonds::iterator bi = std::find_if(_bonds.begin(), _bonds.end(),
            [&b](Bond* ub) { return ub == b; });
        _bonds.erase(bi);
        delete b;
    }
    a->residue()->remove_atom(a);
    typename Atoms::iterator i = std::find_if(_atoms.begin(), _atoms.end(),
        [&a](Atom* ua) { return ua == a; });
    _atoms.erase(i);
    set_gc_shape();
    set_gc_adddel();
    _idatm_valid = false;
    delete a;
}

void
Structure::delete_atom(Atom* a)
{
    if (a->structure() != this) {
        logger::error(_logger, "Atom ", a->residue()->str(), " ", a->name(),
            " does not belong to the structure that it's being deleted from.");
        throw std::invalid_argument("delete_atom called for Atom not in AtomicStructure/Structure");
    }
    if (atoms().size() == 1) {
        Py_XDECREF(py_call_method("cpp_del_model"));
        return;
    }
    auto r = a->residue();
    if (r->atoms().size() == 1) {
        _delete_residue(r);
        return;
    }
    _delete_atom(a);
}

static Atom*
_find_attachment_atom(Residue* r, const std::set<Atom*>& atoms, std::set<Atom*>& bond_losers,
    std::set<Atom*>& begin_missing_structure_atoms, std::set<Atom*>& end_missing_structure_atoms,
    bool left_side)
{
    // if old missing-structure atom is still there, use it
    for (auto a: begin_missing_structure_atoms) {
        if (atoms.find(a) != atoms.end())
            continue;
        if (a->residue() == r) {
            if (end_missing_structure_atoms.find(a) == end_missing_structure_atoms.end())
                return a;
        }
    }

    // then any backbone bond loser; then _any_ bond loser
    for (auto a: bond_losers)
        if (a->residue() == r && a->is_backbone(BBE_MIN))
            return a;
    for (auto a: bond_losers)
        if (a->residue() == r)
            return a;

    // in some weird situations where the residue is internally decimated, the formerly
    // connecting atom may not have been bonded to any other atom of the residue so...
    auto backbone_names = r->ordered_min_backbone_atom_names();
    if (backbone_names == nullptr)
        throw std::logic_error("Missing-structure adjacent residue is not polymeric!?!");
    if (left_side) {
        // we're on the "left side" of the gap, so need to look through backbone names
        // in reverse order
        auto reversed_names = *backbone_names;
        std::reverse(reversed_names.begin(), reversed_names.end());
        for (auto bb_name: reversed_names) {
            Atom *a = r->find_atom(bb_name);
            if (a != nullptr)
                return a;
        }
    } else {
        // we're on the "right side" of the gap, so backbone names in normal order
        for (auto bb_name: *backbone_names) {
            Atom *a = r->find_atom(bb_name);
            if (a != nullptr)
                return a;
        }
    }

    // Okay, no main backbone atoms exist -- use any atom!
    for (auto a: r->atoms())
        return a;
    throw std::logic_error("Residue has no atoms!?!");
}

void
Structure::_delete_atoms(const std::set<Atom*>& atoms, bool verify)
{
    if (verify)
        for (auto a: atoms)
            if (a->structure() != this) {
                logger::error(_logger, "Atom ", a->residue()->str(), " ", a->name(),
                    " does not belong to the structure that it's being deleted from.");
                throw std::invalid_argument("delete_atoms called with Atom not in"
                    " AtomicStructure/Structure");
            }
    if (atoms.size() == _atoms.size()) {
        delete this;
        return;
    }
    // want to put missing-structure pseudobonds across new mid-chain gaps,
    // so note which residues connected to their next existing one, considering
    // pre-existing missing-structure pseudobonds
    std::map<Residue*, int> begin_ri_lookup, end_ri_lookup;
    std::map<int, Residue*> begin_ir_lookup, end_ir_lookup;
    std::map<Residue*, bool> begin_res_connects_to_next, end_res_connects_to_next;
    std::set<Atom*> begin_left_missing_structure_atoms, end_left_missing_structure_atoms,
        begin_right_missing_structure_atoms, end_right_missing_structure_atoms;
    _get_interres_connectivity(begin_ri_lookup, begin_ir_lookup, begin_res_connects_to_next,
        begin_left_missing_structure_atoms, begin_right_missing_structure_atoms);

    std::map<Residue*, std::vector<Atom*>> res_del_atoms;
    for (auto a: atoms) {
        res_del_atoms[a->residue()].push_back(a);
        if (a->element().number() == 1)
            --_num_hyds;
    }
    std::set<Residue*> res_removals;
    for (auto& r_atoms: res_del_atoms) {
        auto r = r_atoms.first;
        auto& dels = r_atoms.second;
        if (dels.size() == r->atoms().size()) {
            res_removals.insert(r);
        } else {
            for (auto a: dels)
                r->remove_atom(a);
        }
    }
    if (res_removals.size() > 0) {
        if (_chains != nullptr) {
            std::map<Chain*, std::set<Residue*>> chain_res_removals;
            for (auto r: res_removals)
                if (r->chain() != nullptr) {
                    chain_res_removals[r->chain()].insert(r);
                }
            for (auto chain_residues: chain_res_removals) {
                chain_residues.first->remove_residues(chain_residues.second);
                set_gc_ribbon();
            }
        }
        // remove_if apparently doesn't guarantee that the _back_ of
        // the vector is all the removed items -- there could be second
        // copies of the retained values in there, so do the delete as
        // part of the lambda rather than in a separate pass through
        // the end of the vector
        auto new_end = std::remove_if(_residues.begin(), _residues.end(),
            [&res_removals](Residue* r) {
                bool rm = res_removals.find(r) != res_removals.end();
                if (rm) delete r;
                return rm;
            });
        _residues.erase(new_end, _residues.end());
    }
    // since the Bond destructor uses info (namely structure()) from its Atoms,
    // delete the bonds first (not willing to add a Structure pointer [and its
    // memory use] to Bond at this point)
    std::set<Bond*> del_bonds;
    std::set<Atom*> bond_losers;
    for (auto a: atoms) {
        if (a->_rings.size() > 0)
            _recompute_rings = true;
        for (auto b: a->bonds()) {
            del_bonds.insert(b);
            auto oa = b->other_atom(a);
            if (atoms.find(oa) == atoms.end()) {
                oa->remove_bond(b);
                bond_losers.insert(oa);
            }
        }
    }
    for (auto b: del_bonds)
        delete b;

    // remove_if doesn't swap the removed items into the end of the vector,
    // so can't just go through the tail of the vector and delete things,
    // need to delete them as part of the lambda
    auto new_a_end = std::remove_if(_atoms.begin(), _atoms.end(),
        [&atoms](Atom* a) { 
            bool rm = atoms.find(a) != atoms.end();
            if (rm) delete a;
            return rm;
        });
    _atoms.erase(new_a_end, _atoms.end());

    auto new_b_end = std::remove_if(_bonds.begin(), _bonds.end(),
        [&del_bonds](Bond* b) {
            return del_bonds.find(b) != del_bonds.end();
        });
    _bonds.erase(new_b_end, _bonds.end());
    
    _get_interres_connectivity(end_ri_lookup, end_ir_lookup, end_res_connects_to_next,
        end_left_missing_structure_atoms, end_right_missing_structure_atoms, &atoms);
    // for residues that don't connect to the next now but did before,
    // may have to create missing-structure pseudobond
    for (auto r: _residues) {
        if (end_res_connects_to_next[r] || !begin_res_connects_to_next[r])
            continue;
        Residue* end_next = end_ir_lookup[end_ri_lookup[r]+1];
        // before the deletion were these residues connected?
        bool connected = true;
        int cur_i = begin_ri_lookup[r]+1;
        Residue* cur_r = begin_ir_lookup[cur_i];
        while (cur_r != end_next) {
            if (!begin_res_connects_to_next[cur_r]) {
                connected = false;
                break;
            }
            cur_r = begin_ir_lookup[++cur_i];
        }
        if (!connected)
            continue;
        auto pbg = _pb_mgr.get_group(PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NORMAL);
        pbg->new_pseudobond(
            _find_attachment_atom(r, atoms, bond_losers,
                begin_left_missing_structure_atoms, end_left_missing_structure_atoms, true),
            _find_attachment_atom(end_next, atoms, bond_losers,
                begin_right_missing_structure_atoms, end_right_missing_structure_atoms, false)
        );
    }

    set_gc_shape();
    set_gc_adddel();
    _idatm_valid = false;
}

void
Structure::_form_chain_check(Atom* a1, Atom* a2, Bond* b)
{
    // If initial construction is over (i.e. Python instance exists) and make_chains()
    // has been called (i.e. _chains is not null), then need to check if new bond
    // or missing-structure pseudobond creates a chain or coalesces chain fragments
    if (_chains == nullptr)
        return;
    auto inst = py_instance(false);
    Py_DECREF(inst);
    if (inst == Py_None)
        return;
    Residue* start_r;
    Residue* other_r;
    bool is_pb = (b == nullptr);
    if (is_pb) {
        // missing structure pseudobond; need to pass through residue list to determine
        // relative ordering of the residues
        for (auto r: residues()) {
            if (r == a1->residue()) {
                start_r = a1->residue();
                other_r = a2->residue();
                break;
            }
            if (r == a2->residue()) {
                start_r = a2->residue();
                other_r = a1->residue();
                break;
            }
        }
    } else {
        auto start_a = b->polymeric_start_atom();
        if (start_a == nullptr)
            return;
        start_r = start_a->residue();
        other_r = b->other_atom(start_a)->residue();
    }
    if (start_r->chain() == nullptr) {
        if (other_r->chain() == nullptr) {
            // form a new chain based on start residue's chain ID
            auto chain = _new_chain(start_r->chain_id(), Sequence::rname_polymer_type(start_r->name()));
            chain->push_back(start_r);
            chain->push_back(other_r);
        } else {
            // incorporate start_r into other_r's chain
            auto other_chain = other_r->chain();
            auto other_index = other_chain->res_map().at(other_r);
            if (other_index == 0 || other_chain->residues()[other_index-1] != nullptr) {
                if (other_index == 0)
                    other_chain->push_front(start_r);
                else
                    other_chain->insert(other_r, start_r);
            } else {
                other_chain->_residues[other_index-1] = start_r;
                other_chain->_res_map[start_r] = other_index-1;
                start_r->set_chain(other_chain);
                change_tracker()->add_modified(this, other_chain, ChangeTracker::REASON_RESIDUES);
                auto old_char = other_chain->contents()[other_index-1];
                auto new_char = Sequence::rname3to1(start_r->name());
                if (old_char != new_char) {
                    other_chain->at(other_index-1) = new_char;
                }
            }
        }
    } else {
        if (other_r->chain() == nullptr) {
            // incorporate other_r into start_r's chain
            auto start_chain = start_r->chain();
            auto start_index = start_chain->res_map().at(start_r);
            if (start_index == start_chain->size() - 1
            || start_chain->residues()[start_index+1] != nullptr) {
                if (start_index == start_chain->size() - 1)
                    start_chain->push_back(other_r);
                else
                    start_chain->insert(start_chain->residues()[start_index+1], other_r);
            } else {
                start_chain->_residues[start_index+1] = other_r;
                start_chain->_res_map[other_r] = start_index+1;
                other_r->set_chain(start_chain);
                change_tracker()->add_modified(this, start_chain, ChangeTracker::REASON_RESIDUES);
                auto old_char = start_chain->contents()[start_index+1];
                auto new_char = Sequence::rname3to1(other_r->name());
                if (old_char != new_char) {
                    start_chain->at(start_index+1) = new_char;
                }
            }
        } else if (start_r->chain() != other_r->chain()) {
            // merge other_r's chain into start_r's chain
            // and demote other_r's chain to a plain sequence
            *start_r->chain() += *other_r->chain();
        } else if (!is_pb) {
            // check if there were missing residues at that sequence position and eliminate any
            auto chain = start_r->chain();
            auto start_index = chain->res_map().at(start_r);
            if (chain->residues()[start_index+1] == nullptr) {
                auto& contents = chain->contents();
                Sequence::Contents new_chars(contents.begin(), contents.begin() + start_index);
                auto other_index = chain->res_map().at(other_r);
                new_chars.insert(new_chars.end(), contents.begin() + other_index, contents.end());
                auto& residues = chain->residues();
                StructureSeq::Residues new_residues(residues.begin(), residues.begin() + start_index);
                new_residues.insert(new_residues.end(), residues.begin() + other_index, residues.end());
                chain->bulk_set(new_residues, &new_chars);
            }
        }
    }
    set_gc_ribbon();
}

void
Structure::_get_interres_connectivity(std::map<Residue*, int>& res_lookup,
    std::map<int, Residue*>& index_lookup,
    std::map<Residue*, bool>& res_connects_to_next,
    std::set<Atom*>& left_missing_structure_atoms,
    std::set<Atom*>& right_missing_structure_atoms,
    const std::set<Atom*>* deleted_atoms) const
{
    int i = 0;
    Residue *prev_r = nullptr;
    for (auto r: _residues) {
        res_lookup[r] = i;
        index_lookup[i++] = r;
        if (prev_r != nullptr) {
            res_connects_to_next[prev_r] = prev_r->connects_to(r);
        }
        prev_r = r;
    }
    res_connects_to_next[prev_r] = false;

    // go through missing-structure pseudobonds
    auto pbg = const_cast<Structure*>(this)->_pb_mgr.get_group(
        PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg != nullptr) {
        for (auto& pb: pbg->pseudobonds()) {
            Atom* a1 = pb->atoms()[0];
            Atom* a2 = pb->atoms()[1];
            // destruction batching hasn't run at this point,
            // so "dead" pseudobonds can still be present
            if (deleted_atoms != nullptr &&
            (deleted_atoms->find(a1) != deleted_atoms->end()
            || deleted_atoms->find(a2) != deleted_atoms->end()))
                continue;
            Residue *r1 = a1->residue();
            Residue *r2 = a2->residue();
            int i1 = res_lookup[r1];
            int i2 = res_lookup[r2];
            if (i1+1 == i2) {
                res_connects_to_next[r1] = true;
                left_missing_structure_atoms.insert(a1);
                right_missing_structure_atoms.insert(a2);
            } else if (i2+1 == i1) {
                res_connects_to_next[r2] = true;
                left_missing_structure_atoms.insert(a2);
                right_missing_structure_atoms.insert(a1);
            }
        }
    }
}

void
Structure::delete_atoms(const std::vector<Atom*>& atoms)
{
    auto db = DestructionBatcher(this);
    // construct set first to ensure uniqueness before tests...
    auto del_atoms_set = std::set<Atom*>(atoms.begin(), atoms.end());
    _delete_atoms(del_atoms_set);
}

void
Structure::delete_bond(Bond *b)
{
    typename Bonds::iterator i = std::find_if(_bonds.begin(), _bonds.end(),
        [&b](Bond* ub) { return ub == b; });
    if (i == _bonds.end())
        throw std::invalid_argument("delete_bond called for Bond not in Structure");
    auto db = DestructionBatcher(this);
    // for backbone bonds, create missing-structure pseudobonds
    // if the criteria for adding the pseudobond is changed, the code in pdb_lib/connect_cpp/connect.cpp
    // in find_missing_structure_bonds for deleting bonds will have to be changed
    if (b->is_backbone()) {
        auto pbg = _pb_mgr.get_group(PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NORMAL);
        pbg->new_pseudobond(b->atoms()[0], b->atoms()[1]);
    }
    for (auto a: b->atoms())
        a->remove_bond(b);
    _bonds.erase(i);
    set_gc_shape();
    set_gc_adddel();
    _structure_cats_dirty = true;
    _idatm_valid = false;
    if (b->_rings.size() > 0)
        _recompute_rings = true;
    delete b;
}

void
Structure::_delete_residue(Residue* r)
{
    auto del_atoms_set = std::set<Atom*>(r->atoms().begin(), r->atoms().end());
    _delete_atoms(del_atoms_set, false);
}

void
Structure::delete_residue(Residue* r)
{
    auto ri = std::find(_residues.begin(), _residues.end(), r);
    if (ri == _residues.end()) {
        logger::error(_logger, "Residue ", r->str(),
            " does not belong to the structure that it's being deleted from.");
        return;
    }
    if (residues().size() == 1) {
        delete this;
        return;
    }
    _delete_residue(r);
}

CoordSet *
Structure::find_coord_set(int id) const
{
    for (auto csi = _coord_sets.begin(); csi != _coord_sets.end(); ++csi) {
        if ((*csi)->id() == id)
            return *csi;
    }

    return nullptr;
}

Residue *
Structure::find_residue(const ChainID &chain_id, int num, char insert) const
{
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = *ri;
        if (r->number() == num && r->chain_id() == chain_id
        && r->insertion_code() == insert)
            return r;
    }
    return nullptr;
}

Residue *
Structure::find_residue(const ChainID& chain_id, int num, char insert, ResName& name) const
{
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = *ri;
        if (r->number() == num && r->name() == name && r->chain_id() == chain_id
        && r->insertion_code() == insert)
            return r;
    }
    return nullptr;
}

void
Structure::make_chains() const
{
    // since Graphs don't have sequences, they don't have chains
    if (_chains != nullptr) {
        for (auto c: *_chains)
            delete c;
        delete _chains;
    }

    _chains = new Chains();
}

Atom *
Structure::new_atom(const char* name, const Element& e)
{
    Atom *a = new Atom(this, name, e);
    add_atom(a);
    if (e.number() == 1)
        ++_num_hyds;
    _idatm_valid = false;
    return a;
}

static bool
backbone_increases(Atom* a1, Atom* a2, const std::vector<AtomName>* names)
{
    if (a1->residue() == a2->residue())
        return std::find(names->begin(), names->end(), a1->name()) <
                std::find(names->begin(), names->end(), a2->name());
    const auto& residues = a1->structure()->residues();
    return std::find(residues.begin(), residues.end(), a1->residue()) <
            std::find(residues.begin(), residues.end(), a2->residue());
}

Atom*
follow_backbone(Atom* bb1, Atom* bb2, Atom* goal, std::set<Atom*>& seen)
{
    for (auto nb: bb2->neighbors()) {
        if (!nb->is_backbone(BBE_MIN))
            continue;
        if (nb == bb1)
            continue;
        if (nb == goal)
            return nullptr;
        if (seen.find(nb) != seen.end())
            // backbone now loops back on itself!
            return nullptr;
        seen.insert(bb1);
        return follow_backbone(bb2, nb, goal, seen);
    }
    return bb2;
}

Bond *
Structure::_new_bond(Atom *a1, Atom *a2, bool bond_only)
{
    Bond *b = new Bond(this, a1, a2, bond_only);
    b->finish_construction(); // virtual calls work now
    add_bond(b);
    if (bond_only)
        return b;
    _idatm_valid = false;
    auto inst = py_instance(false);
    Py_DECREF(inst);
    if (inst != Py_None) {
        // chain adjustment now handled by _form_chain_check
        auto pbg = _pb_mgr.get_group(PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
        if (pbg != nullptr) {
            // possibly adjust missing-chain pseudobonds
            if (a1->is_backbone(BBE_MIN) && a2->is_backbone(BBE_MIN)) {
                for (auto pb: pbg->pseudobonds()) {
                    auto pb_a1 = pb->atoms()[0];
                    auto pb_a2 = pb->atoms()[1];
                    Atom* match_a1 = a1 == pb_a1 ? pb_a1 : (a1 == pb_a2 ? pb_a2 : nullptr);
                    Atom* match_a2 = a2 == pb_a1 ? pb_a1 : (a2 == pb_a2 ? pb_a2 : nullptr);
                    Atom* matched_a;
                    Atom* other_a;
                    if (match_a1 == nullptr) {
                        if (match_a2 == nullptr)
                            continue;
                        matched_a = a2;
                        other_a = a1;
                    } else {
                        if (match_a2 == nullptr) {
                            matched_a = a1;
                            other_a = a2;
                        } else {
                            pbg->delete_pseudobond(pb);
                            if (pbg->pseudobonds().size() == 0)
                                //pbg->manager()->delete_group(pbg);
                                Py_XDECREF(pbg->py_call_method("cpp_del_model"));
                            break;
                        }
                    }
                    // is the unmatched pb atom in "the direction of" the other "a" atom?
                    auto other_pb = pb->other_atom(matched_a);
                    auto ordered_names = matched_a->residue()->ordered_min_backbone_atom_names();
                    if (ordered_names == nullptr)
                        continue;
                    bool a_other_increases = backbone_increases(matched_a, other_a, ordered_names);
                    bool pb_other_increases = backbone_increases(matched_a, other_pb, ordered_names);
                    if (a_other_increases == pb_other_increases) {
                        // Okay, the new bond points into the structure gap;
                        // follow along the new backbone and shorten or eliminate the pseudobond
                        std::set<Atom*> seen;
                        auto new_end = follow_backbone(matched_a, other_a, other_pb, seen);
                        if (new_end == nullptr) {
                            pbg->delete_pseudobond(pb);
                            if (pbg->pseudobonds().size() == 0)
                                //pbg->manager()->delete_group(pbg);
                                Py_XDECREF(pbg->py_call_method("cpp_del_model"));
                        } else {
                            auto shortened = pbg->new_pseudobond(new_end, other_pb);
                            shortened->copy_style(pb);
                            pbg->delete_pseudobond(pb);
                        }
                        break;
                    }
                }
            }
        }
    }
    return b;
}

CoordSet *
Structure::new_coord_set()
{
    if (_coord_sets.empty())
        return new_coord_set(1);
    return new_coord_set(_coord_sets.back()->id()+1);
}

static void
_coord_set_insert(Structure::CoordSets &coord_sets, CoordSet* cs, int index)
{
    if (coord_sets.empty() || coord_sets.back()->id() < index) {
        coord_sets.emplace_back(cs);
        return;
    }
    for (auto csi = coord_sets.begin(); csi != coord_sets.end(); ++csi) {
        if (index < (*csi)->id()) {
            coord_sets.insert(csi, cs);
            return;
        } else if (index == (*csi)->id()) {
            auto pos = csi - coord_sets.begin();
            delete *csi;
            coord_sets[pos] = cs;
            return;
        }
    }
    std::logic_error("CoordSet insertion logic error");
}

CoordSet*
Structure::new_coord_set(int index)
{
    if (!_coord_sets.empty())
        return new_coord_set(index, _coord_sets.back()->coords().size());
    CoordSet* cs = new CoordSet(this, index);
    _coord_set_insert(_coord_sets, cs, index);
    return cs;
}

CoordSet*
Structure::new_coord_set(int index, int size)
{
    CoordSet* cs = new CoordSet(this, index, size);
    _coord_set_insert(_coord_sets, cs, index);
    return cs;
}

Residue*
Structure::new_residue(const ResName& name, const ChainID& chain,
    int pos, char insert, Residue *neighbor, bool after)
{
    if (chain.size() == 0)
        throw std::invalid_argument("Chain ID cannot be the empty string");
    if (neighbor == nullptr) {
        _residues.emplace_back(new Residue(this, name, chain, pos, insert));
        return _residues.back();
    }
    auto ri = std::find_if(_residues.begin(), _residues.end(),
                [&neighbor](Residue* vr) { return vr == neighbor; });
    if (ri == _residues.end())
        throw std::out_of_range("Waypoint residue not in residue list");
    if (after)
        ++ri;
    Residue *r = new Residue(this, name, chain, pos, insert);
    _residues.insert(ri, r);
    return r;
}

std::set<ResName>
Structure::nonstd_res_names() const
{
    std::set<ResName> nonstd;
    for (auto r: residues()) {
        if (Sequence::rname3to1(r->name()) == 'X'
        && Residue::std_solvent_names.find(r->name()) == Residue::std_solvent_names.end())
            nonstd.insert(r->name());
    }
    return nonstd;
}

void
Structure::renumber_residues(const std::vector<Residue*>& res_list, int start)
{
    if (res_list.size() == 0)
        return;
    std::set<Residue*> check(res_list.begin(), res_list.end());
    auto chain_id = res_list[0]->chain_id();
    for (auto r: res_list)
        if (r->chain_id() != chain_id)
            throw std::logic_error("Renumbered residues must all have same chain ID");
    for (auto r: residues()) {
        if (r->chain_id() != chain_id)
            continue;
        if (check.find(r) != check.end())
            continue;
        if (r->insertion_code() != ' ')
            // renumbered residues will have no insertion code
            continue;
        if (r->number() >= start && r->number() < start + static_cast<int>(res_list.size()))
            throw std::logic_error("Renumbering residies will conflict with other existing residues");
    }
    int num = start;
    for (auto r: res_list) {
        r->set_insertion_code(' ');
        r->set_number(num++);
    }
}

void
Structure::reorder_residues(const Structure::Residues& new_order)
{
    if (new_order.size() != _residues.size())
        throw std::invalid_argument("New residue order not same length as old order");
    std::set<Residue*> seen;
    for (auto r: new_order) {
        if (seen.find(r) != seen.end())
            throw std::invalid_argument("Duplicate residue in new residue order");
        seen.insert(r);
        if (r->structure() != this)
            throw std::invalid_argument("Residue not belonging to this structure"
                " in new residue order");
    }
    _residues = new_order;
}

const Structure::Rings&
Structure::rings(bool cross_residues, unsigned int all_size_threshold,
    std::set<const Residue *>* ignore) const
{
    if (_rings_cached(cross_residues, all_size_threshold, ignore)) {
        return _rings;
    }

    auto db = DestructionBatcher(const_cast<Structure*>(this));
    _recompute_rings = false;
    _rings_last_cross_residues = cross_residues;
    _rings_last_all_size_threshold = all_size_threshold;
    _rings_last_ignore = ignore;

    _calculate_rings(cross_residues, all_size_threshold, ignore);

    // clear out ring lists in individual atoms and bonds
    for (auto a: atoms()) {
        a->_rings.clear();
    }
    for (auto b: bonds()) {
        b->_rings.clear();
    }

    // set individual atom/bond ring lists
    for (auto& r: _rings) {
        for (auto a: r.atoms()) {
            a->_rings.push_back(&r);
        }
        for (auto b: r.bonds()) {
            b->_rings.push_back(&r);
        }
    }
    return _rings;
}

bool
Structure::_rings_cached(bool cross_residues, unsigned int all_size_threshold,
    std::set<const Residue *>* ignore) const
{
    return !_recompute_rings && cross_residues == _rings_last_cross_residues
        && all_size_threshold == _rings_last_all_size_threshold
        && ignore == _rings_last_ignore;
}

int
Structure::session_info(PyObject* ints, PyObject* floats, PyObject* misc) const
{
    // The passed-in args need to be empty lists.  This routine will add one object to each
    // list for each of these classes:
    //    AtomicStructure/Structure
    //    Atom
    //    Bond (needs Atoms)
    //    CoordSet (needs Atoms)
    //    PseudobondManager (needs Atoms and CoordSets)
    //    Residue
    //    Chain
    // For the numeric types, the objects will be numpy arrays: one-dimensional for
    // AtomicStructure attributes and two-dimensional for the others.  Except for
    // PseudobondManager; that will be a list of numpy arrays, one per group.  For the misc,
    // The objects will be Python lists, or lists of lists (same scheme as for the arrays),
    // though there may be exceptions (e.g. altloc info).
    //
    // Just let rings get recomputed instead of saving them.  Don't have to set up and
    // tear down a bond map that way (rings are the only thing that needs bond references).

    if (!PyList_Check(ints) || PyList_Size(ints) != 0)
        throw std::invalid_argument("AtomicStructure::session_info: first arg is not an"
            " empty list");
    if (!PyList_Check(floats) || PyList_Size(floats) != 0)
        throw std::invalid_argument("AtomicStructure::session_info: second arg is not an"
            " empty list");
    if (!PyList_Check(misc) || PyList_Size(misc) != 0)
        throw std::invalid_argument("AtomicStructure::session_info: third arg is not an"
            " empty list");

    using pysupport::cchar_to_pystring;
    using pysupport::cvec_of_char_to_pylist;
    using pysupport::cmap_of_chars_to_pydict;

    // AtomicStructure attrs
    int* int_array;
    PyObject* npy_array = python_int_array(SESSION_NUM_INTS(), &int_array);
    *int_array++ = _idatm_valid;
    int x = std::find(_coord_sets.begin(), _coord_sets.end(), _active_coord_set)
        - _coord_sets.begin();
    *int_array++ = x; // can be == size if active coord set is null
    *int_array++ = asterisks_translated;
    *int_array++ = _display;
    *int_array++ = is_traj;
    *int_array++ = lower_case_chains;
    *int_array++ = pdb_version;
    *int_array++ = _ribbon_display_count;
    *int_array++ = _ss_assigned;
    *int_array++ = _ribbon_orientation;
    *int_array++ = _ribbon_show_spine;
    *int_array++ = _ribbon_tether_shape;
    *int_array++ = _ribbon_tether_sides;
    *int_array++ = _ribbon_mode_helix;
    *int_array++ = _ribbon_mode_strand;
    *int_array++ = ss_ids_normalized;
    *int_array++ = _ring_display_count;
    // pb manager version number remembered later
    if (PyList_Append(ints, npy_array) < 0)
        throw std::runtime_error("Couldn't append to int list");

    float* float_array;
    npy_array = python_float_array(SESSION_NUM_FLOATS(), &float_array);
    *float_array++ = _ball_scale;
    *float_array++ = _ribbon_tether_scale;
    *float_array++ = _ribbon_tether_opacity;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            *float_array++ = _position[i][j];
    if (PyList_Append(floats, npy_array) < 0)
        throw std::runtime_error("Couldn't append to floats list");

    PyObject* attr_list = PyList_New(SESSION_NUM_MISC());
    if (attr_list == nullptr)
        throw std::runtime_error("Cannot create Python list for misc info");
    if (PyList_Append(misc, attr_list) < 0)
        throw std::runtime_error("Couldn't append to misc list");
    // input_seq_info
    PyList_SET_ITEM(attr_list, 0, cmap_of_chars_to_pydict(_input_seq_info,
        "residue chain ID", "residue name"));
    // input_seq_source
    PyList_SET_ITEM(attr_list, 1, cchar_to_pystring(input_seq_source, "seq info source"));
    // metadata
    PyList_SET_ITEM(attr_list, 2, cmap_of_chars_to_pydict(metadata,
        "metadata key", "metadata value"));

    // atoms
    // We need to remember names and elements ourself for constructing the atoms.
    // Make a list of num_atom+1 items, the first of which will be the list of
    //   names and the remainder of which will be empty lists which will be handed
    //   off individually to the atoms.
    int num_atoms = atoms().size();
    int num_ints = num_atoms; // list of elements
    int num_floats = 0;
    PyObject* atoms_misc = PyList_New(num_atoms+1);
    if (atoms_misc == nullptr)
        throw std::runtime_error("Cannot create Python list for atom misc info");
    if (PyList_Append(misc, atoms_misc) < 0)
        throw std::runtime_error("Couldn't append atom misc list to misc list");
    PyObject* atom_names = PyList_New(num_atoms);
    if (atom_names == nullptr)
        throw std::runtime_error("Cannot create Python list for atom names");
    PyList_SET_ITEM(atoms_misc, 0, atom_names);
    int i = 0;
    for (auto a: atoms()) {
        num_ints += a->session_num_ints();
        num_floats += a->session_num_floats();

        // remember name
        PyList_SET_ITEM(atom_names, i++, cchar_to_pystring(a->name(), "atom name"));
    }
    int* atom_ints;
    PyObject* atom_npy_ints = python_int_array(num_ints, &atom_ints);
    for (auto a: atoms()) {
        *atom_ints++ = a->element().number();
    }
    if (PyList_Append(ints, atom_npy_ints) < 0)
        throw std::runtime_error("Couldn't append atom ints to int list");
    float* atom_floats;
    PyObject* atom_npy_floats = python_float_array(num_floats, &atom_floats);
    if (PyList_Append(floats, atom_npy_floats) < 0)
        throw std::runtime_error("Couldn't append atom floats to float list");
    i = 1;
    for (auto a: atoms()) {
        PyObject* empty_list = PyList_New(0);
        if (empty_list == nullptr)
            throw std::runtime_error("Cannot create Python list for individual atom misc info");
        PyList_SET_ITEM(atoms_misc, i++, empty_list);
        a->session_save(&atom_ints, &atom_floats, empty_list);
    }

    // bonds
    // We need to remember atom indices ourself for constructing the bonds.
    int num_bonds = bonds().size();
    num_ints = 1 + 2 * num_bonds; // to hold the # of bonds, and atom indices
    num_floats = 0;
    num_ints += num_bonds * Bond::session_num_ints();
    num_floats += num_bonds * Bond::session_num_floats();
    PyObject* bonds_misc = PyList_New(0);
    if (bonds_misc == nullptr)
        throw std::runtime_error("Cannot create Python list for bond misc info");
    if (PyList_Append(misc, bonds_misc) < 0)
        throw std::runtime_error("Couldn't append bond misc list to misc list");
    int* bond_ints;
    PyObject* bond_npy_ints = python_int_array(num_ints, &bond_ints);
    *bond_ints++ = num_bonds;
    for (auto b: bonds()) {
        *bond_ints++ = (*session_save_atoms)[b->atoms()[0]];
        *bond_ints++ = (*session_save_atoms)[b->atoms()[1]];
    }
    if (PyList_Append(ints, bond_npy_ints) < 0)
        throw std::runtime_error("Couldn't append bond ints to int list");
    float* bond_floats;
    PyObject* bond_npy_floats = python_float_array(num_floats, &bond_floats);
    if (PyList_Append(floats, bond_npy_floats) < 0)
        throw std::runtime_error("Couldn't append bond floats to float list");
    for (auto b: bonds()) {
        b->session_save(&bond_ints, &bond_floats);
    }

    // coord sets
    int num_cs = coord_sets().size();
    num_ints = 1 + num_cs; // to note the total # of coord sets, and coord set IDs
    num_floats = 0;
    for (auto cs: _coord_sets) {
        num_ints += cs->session_num_ints();
        num_floats += cs->session_num_floats();
    }
    PyObject* cs_misc = PyList_New(0);
    if (cs_misc == nullptr)
        throw std::runtime_error("Cannot create Python list for coord set misc info");
    if (PyList_Append(misc, cs_misc) < 0)
        throw std::runtime_error("Couldn't append coord set misc list to misc list");
    int* cs_ints;
    PyObject* cs_npy_ints = python_int_array(num_ints, &cs_ints);
    *cs_ints++ = num_cs;
    for (auto cs: coord_sets()) {
        *cs_ints++ = cs->id();
    }
    if (PyList_Append(ints, cs_npy_ints) < 0)
        throw std::runtime_error("Couldn't append coord set ints to int list");
    float* cs_floats;
    PyObject* cs_npy_floats = python_float_array(num_floats, &cs_floats);
    if (PyList_Append(floats, cs_npy_floats) < 0)
        throw std::runtime_error("Couldn't append coord set floats to float list");
    for (auto cs: coord_sets()) {
        cs->session_save(&cs_ints, &cs_floats);
    }

    // PseudobondManager groups;
    PyObject* pb_ints;
    PyObject* pb_floats;
    PyObject* pb_misc;
    // pb manager version now locked to main version number, so the next line
    // is really a historical artifact
    *int_array = _pb_mgr.session_info(&pb_ints, &pb_floats, &pb_misc);
    if (PyList_Append(ints, pb_ints) < 0)
        throw std::runtime_error("Couldn't append pseudobond ints to int list");
    if (PyList_Append(floats, pb_floats) < 0)
        throw std::runtime_error("Couldn't append pseudobond floats to float list");
    if (PyList_Append(misc, pb_misc) < 0)
        throw std::runtime_error("Couldn't append pseudobond misc info to misc list");

    // residues
    int num_residues = residues().size();
    num_ints = 2 * num_residues; // to note position and insertion code for constructor
    num_floats = 0;
    for (auto res: _residues) {
        num_ints += res->session_num_ints();
        num_floats += res->session_num_floats();
    }
    PyObject* res_misc = PyList_New(3);
    if (res_misc == nullptr)
        throw std::runtime_error("Cannot create Python list for residue misc info");
    if (PyList_Append(misc, res_misc) < 0)
        throw std::runtime_error("Couldn't append residue misc list to misc list");
    int* res_ints;
    PyObject* res_npy_ints = python_int_array(num_ints, &res_ints);
    if (PyList_Append(ints, res_npy_ints) < 0)
        throw std::runtime_error("Couldn't append residue ints to int list");
    float* res_floats;
    PyObject* res_npy_floats = python_float_array(num_floats, &res_floats);
    if (PyList_Append(floats, res_npy_floats) < 0)
        throw std::runtime_error("Couldn't append residue floats to float list");
    PyObject* py_res_names = PyList_New(num_residues);
    if (py_res_names == nullptr)
        throw std::runtime_error("Cannot create Python list for residue names");
    PyList_SET_ITEM(res_misc, 0, py_res_names);
    PyObject* py_chain_ids = PyList_New(num_residues);
    if (py_chain_ids == nullptr)
        throw std::runtime_error("Cannot create Python list for chain IDs");
    PyList_SET_ITEM(res_misc, 1, py_chain_ids);
    PyObject* py_mmcif_chain_ids = PyList_New(num_residues);
    if (py_mmcif_chain_ids == nullptr)
        throw std::runtime_error("Cannot create Python list for mmCIF chain IDs");
    PyList_SET_ITEM(res_misc, 2, py_mmcif_chain_ids);
    i = 0;
    for (auto res: residues()) {
        // remember res name and chain ID
        PyList_SET_ITEM(py_res_names, i, cchar_to_pystring(res->name(), "residue name"));
        PyList_SET_ITEM(py_chain_ids, i, cchar_to_pystring(res->chain_id(), "residue chain ID"));
        PyList_SET_ITEM(py_mmcif_chain_ids, i++,
            cchar_to_pystring(res->mmcif_chain_id(), "residue mmCIF chain ID"));
        *res_ints++ = res->number();
        *res_ints++ = res->insertion_code();
        res->session_save(&res_ints, &res_floats);
    }

    // chains
    int num_chains = _chains == nullptr ? -1 : _chains->size();
    num_ints = 1; // for storing num_chains, since len(chain_ids) can't show nullptr
    num_floats = 0;
    // allocate for list of chain IDs; descriptions
    PyObject* chain_misc = PyList_New(2);
    if (chain_misc == nullptr)
        throw std::runtime_error("Cannot create Python list for chain misc info");
    if (PyList_Append(misc, chain_misc) < 0)
        throw std::runtime_error("Couldn't append chain misc list to misc list");
    PyObject* chain_ids = PyList_New(num_chains);
    if (chain_ids == nullptr)
        throw std::runtime_error("Cannot create Python list for chain IDs");
    PyList_SET_ITEM(chain_misc, 0, chain_ids);
    PyObject* descriptions = PyList_New(num_chains);
    if (descriptions == nullptr)
        throw std::runtime_error("Cannot create Python list for chain descriptions");
    PyList_SET_ITEM(chain_misc, 1, descriptions);
    i = 0;
    if (_chains != nullptr) {
        for (auto ch: *_chains) {
            num_ints += ch->session_num_ints();
            num_floats += ch->session_num_floats();

            // remember chain ID, description
            PyList_SET_ITEM(chain_ids, i, cchar_to_pystring(ch->chain_id(), "chain chain ID"));
            PyList_SET_ITEM(descriptions, i++, cchar_to_pystring(ch->description(), "chain description"));
        }
    }
    int* chain_ints;
    PyObject* chain_npy_ints = python_int_array(num_ints, &chain_ints);
    if (PyList_Append(ints, chain_npy_ints) < 0)
        throw std::runtime_error("Couldn't append chain ints to int list");
    float* chain_floats;
    PyObject* chain_npy_floats = python_float_array(num_floats, &chain_floats);
    if (PyList_Append(floats, chain_npy_floats) < 0)
        throw std::runtime_error("Couldn't append chain floats to float list");
    *chain_ints++ = num_chains;
    if (_chains != nullptr) {
        for (auto ch: *_chains) {
            ch->session_save(&chain_ints, &chain_floats);
        }
    }

    return CURRENT_SESSION_VERSION;  // version number
}

void
Structure::session_restore(int version, PyObject* ints, PyObject* floats, PyObject* misc)
{
    // restore the stuff saved by session_info()

    if (version > CURRENT_SESSION_VERSION)
        throw std::invalid_argument("Don't know how to restore new session data; update your"
            " version of ChimeraX");

    if (!PyTuple_Check(ints) || PyTuple_Size(ints) != 7)
        throw std::invalid_argument("AtomicStructure::session_restore: first arg is not a"
            " 7-element tuple");
    if (!PyTuple_Check(floats) || PyTuple_Size(floats) != 7)
        throw std::invalid_argument("AtomicStructure::session_restore: second arg is not a"
            " 7-element tuple");
    if (!PyTuple_Check(misc) || PyTuple_Size(misc) != 7)
        throw std::invalid_argument("AtomicStructure::session_restore: third arg is not a"
            " 7-element tuple");

    using pysupport::pysequence_of_string_to_cvec;
    using pysupport::pystring_to_cchar;

    // AtomicStructure ints
    PyObject* item = PyTuple_GET_ITEM(ints, 0);
    auto iarray = Numeric_Array();
    if (!array_from_python(item, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("AtomicStructure int data is not a one-dimensional"
            " numpy int array");
    if (iarray.size() != SESSION_NUM_INTS(version))
        throw std::invalid_argument("AtomicStructure int array wrong size");
    int* int_array = static_cast<int*>(iarray.values());
    _idatm_valid = *int_array++;
    int active_cs = *int_array++; // have to wait until CoordSets restored to set
    asterisks_translated = *int_array++;
    _display = *int_array++;
    is_traj = *int_array++;
    lower_case_chains = *int_array++;
    pdb_version = *int_array++;
    _ribbon_display_count = *int_array++;
    if (version == 1)
        _ss_assigned = true;
    else
        _ss_assigned = *int_array++;
    if (version >= 5) {
        _ribbon_orientation = static_cast<RibbonOrientation>(*int_array++);
        _ribbon_show_spine = *int_array++;
        _ribbon_tether_shape = static_cast<TetherShape>(*int_array++);
        _ribbon_tether_sides = *int_array++;
        _ribbon_mode_helix = static_cast<RibbonMode>(*int_array++);
        _ribbon_mode_strand = static_cast<RibbonMode>(*int_array++);
    }
    if (version >= 12)
        ss_ids_normalized = *int_array++;
    if (version >= 16)
        _ring_display_count = *int_array++;
    auto pb_manager_version = *int_array++;
    // if more added, change the array dimension check above

    // AtomicStructure floats
    item = PyTuple_GET_ITEM(floats, 0);
    auto farray = Numeric_Array();
    if (!array_from_python(item, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("AtomicStructure float data is not a one-dimensional"
            " numpy float array");
    if (farray.size() != SESSION_NUM_FLOATS(version))
        throw std::invalid_argument("AtomicStructure float array wrong size");
    float* float_array = static_cast<float*>(farray.values());
    _ball_scale = *float_array++;
    if (version >= 5) {
        _ribbon_tether_scale = *float_array++;
        _ribbon_tether_opacity = *float_array++;
    }
    if (version >= 12) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                _position[i][j] = *float_array++;
    }
    // if more added, change the array dimension check above

    // AtomicStructure misc info
    item = PyTuple_GET_ITEM(misc, 0);
    if (!(PyTuple_Check(item) || PyList_Check(item)) || PySequence_Fast_GET_SIZE(item) != SESSION_NUM_MISC(version))
        throw std::invalid_argument("AtomicStructure misc data is not a tuple or is wrong size");
    // input_seq_info
    PyObject* map = PySequence_Fast_GET_ITEM(item, 0);
    if (!PyDict_Check(map))
        throw std::invalid_argument("input seq info is not a dict!");
    Py_ssize_t index = 0;
    PyObject* py_chain_id;
    PyObject* py_residues;
    _input_seq_info.clear();
    while (PyDict_Next(map, &index, &py_chain_id, &py_residues)) {
        ChainID chain_id = pystring_to_cchar(py_chain_id, "input seq chain ID");
        auto& res_names = _input_seq_info[chain_id];
        pysequence_of_string_to_cvec(py_residues, res_names, "chain residue name");
    }
    int list_index = 1;
    if (version < 8) {
        // was name
        list_index++;
    }
    // input_seq_source
    input_seq_source = pystring_to_cchar(PySequence_Fast_GET_ITEM(item, list_index++),
        "structure input seq source");
    // metadata
    map = PySequence_Fast_GET_ITEM(item, list_index);
    if (!PyDict_Check(map))
        throw std::invalid_argument("structure metadata is not a dict!");
    index = 0;
    PyObject* py_hdr_type;
    PyObject* py_headers;
    _input_seq_info.clear();
    while (PyDict_Next(map, &index, &py_hdr_type, &py_headers)) {
        auto hdr_type = pystring_to_cchar(py_hdr_type, "structure metadata key");
        auto& headers = metadata[hdr_type];
        pysequence_of_string_to_cvec(py_headers, headers, "structure metadata");
    }

    // atoms
    PyObject* atoms_misc = PyTuple_GET_ITEM(misc, 1);
    if (!(PyTuple_Check(atoms_misc) || PyList_Check(atoms_misc)))
        throw std::invalid_argument("atom misc info is not a tuple");
    if (PySequence_Fast_GET_SIZE(atoms_misc) < 1)
        throw std::invalid_argument("atom names missing");
    std::vector<AtomName> atom_names;
    pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(atoms_misc, 0), atom_names, "atom name");
    if ((decltype(atom_names)::size_type)(PySequence_Fast_GET_SIZE(atoms_misc)) != atom_names.size() + 1)
        throw std::invalid_argument("bad atom misc info");
    PyObject* atom_ints = PyTuple_GET_ITEM(ints, 1);
    iarray = Numeric_Array();
    if (!array_from_python(atom_ints, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("Atom int data is not a one-dimensional"
            " numpy int array");
    int_array = static_cast<int*>(iarray.values());
    auto element_ints = int_array;
    int_array += atom_names.size();
    PyObject* atom_floats = PyTuple_GET_ITEM(floats, 1);
    farray = Numeric_Array();
    if (!array_from_python(atom_floats, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("Atom float data is not a one-dimensional"
            " numpy float array");
    float_array = static_cast<float*>(farray.values());
    int i = 1; // atom names are in slot zero
    for (auto aname: atom_names) {
        auto a = new_atom(aname.c_str(), Element::get_element(*element_ints++));
        a->session_restore(version, &int_array, &float_array, PySequence_Fast_GET_ITEM(atoms_misc, i++));
    }

    // bonds
    PyObject* bond_ints = PyTuple_GET_ITEM(ints, 2);
    iarray = Numeric_Array();
    if (!array_from_python(bond_ints, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("Bond int data is not a one-dimensional"
            " numpy int array");
    int_array = static_cast<int*>(iarray.values());
    auto num_bonds = *int_array++;
    auto bond_index_ints = int_array;
    int_array += 2 * num_bonds;
    PyObject* bond_floats = PyTuple_GET_ITEM(floats, 2);
    farray = Numeric_Array();
    if (!array_from_python(bond_floats, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("Bond float data is not a one-dimensional"
            " numpy float array");
    float_array = static_cast<float*>(farray.values());
    for (i = 0; i < num_bonds; ++i) {
        Atom *a1 = atoms()[*bond_index_ints++];
        Atom *a2 = atoms()[*bond_index_ints++];
        auto b = new_bond(a1, a2);
        b->session_restore(version, &int_array, &float_array);
    }

    // coord sets
    PyObject* cs_ints = PyTuple_GET_ITEM(ints, 3);
    iarray = Numeric_Array();
    if (!array_from_python(cs_ints, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("Coord set int data is not a one-dimensional"
            " numpy int array");
    int_array = static_cast<int*>(iarray.values());
    auto num_cs = *int_array++;
    auto cs_id_ints = int_array;
    int_array += num_cs;
    PyObject* cs_floats = PyTuple_GET_ITEM(floats, 3);
    farray = Numeric_Array();
    if (!array_from_python(cs_floats, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("Coord set float data is not a one-dimensional"
            " numpy float array");
    float_array = static_cast<float*>(farray.values());
    for (i = 0; i < num_cs; ++i) {
        auto cs = new_coord_set(*cs_id_ints++, atom_names.size());
        cs->session_restore(version, &int_array, &float_array);
    }
    // can now resolve the active coord set
    if ((CoordSets::size_type)active_cs < _coord_sets.size())
        _active_coord_set = _coord_sets[active_cs];
    else
        _active_coord_set = nullptr;

    // PseudobondManager groups;
    PyObject* pb_ints = PyTuple_GET_ITEM(ints, 4);
    iarray = Numeric_Array();
    if (!array_from_python(pb_ints, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("Pseudobond int data is not a one-dimensional"
            " numpy int array");
    int_array = static_cast<int*>(iarray.values());
    PyObject* pb_floats = PyTuple_GET_ITEM(floats, 4);
    farray = Numeric_Array();
    if (!array_from_python(pb_floats, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("Pseudobond float data is not a one-dimensional"
            " numpy float array");
    float_array = static_cast<float*>(farray.values());
    _pb_mgr.session_restore(pb_manager_version, &int_array, &float_array, PyTuple_GET_ITEM(misc, 4));

    // residues
    PyObject* res_misc = PyTuple_GET_ITEM(misc, 5);
    if (version < 4) {
        if (!(PyTuple_Check(res_misc) || PyList_Check(res_misc)) || PySequence_Fast_GET_SIZE(res_misc) != 2)
            throw std::invalid_argument("residue misc info is not a two-item tuple");
    } else {
        if (!(PyTuple_Check(res_misc) || PyList_Check(res_misc)) || PySequence_Fast_GET_SIZE(res_misc) != 3)
            throw std::invalid_argument("residue misc info is not a three-item tuple");
    }
    std::vector<ResName> res_names;
    pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(res_misc, 0), res_names, "residue name");
    std::vector<ChainID> res_chain_ids;
    pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(res_misc, 1), res_chain_ids, "chain ID");
    std::vector<ChainID> res_mmcif_chain_ids;
    if (version >= 4) {
        pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(res_misc, 2),
            res_mmcif_chain_ids, "mmCIF chain ID");
    }
    PyObject* py_res_ints = PyTuple_GET_ITEM(ints, 5);
    iarray = Numeric_Array();
    if (!array_from_python(py_res_ints, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("Residue int data is not a one-dimensional numpy int array");
    auto res_ints = static_cast<int*>(iarray.values());
    PyObject* py_res_floats = PyTuple_GET_ITEM(floats, 5);
    farray = Numeric_Array();
    if (!array_from_python(py_res_floats, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("Residue float data is not a one-dimensional"
            " numpy float array");
    auto res_floats = static_cast<float*>(farray.values());
    for (decltype(res_names)::size_type i = 0; i < res_names.size(); ++i) {
        auto& res_name = res_names[i];
        auto& chain_id = res_chain_ids[i];
        auto pos = *res_ints++;
        auto insert = *res_ints++;
        auto r = new_residue(res_name, chain_id, pos, insert);
        if (version >= 4)
            r->set_mmcif_chain_id(res_mmcif_chain_ids[i]);
        r->session_restore(version, &res_ints, &res_floats);
    }

    // chains
    PyObject* chain_misc = PyTuple_GET_ITEM(misc, 6);
    int misc_size = 1;
    if (version > 16)
        misc_size = 2;
    if (!(PyTuple_Check(chain_misc) || PyList_Check(chain_misc)) || PySequence_Fast_GET_SIZE(chain_misc) != misc_size)
        throw std::invalid_argument("chain misc info list is wrong size");
    std::vector<ChainID> chain_chain_ids;
    pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(chain_misc, 0), chain_chain_ids, "chain ID");
    std::vector<std::string> chain_descriptions;
    if (version > 16)
        pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(chain_misc, 1), chain_descriptions, "chain description");
    PyObject* py_chain_ints = PyTuple_GET_ITEM(ints, 6);
    iarray = Numeric_Array();
    if (!array_from_python(py_chain_ints, 1, Numeric_Array::Int, &iarray, false))
        throw std::invalid_argument("Chain int data is not a one-dimensional numpy int array");
    auto chain_ints = static_cast<int*>(iarray.values());
    PyObject* py_chain_floats = PyTuple_GET_ITEM(floats, 6);
    farray = Numeric_Array();
    if (!array_from_python(py_chain_floats, 1, Numeric_Array::Float, &farray, false))
        throw std::invalid_argument("Chain float data is not a one-dimensional"
            " numpy float array");
    auto chain_floats = static_cast<float*>(farray.values());
    auto num_chains = *chain_ints++;
    if (num_chains < 0) {
        _chains = nullptr;
    } else {
        _chains = new Chains();
        i = 0;
        for (auto chain_id: chain_chain_ids) {
            auto chain = _new_chain(chain_id);
            chain->session_restore(version, &chain_ints, &chain_floats);
            if (version > 16)
                chain->set_description(chain_descriptions[i++]);
        }
    }
}

void
Structure::session_save_setup() const
{
    size_t index = 0;

    session_save_atoms = new std::unordered_map<const Atom*, size_t>;
    for (auto a: atoms()) {
        (*session_save_atoms)[a] = index++;
    }

    index = 0;
    session_save_bonds = new std::unordered_map<const Bond*, size_t>;
    for (auto b: bonds()) {
        (*session_save_bonds)[b] = index++;
    }

    index = 0;
    session_save_chains = new std::unordered_map<const Chain*, size_t>;
    for (auto c: chains()) {
        (*session_save_chains)[c] = index++;
    }

    index = 0;
    session_save_crdsets = new std::unordered_map<const CoordSet*, size_t>;
    for (auto cs: coord_sets()) {
        (*session_save_crdsets)[cs] = index++;
    }

    index = 0;
    session_save_residues = new std::unordered_map<const Residue*, size_t>;
    for (auto r: residues()) {
        (*session_save_residues)[r] = index++;
    }

    _pb_mgr.session_save_setup();
}

void
Structure::session_save_teardown() const
{
    delete session_save_atoms;
    delete session_save_bonds;
    delete session_save_chains;
    delete session_save_crdsets;
    delete session_save_residues;

    _pb_mgr.session_save_teardown();
}

void
Structure::set_active_coord_set(CoordSet *cs)
{
    CoordSet *new_active;
    if (cs == nullptr) {
        if (_coord_sets.empty())
            return;
        new_active = _coord_sets.front();
    } else {
        CoordSets::iterator csi = std::find_if(_coord_sets.begin(), _coord_sets.end(),
                [&cs](CoordSet* vcs) { return vcs == cs; });
        if (csi == _coord_sets.end())
            throw std::out_of_range("Requested active coord set not in coord sets");
        new_active = cs;
    }
    if (_active_coord_set != new_active) {
        _active_coord_set = new_active;
        pb_mgr().change_cs(new_active);
        if (active_coord_set_change_notify()) {
            set_gc_shape();
            set_gc_ribbon();
            set_gc_ring();
            change_tracker()->add_modified(this, this, ChangeTracker::REASON_ACTIVE_COORD_SET);
            change_tracker()->add_modified(this, this, ChangeTracker::REASON_SCENE_COORD);
        }
    }
}

void
Structure::set_color(const Rgba& rgba)
{
    for (auto a: _atoms)
        a->set_color(rgba);
    for (auto b: _bonds)
        b->set_color(rgba);
    for (auto r: _residues) {
        r->set_ribbon_color(rgba);
        r->set_ring_color(rgba);
    }
}

static bool
compare_chains(Chain* c1, Chain* c2)
{
    Atom* a1 = nullptr;
    Atom* a2 = nullptr;
    for (auto r: c1->residues()) {
        if (r != nullptr) {
            a1 = r->atoms()[0];
            break;
        }
    }
    for (auto r: c2->residues()) {
        if (r != nullptr) {
            a2 = r->atoms()[0];
            break;
        }
    }
    if (a1 == nullptr || a2 == nullptr)
        return c1 < c2;
    return a1->serial_number() < a2->serial_number();
}

void
Structure::set_input_seq_info(const ChainID& chain_id, const std::vector<ResName>& res_names,
        const std::vector<Residue*>* correspondences, PolymerType pt, bool one_letter_names)
{
    _input_seq_info[chain_id] = res_names;
    if (correspondences != nullptr) {
        if (correspondences->size() != res_names.size())
            throw std::invalid_argument(
                "Sequence length differs from number of corresponding residues");
        if (pt == PT_NONE) {
            if (one_letter_names) {
                if (correspondences != nullptr) {
                    for (auto r: *correspondences) {
                        if (r == nullptr)
                            continue;
                        pt = Sequence::rname_polymer_type(r->name());
                        if (pt != PT_NONE)
                            break;
                    }
                }
            } else {
                for (auto rn: res_names) {
                    pt = Sequence::rname_polymer_type(rn);
                    if (pt != PT_NONE)
                        break;
                }
            }
            if (pt == PT_NONE)
                throw std::invalid_argument("Cannot determine polymer type of input sequence");
        }
        if (_chains == nullptr)
            _chains = new Chains();
        auto chain = _new_chain(chain_id, pt);
        auto res_chars = new Sequence::Contents(res_names.size());
        auto chars_ptr = res_chars->begin();
        for (auto rni = res_names.begin(); rni != res_names.end(); ++rni, ++chars_ptr) {
            if (one_letter_names)
                (*chars_ptr) = (*rni)[0];
            else
                (*chars_ptr) = Sequence::rname3to1(*rni);
        }
        chain->bulk_set(*correspondences, res_chars);
        chain->set_from_seqres(true);
        delete res_chars;
        std::sort(_chains->begin(), _chains->end(), compare_chains);
    }
}

void
Structure::set_position_matrix(double* pos)
{
    double *_pos = &_position[0][0];
    for (int i=0; i<12; ++i)
        *_pos++ = *pos++;
    change_tracker()->add_modified(this, this, ChangeTracker::REASON_SCENE_COORD);
}

void
Structure::use_best_alt_locs()
{
    std::map<Residue *, char> alt_loc_map = best_alt_locs();
    for (auto almi = alt_loc_map.begin(); almi != alt_loc_map.end(); ++almi) {
        (*almi).first->set_alt_loc((*almi).second);
    }
}

void
Structure::use_default_atom_radii()
{
    for (auto a: atoms()) a->use_default_radius();
}

int
Structure::get_all_graphics_changes() const
{
  int gc = get_graphics_changes();
  for (auto g: pb_mgr().group_map())
    gc |= g.second->get_graphics_changes();
  return gc;
}

void
Structure::set_all_graphics_changes(int changes)
{
  set_graphics_changes(changes);
  for (auto g: pb_mgr().group_map())
    g.second->set_graphics_changes(changes);
}

Structure::RibbonOrientation
Structure::ribbon_orient(const Residue *r) const
{
    if (r->polymer_type() == PT_NUCLEIC)
        return Structure::RIBBON_ORIENT_GUIDES;
    if (r->is_helix())
        return Structure::RIBBON_ORIENT_ATOMS;
    if (r->is_strand())
        return Structure::RIBBON_ORIENT_PEPTIDE;
    return Structure::RIBBON_ORIENT_ATOMS;
}

size_t Structure::num_ribbon_residues() const
{
    size_t count = 0;
    for (auto r: _residues)
        if (r->ribbon_display())
            count += 1;
    return count;
}

} //  namespace atomstruct
