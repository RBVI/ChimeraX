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
#include <set>

#include <logger/logger.h>
#include <pysupport/convert.h>
#include <arrays/pythonarray.h>

#include "Python.h"

#define ATOMSTRUCT_EXPORT
#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include "destruct.h"
#include "Structure.h"
#include "PBGroup.h"
#include "Pseudobond.h"
#include "Residue.h"

namespace {

class AcquireGIL {
    // RAII for Python GIL
    PyGILState_STATE gil_state;
public:
    AcquireGIL() {
        gil_state = PyGILState_Ensure();
    }
    ~AcquireGIL() {
        PyGILState_Release(gil_state);
    }
};

}

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
    lower_case_chains(false), pdb_version(0)
{
    change_tracker()->add_created(this);
}

Structure::~Structure() {
    // assign to variable so that it lives to end of destructor
    auto du = DestructionUser(this);
    change_tracker()->add_deleted(this);
    for (auto b: _bonds)
        delete b;
    for (auto a: _atoms)
        delete a;
    if (_chains != nullptr) {
        for (auto ch: *_chains)
            ch->clear_residues();
        // don't delete the actual chains -- they may be being
        // used as Sequences and the Python layer will delete 
        // them (as sequences) as appropriate
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
    }
}

void Structure::_copy(Structure* g) const
{
    for (auto h = metadata.begin() ; h != metadata.end() ; ++h)
        g->metadata[h->first] = h->second;
    g->pdb_version = pdb_version;
    g->lower_case_chains = lower_case_chains;
    g->set_ss_assigned(ss_assigned());
    g->set_ribbon_tether_scale(ribbon_tether_scale());
    g->set_ribbon_tether_shape(ribbon_tether_shape());
    g->set_ribbon_tether_sides(ribbon_tether_sides());
    g->set_ribbon_tether_opacity(ribbon_tether_opacity());
    g->set_ribbon_show_spine(ribbon_show_spine());
    g->set_ribbon_orientation(ribbon_orientation());
    g->set_ribbon_mode_helix(ribbon_mode_helix());
    g->set_ribbon_mode_strand(ribbon_mode_strand());

    std::map<Residue*, Residue*> rmap;
    for (auto ri = residues().begin() ; ri != residues().end() ; ++ri) {
        Residue* r = *ri;
        Residue* cr = g->new_residue(r->name(), r->chain_id(), r->number(), r->insertion_code());
        cr->set_mmcif_chain_id(r->mmcif_chain_id());
        cr->set_ribbon_display(r->ribbon_display());
        cr->set_ribbon_color(r->ribbon_color());
        cr->set_is_het(r->is_het());
        cr->set_ss_id(r->ss_id());
        cr->set_ss_type(r->ss_type());
        cr->_alt_loc = r->_alt_loc;
        cr->_ribbon_hide_backbone = r->_ribbon_hide_backbone;
        cr->_ribbon_selected = r->_ribbon_selected;
        cr->_ribbon_adjust = r->_ribbon_adjust;
        rmap[r] = cr;
    }
    std::map<CoordSet*, CoordSet*> cs_map;
    for (auto cs: coord_sets()) {
        auto new_cs = g->new_coord_set(cs->id());
        *new_cs = *cs;
        cs_map[cs] = new_cs;
    }
    g->set_active_coord_set(cs_map[active_coord_set()]);

    std::map<Atom*, Atom*> amap;
    for (auto ai = atoms().begin() ; ai != atoms().end() ; ++ai) {
        Atom* a = *ai;
        Atom* ca = g->new_atom(a->name(), a->element());
        Residue *cr = rmap[a->residue()];
        cr->add_atom(ca);	// Must set residue before setting alt locs
        ca->_coord_index = a->coord_index();
        std::set<char> alocs = a->alt_locs();
        if (!alocs.empty()) {
            char aloc = a->alt_loc();	// Remember original alt loc.
            for (auto ali = alocs.begin() ; ali != alocs.end() ; ++ali) {
                char al = *ali;
                a->set_alt_loc(al);
                ca->set_alt_loc(al, true);
                ca->set_coord(a->coord());
                ca->set_bfactor(a->bfactor());
                ca->set_occupancy(a->occupancy());
            }
            a->set_alt_loc(aloc);	// Restore original alt loc.
            ca->set_alt_loc(aloc);
        }
        ca->set_draw_mode(a->draw_mode());
        ca->set_radius(a->radius());
        ca->set_color(a->color());
        ca->set_display(a->display());
        amap[a] = ca;
    }
    
    for (auto bi = bonds().begin() ; bi != bonds().end() ; ++bi) {
        Bond* b = *bi;
        const Bond::Atoms& a = b->atoms();
        Bond* cb = g->new_bond(amap[a[0]], amap[a[1]]);
        cb->set_display(b->display());
        cb->set_color(b->color());
        cb->set_halfbond(b->halfbond());
        cb->set_radius(b->radius());
    }

    // Copy pseudobond groups.
    const AS_PBManager::GroupMap &gm = pb_mgr().group_map();
    for (auto gi = gm.begin() ; gi != gm.end() ; ++gi) {
        Proxy_PBGroup *pbg = gi->second;
        auto group_type = pbg->group_type();
        Proxy_PBGroup *pbgc = g->pb_mgr().get_group(gi->first, group_type);
        if (group_type == AS_PBManager::GRP_NORMAL) {
            _copy_pseudobonds(pbgc, pbg->pseudobonds(), amap);
        } else {
            // per coordinate set pseudobond groups
            for (auto cs: coord_sets()) {
                _copy_pseudobonds(pbgc, pbg->pseudobonds(cs), amap, cs_map[cs]);
            }
        }
    }
}

Structure*
Structure::copy() const
{
    Structure* g = new Structure(_logger);
    _copy(g);
    return g;
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
        change_tracker()->add_modified(a, ChangeTracker::REASON_ALT_LOC);
        a->_alt_loc_map.clear();
        if (aniso_u != nullptr)
            a->set_aniso_u((*aniso_u)[0], (*aniso_u)[1], (*aniso_u)[2],
                (*aniso_u)[3], (*aniso_u)[4], (*aniso_u)[5]);
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
    if (a->element().number() == 1)
        --_num_hyds;
    for (auto b: a->bonds()) {
        b->other_atom(a)->remove_bond(b);
        typename Bonds::iterator bi = std::find_if(_bonds.begin(), _bonds.end(),
            [&b](Bond* ub) { return ub == b; });
        _bonds.erase(bi);
    }
    a->residue()->remove_atom(a);
    typename Atoms::iterator i = std::find_if(_atoms.begin(), _atoms.end(),
        [&a](Atom* ua) { return ua == a; });
    _atoms.erase(i);
    set_gc_shape();
    set_gc_adddel();
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
        delete this;
        return;
    }
    auto r = a->residue();
    if (r->atoms().size() == 1) {
        _delete_residue(r);
        return;
    }
    _delete_atom(a);
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
        for (auto r: res_removals)
            if (r->chain() != nullptr) {
                r->chain()->remove_residue(r);
                set_gc_ribbon();
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

    for (auto a: _atoms) {
        std::vector<Bond*> removals;
        for (auto b: a->bonds()) {
            if (atoms.find(b->other_atom(a)) != atoms.end())
                removals.push_back(b);
        }
        for (auto b: removals)
            a->remove_bond(b);
    }

    auto new_b_end = std::remove_if(_bonds.begin(), _bonds.end(),
        [&atoms](Bond* b) {
            bool rm = atoms.find(b->atoms()[0]) != atoms.end()
            || atoms.find(b->atoms()[1]) != atoms.end();
            if (rm) delete b;
            return rm;
        });
    _bonds.erase(new_b_end, _bonds.end());
    set_gc_shape();
    set_gc_adddel();
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
    for (auto a: b->atoms())
        a->remove_bond(b);
    _bonds.erase(i);
    set_gc_shape();
    set_gc_adddel();
    _structure_cats_dirty = true;
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

Bond *
Structure::new_bond(Atom *a1, Atom *a2)
{
    Bond *b = new Bond(this, a1, a2);
    b->finish_construction(); // virtual calls work now
    add_bond(b);
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
    // pb manager version number remembered later
    if (PyList_Append(ints, npy_array) < 0)
        throw std::runtime_error("Couldn't append to int list");

    float* float_array;
    npy_array = python_float_array(SESSION_NUM_FLOATS(), &float_array);
    *float_array++ = _ball_scale;
    *float_array++ = _ribbon_tether_scale;
    *float_array++ = _ribbon_tether_opacity;
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
    // allocate for list of chain IDs
    PyObject* chain_misc = PyList_New(1);
    if (chain_misc == nullptr)
        throw std::runtime_error("Cannot create Python list for chain misc info");
    if (PyList_Append(misc, chain_misc) < 0)
        throw std::runtime_error("Couldn't append chain misc list to misc list");
    PyObject* chain_ids = PyList_New(num_chains);
    if (chain_ids == nullptr)
        throw std::runtime_error("Cannot create Python list for chain IDs");
    PyList_SET_ITEM(chain_misc, 0, chain_ids);
    i = 0;
    if (_chains != nullptr) {
        for (auto ch: *_chains) {
            num_ints += ch->session_num_ints();
            num_floats += ch->session_num_floats();

            // remember chain ID
            PyList_SET_ITEM(chain_ids, i++, cchar_to_pystring(ch->chain_id(), "chain chain ID"));
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
        auto a = new_atom(aname, Element::get_element(*element_ints++));
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
    if (!(PyTuple_Check(chain_misc) || PyList_Check(chain_misc)) || PySequence_Fast_GET_SIZE(chain_misc) != 1)
        throw std::invalid_argument("chain misc info is not a one-item tuple");
    std::vector<ChainID> chain_chain_ids;
    pysequence_of_string_to_cvec(PySequence_Fast_GET_ITEM(chain_misc, 0), chain_chain_ids, "chain ID");
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
        for (auto chain_id: chain_chain_ids) {
            auto chain = _new_chain(chain_id);
            chain->session_restore(version, &chain_ints, &chain_floats);
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
            change_tracker()->add_modified(this, ChangeTracker::REASON_ACTIVE_COORD_SET);
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
    for (auto r: _residues)
        r->set_ribbon_color(rgba);
}

void
Structure::use_best_alt_locs()
{
    std::map<Residue *, char> alt_loc_map = best_alt_locs();
    for (auto almi = alt_loc_map.begin(); almi != alt_loc_map.end(); ++almi) {
        (*almi).first->set_alt_loc((*almi).second);
    }
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

} //  namespace atomstruct
