// vim: set expandtab ts=4 sw=4:
#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
#include <logger/logger.h>
#include "Element.h"
#include "AtomicStructure.h"
#include "Residue.h"
#include "Pseudobond.h"

#include <algorithm>  // for std::find, std::sort
#include <stdexcept>
#include <set>
#include <unordered_map>

namespace atomstruct {

const char*  AtomicStructure::PBG_METAL_COORDINATION = "metal coordination bonds";
const char*  AtomicStructure::PBG_MISSING_STRUCTURE = "missing structure";

AtomicStructure::AtomicStructure(PyObject* logger): _active_coord_set(NULL),
    asterisks_translated(false), _being_destroyed(false), _chains(nullptr),
    _idatm_valid(false), _logger(logger), lower_case_chains(false),
    _name("unknown AtomicStructure"), _pb_mgr(this), _recompute_rings(true),
    pdb_version(0), is_traj(false)
{
}

std::unordered_map<Residue *, char>
AtomicStructure::best_alt_locs() const
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
    std::unordered_map<Residue *, char> best_locs;
    if (all_blank) {
        return best_locs;
    }

    // go through the residues and collate a group of residues with
    //   related alt locs
    // use the alt loc with the highest average occupancy; if tied,
    //  the lowest bfactors; if tied, first alphabetically
    std::set<Residue *> seen;
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = (*ri).get();
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
        std::unordered_map<char, int> occurances;
        std::unordered_map<char, float> occupancies, bfactors;
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
                    Atom::_Alt_loc_info info = a->_alt_loc_map[alt_loc];
                    occupancies[alt_loc] += info.occupancy;
                    bfactors[alt_loc] += info.bfactor;
                }
                if (check_neighbors) {
                    for (auto nb: a->neighbors()) {
                        Residue *nr = nb->residue();
                        if (nr != cr && nb->has_alt_loc(*alt_loc_set.begin())
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

CoordSet *
AtomicStructure::find_coord_set(int id) const
{
    for (auto csi = _coord_sets.begin(); csi != _coord_sets.end(); ++csi) {
        if ((*csi)->id() == id)
            return csi->get();
    }

    return NULL;
}

Residue *
AtomicStructure::find_residue(std::string &chain_id, int pos, char insert) const
{
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = (*ri).get();
        if (r->position() == pos && r->chain_id() == chain_id
        && r->insertion_code() == insert)
            return r;
    }
    return NULL;
}

Residue *
AtomicStructure::find_residue(std::string &chain_id, int pos, char insert, std::string &name) const
{
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = (*ri).get();
        if (r->position() == pos && r->name() == name && r->chain_id() == chain_id
        && r->insertion_code() == insert)
            return r;
    }
    return NULL;
}

void
AtomicStructure::make_chains() const
{
    if (_chains != nullptr)
        delete _chains;

    _chains = new Chains();
    auto polys = polymers();

    // for chain IDs associated with a single polymer, we can try to
    // form a Chain using SEQRES record.  Otherwise, form a Chain based
    // on structure only
    std::map<std::string, bool> unique_chain_id;
    if (!_input_seq_info.empty()) {
        for (auto polymer: polys) {
            auto chain_id = polymer[0]->chain_id();
            if (unique_chain_id.find(chain_id) == unique_chain_id.end()) {
                unique_chain_id[chain_id] = true;
            } else {
                unique_chain_id[chain_id] = false;
            }
        }
    }
    for (auto polymer: polys) {
        const std::string& chain_id = polymer[0]->chain_id();
        auto chain = new Chain(chain_id);
        _chains->emplace_back(chain);

        // first, create chain directly from structure
        chain->bulk_set(polymer, nullptr);

        auto three_let_i = _input_seq_info.find(chain_id);
        if (three_let_i != _input_seq_info.end()
        && unique_chain_id[chain_id]) {
            // try to adjust chain based on SEQRES
            auto& three_let_seq = three_let_i->second;
            auto seqres_size = three_let_seq.size();
            auto chain_size = chain->size();
            if (seqres_size == chain_size) {
                // presumably no adjustment necessary
                chain->set_from_seqres(true);
                continue;
            }

            if (seqres_size < chain_size) {
                logger::warning(_logger, input_seq_source, " for chain ",
                    chain_id, " of ", _name, " is incomplete.\n"
                    "Ignoring input sequence records as basis for sequence.");
                continue;
            }

            // skip if standard residues have been removed but the
            // sequence records haven't been...
            Sequence sr_seq(three_let_seq);
            if (std::count(chain->begin(), chain->end(), 'X') == chain_size
            && std::search(sr_seq.begin(), sr_seq.end(),
            chain->begin(), chain->end()) == sr_seq.end()) {
                logger::warning(_logger, "Residues corresponding to ",
                    input_seq_source, " for chain ", chain_id, " of ", _name,
                    " are missing.\nIgnoring record as basis for sequence.");
                continue;
            }

            // okay, seriously try to match up with SEQRES
        }
    }
}

Atom *
AtomicStructure::new_atom(const std::string &name, Element e)
{
    Atom *a = new Atom(this, name, e);
    add_vertex(a);
    return a;
}

Bond *
AtomicStructure::new_bond(Atom *a1, Atom *a2)
{
    Bond *b = new Bond(this, a1, a2);
    add_edge(b);
    return b;
}

CoordSet *
AtomicStructure::new_coord_set()
{
    if (_coord_sets.empty())
        return new_coord_set(0);
    return new_coord_set(_coord_sets.back()->id());
}

static void
_coord_set_insert(AtomicStructure::CoordSets &coord_sets,
    std::unique_ptr<CoordSet>& cs, int index)
{
    if (coord_sets.empty() || coord_sets.back()->id() < index) {
        coord_sets.emplace_back(cs.release());
        return;
    }
    for (auto csi = coord_sets.begin(); csi != coord_sets.end(); ++csi) {
        if (index < (*csi)->id()) {
            coord_sets.insert(csi, std::move(cs));
            return;
        } else if (index == (*csi)->id()) {
            (*csi).reset(cs.release());
            return;
        }
    }
    std::logic_error("CoordSet insertion logic error");
}

CoordSet*
AtomicStructure::new_coord_set(int index)
{
    if (!_coord_sets.empty())
        return new_coord_set(index, _coord_sets.back()->coords().size());
    std::unique_ptr<CoordSet> cs(new CoordSet(this, index));
    CoordSet* retval = cs.get();
    _coord_set_insert(_coord_sets, cs, index);
    return retval;
}

CoordSet*
AtomicStructure::new_coord_set(int index, int size)
{
    std::unique_ptr<CoordSet> cs(new CoordSet(this, index, size));
    CoordSet* retval = cs.get();
    _coord_set_insert(_coord_sets, cs, index);
    return retval;
}

Residue*
AtomicStructure::new_residue(const std::string &name, const std::string &chain,
    int pos, char insert, Residue *neighbor, bool after)
{
    if (neighbor == NULL) {
        _residues.emplace_back(new Residue(this, name, chain, pos, insert));
        return _residues.back().get();
    }
    auto ri = std::find_if(_residues.begin(), _residues.end(),
                [&neighbor](std::unique_ptr<Residue>& vr)
                { return vr.get() == neighbor; });
    if (ri == _residues.end())
        throw std::out_of_range("Waypoint residue not in residue list");
    if (after)
        ++ri;
    Residue *r = new Residue(this, name, chain, pos, insert);
    _residues.insert(ri, std::unique_ptr<Residue>(r));
    return r;
}

std::vector<Chain::Residues>
AtomicStructure::polymers() const
{
    // connected polymeric residues have to be adjacent in the residue list,
    // so make an index map
    int i = 0;
    std::unordered_map<const Residue*, int> res_lookup;
    for (auto& r: _residues) {
        res_lookup[r.get()] = i++;
    }

    // Find all polymeric connections and make a map
    // keyed on residue with value of whether that residue
    // is connected to the next one
    std::unordered_map<Residue*, bool> connected;
    for (auto& b: bonds()) {
        Atom* start = b->polymeric_start_atom();
        if (start != nullptr) {
            Residue* sr = start->residue();
            Residue* nr = b->other_atom(start)->residue();
            if (res_lookup[sr] + 1 == res_lookup[nr])
                connected[sr] = true;
        }
    }

    // go through missing-structure pseudobonds
    auto pbg = (Owned_PBGroup*) _pb_mgr.get_group(PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
    if (pbg != nullptr) {
        for (auto& pb: pbg->pseudobonds()) {
            Residue *r1 = pb->atoms()[0]->residue();
            Residue *r2 = pb->atoms()[1]->residue();
            int index1 = res_lookup[r1], index2 = res_lookup[r2];
            if (abs(index1 - index2) == 1) {
                if (index1 < index2) {
                    connected[r1] = true;
                } else {
                    connected[r2] = true;
                }
            }
        }
    }

    // Go through residue list; start chains with initially-connected residues
    std::vector<Chain::Residues> polys;
    Chain::Residues chain;
    bool in_chain = false;
    for (auto& upr: _residues) {
        Residue* r = upr.get();
        auto connection = connected.find(r);
        if (connection == connected.end()) {
            if (in_chain) {
                chain.push_back(r);
                polys.push_back(chain);
                chain.clear();
                in_chain = false;
            }
        } else {
            chain.push_back(r);
            in_chain = true;
        }
    }
    if (in_chain) {
        polys.push_back(chain);
    }

    return polys;
}

const AtomicStructure::Rings&
AtomicStructure::rings(bool cross_residues, unsigned int all_size_threshold,
    std::set<const Residue *>* ignore) const
{
    if (!_recompute_rings && cross_residues == _rings_last_cross_residues
    && all_size_threshold == _rings_last_all_size_threshold
    && ignore == _rings_last_ignore)
        return _rings;

    _recompute_rings = false;
    _rings_last_cross_residues = cross_residues;
    _rings_last_all_size_threshold = all_size_threshold;
    _rings_last_ignore = ignore;

    _calculate_rings(cross_residues, all_size_threshold, ignore);

    // clear out ring lists in individual atoms and bonds
    for (auto& a: atoms()) {
        a->_rings.clear();
    }
    for (auto& b: bonds()) {
        b->_rings.clear();
    }

    // set individual atom/bond ring lists
    for (auto &r: _rings) {
        for (auto a: r.atoms()) {
            a->_rings.push_back(&r);
        }
        for (auto b: r.bonds()) {
            b->_rings.push_back(&r);
        }
    }
    return _rings;
}

void
AtomicStructure::set_active_coord_set(CoordSet *cs)
{
    CoordSet *new_active;
    if (cs == NULL) {
        if (_coord_sets.empty())
            return;
        new_active = _coord_sets.front().get();
    } else {
        CoordSets::iterator csi = std::find_if(_coord_sets.begin(), _coord_sets.end(),
                [&cs](std::unique_ptr<CoordSet>& vcs) { return vcs.get() == cs; });
        if (csi == _coord_sets.end())
            throw std::out_of_range("Requested active coord set not in coord sets");
        new_active = cs;
    }
    _active_coord_set = new_active;
}

void
AtomicStructure::use_best_alt_locs()
{
    std::unordered_map<Residue *, char> alt_loc_map = best_alt_locs();
    for (auto almi = alt_loc_map.begin(); almi != alt_loc_map.end(); ++almi) {
        (*almi).first->set_alt_loc((*almi).second);
    }
}

}  // namespace atomstruct
