// vim: set expandtab ts=4 sw=4:
#include "Atom.h"
#include "Bond.h"
#include "CoordSet.h"
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

AtomicStructure::AtomicStructure(): _active_coord_set(NULL),
    asterisks_translated(false), _being_destroyed(false), _chains(nullptr),
    lower_case_chains(false), _pb_mgr(this), pdb_version(0), is_traj(false)
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
                    const Atom::BondsMap &bm = a->bonds_map();
                    for (auto bi = bm.begin(); bi != bm.end(); ++bi) {
                        Atom *nb = (*bi).first;
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
AtomicStructure::make_chains(const AtomicStructure::ChainInfo* chain_info) const
// if null, fall back to chains derived from the structure directly
{
    ChainInfo* ci = (ChainInfo*)chain_info;

    if (ci == nullptr) {
        auto polys = polymers();
        // gather chain IDs and, if not unique, use numbers instead
        std::vector<std::string> ids;
        for (auto pi = polys.begin(); pi != polys.end(); ++pi) {
            ids.push_back((*(*pi).begin())->chain_id());
        }
        std::set<std::string> unique_ids(ids.begin(), ids.end());
        if (ids.size() != unique_ids.size()) {
            ids.clear();
            for (int i = 1; i <= polys.size(); ++i) {
                ids.push_back(std::to_string(i));
            }
        }
        ci = new ChainInfo();
        auto idi = ids.begin();
        for (auto pi = polys.begin(); pi != polys.end(); ++pi, ++idi) {
            ci->insert(ChainInfo::value_type(
                *idi, CI_Chain_Pairing(*pi, nullptr)));
        }
    }

    if (_chains != nullptr)
        delete _chains;

    _chains = new Chains();
    for (auto cii = ci->begin(); cii != ci->end(); ++cii) {
       const std::string& chain_id = (*cii).first;
       CI_Chain_Pairing chain_pairing = (*cii).second;
       Chain::Residues& residues = chain_pairing.first;
       Sequence::Contents* chars = chain_pairing.second;
       auto chain = new Chain(chain_id);
       _chains->emplace_back(chain);
       chain->bulk_set(residues, chars);
    }

    if (chain_info == nullptr)
        delete ci;
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
