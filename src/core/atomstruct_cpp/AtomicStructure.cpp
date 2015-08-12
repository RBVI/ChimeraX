// vi: set expandtab ts=4 sw=4:
#include "Atom.h"
#include "AtomicStructure.h"
#include <basegeom/destruct.h>
#include <basegeom/Graph.tcc>
#include "Bond.h"
#include "CoordSet.h"
#include "Element.h"
#include <logger/logger.h>
#include "Pseudobond.h"
#include "Residue.h"
#include "seq_assoc.h"

#include <algorithm>  // for std::find, std::sort, std::remove_if
#include <map>
#include <stdexcept>
#include <set>

namespace atomstruct {

const char*  AtomicStructure::PBG_METAL_COORDINATION = "metal coordination bonds";
const char*  AtomicStructure::PBG_MISSING_STRUCTURE = "missing structure";
const char*  AtomicStructure::PBG_HYDROGEN_BONDS = "hydrogen bonds";

AtomicStructure::AtomicStructure(PyObject* logger): _active_coord_set(NULL),
    _chains(nullptr), _idatm_valid(false), _logger(logger),
    _name("unknown AtomicStructure"), _pb_mgr(this), _polymers_computed(false),
    _recompute_rings(true), asterisks_translated(false), is_traj(false),
    lower_case_chains(false), pdb_version(0)
{
}

AtomicStructure::~AtomicStructure() {
    // assign to variable so that it lives to end of destructor
    auto du = basegeom::DestructionUser(this);
    if (_chains != nullptr) {
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

AtomicStructure *AtomicStructure::copy() const
{
  AtomicStructure *m = new AtomicStructure(_logger);

  m->set_name(name());

  for (auto h = pdb_headers.begin() ; h != pdb_headers.end() ; ++h)
    m->pdb_headers[h->first] = h->second;
  m->pdb_version = pdb_version;

  std::map<Residue *, Residue *> rmap;
  for (auto ri = residues().begin() ; ri != residues().end() ; ++ri)
    {
      Residue *r = *ri;
      Residue *cr = m->new_residue(r->name(), r->chain_id(), r->position(), r->insertion_code());
      cr->set_ribbon_display(r->ribbon_display());
      cr->set_ribbon_color(r->ribbon_color());
      cr->set_is_helix(r->is_helix());
      cr->set_is_sheet(r->is_sheet());
      cr->set_is_het(r->is_het());
      rmap[r] = cr;
    }

  std::map<Atom *, Atom*> amap;
  for (auto ai = atoms().begin() ; ai != atoms().end() ; ++ai)
    {
      Atom *a = *ai;
      Atom *ca = m->new_atom(a->name(), a->element());
      std::set<char> alocs = a->alt_locs();
      if (alocs.empty())
	{
	  ca->set_coord(a->coord());
	  ca->set_bfactor(a->bfactor());
	  ca->set_occupancy(a->occupancy());
	}
      else
	{
	  char aloc = a->alt_loc();	// Remember original alt loc.
	  for (auto ali = alocs.begin() ; ali != alocs.end() ; ++ali)
	    {
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
      Residue *cr = rmap[a->residue()];
      cr->add_atom(ca);
    }

  for (auto bi = bonds().begin() ; bi != bonds().end() ; ++bi)
    {
      Bond *b = *bi;
      const Bond::Atoms &a = b->atoms();
      Bond *cb = m->new_bond(amap[a[0]], amap[a[1]]);
      cb->set_display(b->display());
      cb->set_color(b->color());
      cb->set_halfbond(b->halfbond());
      cb->set_radius(b->radius());
    }

  return m;
}

std::map<Residue *, char>
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
                    Atom::_Alt_loc_info info = a->_alt_loc_map[alt_loc];
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
AtomicStructure::delete_atom(Atom* a)
{
    if (a->structure() != this) {
        logger::error(_logger, "Atom ", a->residue()->str(), " ", a->name(),
            " does not belong to the structure that it's being deleted from.");
        return;
    }
    if (atoms().size() == 1) {
        delete this;
        return;
    }
    auto r = a->residue();
    if (r->atoms().size() == 1) {
        _delete_residue(r, std::find(_residues.begin(), _residues.end(), r));
        return;
    }
    _delete_atom(a);
}

void
AtomicStructure::delete_atoms(std::vector<Atom*> del_atoms)
{
    auto du = basegeom::DestructionBatcher(this);

    // construct set first to ensure uniqueness before tests...
    auto del_atoms_set = std::set<Atom*>(del_atoms.begin(), del_atoms.end());
    if (del_atoms_set.size() == atoms().size()) {
        delete this;
        return;
    }
    std::map<Residue*, std::vector<Atom*>> res_del_atoms;
    for (auto a: del_atoms_set) {
        res_del_atoms[a->residue()].push_back(a);
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
        // remove_if apparently doesn't guarantee that the _back_ of
        // the vector is all the removed items -- there could be second
        // copies of the retained values in there, so do the delete as
        // part of the lambda rather than in a separate pass through
        // the end of the vector
        auto new_end = std::remove_if(_residues.begin(), _residues.end(),
            [&res_removals](Residue* r) {
                bool rm = res_removals.find(r) != res_removals.end();
                if (rm) delete r; return rm;
            });
        _residues.erase(new_end, _residues.end());
    }
    delete_vertices(std::set<Atom*>(del_atoms.begin(), del_atoms.end()));
}

void
AtomicStructure::_delete_residue(Residue* r,
    const AtomicStructure::Residues::iterator& ri)
{
    auto db = basegeom::DestructionBatcher(r);
    for (auto a: r->atoms()) {
        _delete_atom(a);
    }
    _residues.erase(ri);
    delete r;
}

void
AtomicStructure::delete_residue(Residue* r)
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
    _delete_residue(r, ri);
}

CoordSet *
AtomicStructure::find_coord_set(int id) const
{
    for (auto csi = _coord_sets.begin(); csi != _coord_sets.end(); ++csi) {
        if ((*csi)->id() == id)
            return *csi;
    }

    return NULL;
}

Residue *
AtomicStructure::find_residue(const ChainID &chain_id, int pos, char insert) const
{
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = *ri;
        if (r->position() == pos && r->chain_id() == chain_id
        && r->insertion_code() == insert)
            return r;
    }
    return NULL;
}

Residue *
AtomicStructure::find_residue(const ChainID& chain_id, int pos, char insert, ResName& name) const
{
    for (auto ri = _residues.begin(); ri != _residues.end(); ++ri) {
        Residue *r = *ri;
        if (r->position() == pos && r->name() == name && r->chain_id() == chain_id
        && r->insertion_code() == insert)
            return r;
    }
    return NULL;
}

void
AtomicStructure::make_chains() const
{
    if (_chains != nullptr) {
        for (auto c: *_chains)
            delete c;
        delete _chains;
    }

    _chains = new Chains();
    auto polys = polymers();

    // for chain IDs associated with a single polymer, we can try to
    // form a Chain using SEQRES record.  Otherwise, form a Chain based
    // on structure only
    std::map<ChainID, bool> unique_chain_id;
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
        const ChainID& chain_id = polymer[0]->chain_id();
        auto chain = _new_chain(chain_id);

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
                    chain_id, " of ", _name, " is incomplete.  "
                    "Ignoring input sequence records as basis for sequence.");
                continue;
            }

            // skip if standard residues have been removed but the
            // sequence records haven't been...
            Sequence sr_seq(three_let_seq);
            if ((unsigned)std::count(chain->begin(), chain->end(), 'X') == chain_size
            && std::search(sr_seq.begin(), sr_seq.end(),
            chain->begin(), chain->end()) == sr_seq.end()) {
                logger::warning(_logger, "Residues corresponding to ",
                    input_seq_source, " for chain ", chain_id, " of ", _name,
                    " are missing.  Ignoring record as basis for sequence.");
                continue;
            }

            // okay, seriously try to match up with SEQRES
            auto ap = estimate_assoc_params(*chain);

            // UNK residues may be jammed up against the regular sequnce
            // in SEQRES records (3dh4, 4gns) despite missing intervening
            // residues; compensate...
            //
            // can't just test against est_len since there can be other
            // missing structure

            // leading Xs...
            unsigned int additional_Xs = 0;
            unsigned int existing_Xs = 0;
            auto gi = ap.gaps.begin();
            for (auto si = ap.segments.begin(); si != ap.segments.end()
            && si+1 != ap.segments.end(); ++si, ++gi) {
                auto seg = *si;
                if (std::find_if_not(seg.begin(), seg.end(),
                [](char c){return c == 'X';}) == seg.end()) {
                    // all 'X'
                    existing_Xs += seg.size();
                    additional_Xs += *gi;
                } else {
                    break;
                }
            }
            if (existing_Xs && sr_seq.size() >= existing_Xs
            && std::count(sr_seq.begin(), sr_seq.begin() + existing_Xs, 'X')
            == existing_Xs)
                sr_seq.insert(sr_seq.begin(), additional_Xs, 'X');

            // trailing Xs...
            additional_Xs = 0;
            existing_Xs = 0;
            auto rgi = ap.gaps.rbegin();
            for (auto rsi = ap.segments.rbegin(); rsi != ap.segments.rend()
            && rsi+1 != ap.segments.rend(); ++rsi, ++rgi) {
                auto seg = *rsi;
                if (std::find_if_not(seg.begin(), seg.end(),
                [](char c){return c == 'X';}) == seg.end()) {
                    // all 'X'
                    existing_Xs += seg.size();
                    additional_Xs += *rgi;
                } else {
                    break;
                }
            }
            if (existing_Xs && sr_seq.size() >= existing_Xs
            && std::count(sr_seq.rbegin(), sr_seq.rbegin() + existing_Xs, 'X')
            == existing_Xs)
                sr_seq.insert(sr_seq.end(), additional_Xs, 'X');

            // if a jump in numbering is in an unresolved part of the structure,
            // the estimated length can be too long...
            if (ap.est_len < sr_seq.size())
                ap.est_len = sr_seq.size();

            // since gapping a structure sequence is considered an "error",
            // need to allow a lot more errors than normal.  However, allowing
            // a _lot_ of errors can make it take a very long time to find the
            // answer, so limit the maximum...
            // (1vqn, chain 0 is > 2700 residues)
            unsigned int seq_len = chain->size();
            unsigned int gap_sum = 0;
            for (auto gap: ap.gaps) {
                gap_sum += gap;
            }
            unsigned int max_errs = std::min(seq_len/2,
                std::max(seq_len/10, gap_sum));
            AssocRetvals retvals;
            try {
                retvals = try_assoc(sr_seq, *chain, ap, max_errs);
            } catch (SA_AssocFailure) {
                chain->set_from_seqres(false);
                continue;
            }
            auto& p2r = retvals.match_map.pos_to_res();
            Chain::Residues new_residues;
            for (Chain::SeqPos i = 0; i < sr_seq.size(); ++i ) {
                auto pi = p2r.find(i);
                if (pi == p2r.end())
                    new_residues.push_back(nullptr);
                else
                    new_residues.push_back((*pi).second);
            }
            chain->bulk_set(new_residues, &sr_seq.contents());
        }
    }
}

Atom *
AtomicStructure::new_atom(const char* name, Element e)
{
    Atom *a = new Atom(this, name, e);
    add_vertex(a);
    if (e.number() == 1)
        ++_num_hyds;
    return a;
}

Bond *
AtomicStructure::new_bond(Atom *a1, Atom *a2)
{
    Bond *b = new Bond(this, a1, a2);
    b->finish_construction(); // virtual calls work now
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
    CoordSet* cs, int index)
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
            delete *csi;
            coord_sets.insert(csi, cs);
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
    CoordSet* cs = new CoordSet(this, index);
    _coord_set_insert(_coord_sets, cs, index);
    return cs;
}

CoordSet*
AtomicStructure::new_coord_set(int index, int size)
{
    CoordSet* cs = new CoordSet(this, index, size);
    _coord_set_insert(_coord_sets, cs, index);
    return cs;
}

Residue*
AtomicStructure::new_residue(const ResName& name, const ChainID& chain,
    int pos, char insert, Residue *neighbor, bool after)
{
    if (neighbor == NULL) {
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

std::vector<Chain::Residues>
AtomicStructure::polymers(bool consider_missing_structure,
    bool consider_chain_ids) const
{
    // if consider_missing_structure is false, just consider actual
    // existing polymeric bonds (not missing-segment pseudobonds);
    // if consider_chain_ids is true, don't have a polymer span
    // a change in chain ID

    // connected polymeric residues have to be adjacent in the residue list,
    // so make an index map
    int i = 0;
    std::map<const Residue*, int> res_lookup;
    for (auto r: _residues) {
        res_lookup[r] = i++;
    }

    // Find all polymeric connections and make a map
    // keyed on residue with value of whether that residue
    // is connected to the next one
    std::map<Residue*, bool> connected;
    for (auto b: bonds()) {
        Atom* start = b->polymeric_start_atom();
        if (start != nullptr) {
            Residue* sr = start->residue();
            Residue* nr = b->other_atom(start)->residue();
            if (res_lookup[sr] + 1 == res_lookup[nr]
            && (!consider_chain_ids || sr->chain_id() == nr->chain_id()))
                // If consider_chain_ids is true,
                // if an artificial linker is used to join
                // otherwise unconnected amino acid chains,
                // they all can have different chain IDs,
                // and should be treated as separate chains (2atp)
                connected[sr] = true;
        }
    }

    if (consider_missing_structure) {
        // go through missing-structure pseudobonds
        auto pbg = (Owned_PBGroup*) const_cast<AtomicStructure*>(this)->_pb_mgr.get_group(
            PBG_MISSING_STRUCTURE, AS_PBManager::GRP_NONE);
        if (pbg != nullptr) {
            for (auto& pb: pbg->pseudobonds()) {
                Residue *r1 = pb->atoms()[0]->residue();
                Residue *r2 = pb->atoms()[1]->residue();
                int index1 = res_lookup[r1], index2 = res_lookup[r2];
                if (abs(index1 - index2) == 1
                && r1->chain_id() == r2->chain_id()) {
                    if (index1 < index2) {
                        connected[r1] = true;
                    } else {
                        connected[r2] = true;
                    }
                }
            }
        }
    }

    // Go through residue list; start chains with initially-connected residues
    std::vector<Chain::Residues> polys;
    Chain::Residues chain;
    bool in_chain = false;
    for (auto& upr: _residues) {
        Residue* r = upr;
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

    _polymers_computed = true;
    return polys;
}

const AtomicStructure::Rings&
AtomicStructure::rings(bool cross_residues, unsigned int all_size_threshold,
    std::set<const Residue *>* ignore) const
{
    if (_rings_cached(cross_residues, all_size_threshold, ignore)) {
        return _rings;
    }

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

bool
AtomicStructure::_rings_cached(bool cross_residues,
    unsigned int all_size_threshold,
    std::set<const Residue *>* ignore) const
{
    return !_recompute_rings && cross_residues == _rings_last_cross_residues
        && all_size_threshold == _rings_last_all_size_threshold
        && ignore == _rings_last_ignore;
}

void
AtomicStructure::set_active_coord_set(CoordSet *cs)
{
    CoordSet *new_active;
    if (cs == NULL) {
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
        set_gc_shape();
    }
}

void
AtomicStructure::use_best_alt_locs()
{
    std::map<Residue *, char> alt_loc_map = best_alt_locs();
    for (auto almi = alt_loc_map.begin(); almi != alt_loc_map.end(); ++almi) {
        (*almi).first->set_alt_loc((*almi).second);
    }
}

}  // namespace atomstruct
