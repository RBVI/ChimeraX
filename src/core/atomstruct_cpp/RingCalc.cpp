// vi: set expandtab ts=4 sw=4:

#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#define MAP map
#define SET set

#include "AtomicStructure.h"

namespace atomstruct {

static Ring::Bonds::iterator contains_exactly_one(
        const Ring &ring, Ring::Bonds *bond_set);


// find all rings
//  if 'cross_residues', look for rings that cross residue boundaries
//      in addition to intraresidue rings
//  if 'all_size_threshold' is non-zero, in addition to minimal rings return
//      _all_ rings of at least the given size [ and no rings
//      greater than that size, even if minimal ]
void
AtomicStructure::_calculate_rings(bool cross_residue,
    unsigned int all_size_threshold,
    std::set<const Residue *>* ignore) const
{
    // this routine largely based on the algorithm found in:
    //  "An Algorithm for Machine Perception of Synthetically
    //  Significant Rings in Complex Cyclic Organic Structures";
    //  E.J. Corey and George A. Petersson, JACS, 94:2, Jan. 26, 1972
    //  pp. 460-465

    // set_symmetric_difference only works on sorted ranges, so use
    // std::set instead of std::unordered_set
    typedef std::set<Bond*> SpanningBonds;
    typedef std::MAP<Atom*, SpanningBonds > Atom2SpanningBonds;
    std::MAP<Bond*, bool> traversed;
    std::MAP<Atom*, bool> visited;
    // rcb2fr:  ring closure bond to fundamental ring map
    std::MAP<Bond *, Ring> rcb2fr;
    std::vector<Rings> fundamentals;

    // trace spanning tree and find fundamental rings
    // look for an atom not yet in the spanning tree;
    // keep a list of fundamental rings per spanning tree to
    // optimize the ring summing step to follow
    for (auto& uat: atoms()) {
        Atom* at = uat.get();
        if (visited.find(at) != visited.end())
            continue;
        if (ignore != nullptr && ignore->find(at->residue()) != ignore->end())
            continue;
        
        // very big residues cause fits; just skip them
        if (!cross_residue && at->residue()->atoms().size() > 2500)
            continue;

        Atom2SpanningBonds spanning_bonds;
        spanning_bonds[at] = SpanningBonds();
        std::list<Atom *> leaves;
        leaves.push_back(at);
        Rings fundamental;

        while (leaves.size() > 0) {
            Atom *node = leaves.front();
            leaves.pop_front();
            if (ignore != nullptr
            && ignore->find(node->residue()) != ignore->end())
                continue;
            visited[node] = true;

            auto bi = node->bonds().begin();
            auto ni = node->neighbors().begin();
            for (; bi != node->bonds().end(); ++bi, ++ni) {
                Bond* b = *bi;
                if (traversed[b])
                    continue;
                traversed[b] = true;

                Atom *next = *ni;
                if (!cross_residue && node->residue() != next->residue())
                    continue;
                if (ignore != nullptr
                && ignore->find(next->residue()) != ignore->end())
                    continue;

                if (!visited[next]) {
                    // not yet visited
                    visited[next] = true;
                    spanning_bonds[next] = spanning_bonds[node];
                    (void)spanning_bonds[next].insert(b);
                    leaves.push_back(next);
                    continue;
                }

                // else:
                // visited; fundamental ring
                //
                // do an "exclusive or" of the spanning
                // bonds of the two ends of the ring
                // closure bond to find the ring bonds
                // (and add the ring closure bond)
                Ring::Bonds ring_bonds;
                SpanningBonds &sb_node = spanning_bonds[node];
                SpanningBonds &sb_next = spanning_bonds[next];
                std::set_symmetric_difference(
                  sb_node.begin(), sb_node.end(),
                  sb_next.begin(), sb_next.end(),
                  std::insert_iterator<Ring::Bonds>(
                  ring_bonds, ring_bonds.begin()));
                (void)ring_bonds.insert(b);
                Ring r(ring_bonds);
                (void)fundamental.insert(r);
                (void)rcb2fr.insert(std::MAP<Bond *, Ring>::value_type(b,r));
            }
        }
        fundamentals.push_back(fundamental);
    }

    // "ring sum" pairs of fundamental rings; keep smallest two
    Rings basis;
    std::set<Bond *> sum;
    for (auto& fundamental: fundamentals) {
        if (fundamental.size() == 1) {
            basis.insert(*fundamental.begin());
            continue;
        }
        for (auto fi1 = fundamental.begin(); fi1 != fundamental.end(); ++fi1) {
            const Ring& r1 = *fi1;
            Rings::const_iterator fi2 = fi1;
            for (fi2++; fi2 != fundamental.end(); ++fi2) {
                const Ring &r2 = *fi2;
                const Ring *larger = r1.bonds().size() >
                  r2.bonds().size() ? &r1 : &r2;
                sum.clear();
                std::set_symmetric_difference(
                  r1.bonds().begin(), r1.bonds().end(),
                  r2.bonds().begin(), r2.bonds().end(),
                  std::insert_iterator<std::set<Bond *>>(sum, sum.begin()));
                
                if (sum.size() < larger->bonds().size()) {
                    (void)basis.insert(Ring(sum));
                    (void)basis.insert(larger == &r1 ?  r2 : r1);
                } else {
                    (void)basis.insert(r1);
                    (void)basis.insert(r2);
                }
            }
        }
    }
    if (basis.size() == 0) {
        _rings.clear();
        return;
    }

    // grow rings from ring closure bonds that are no bigger than the
    // smaller of: that bond's fundamental ring or largest basis ring
    Ring::Bonds::size_type largest_basis = 0;
    for (auto& r: basis) {
        if (r.bonds().size() > largest_basis)
            largest_basis = r.bonds().size();
    }
    if (all_size_threshold && largest_basis > all_size_threshold)
        largest_basis = all_size_threshold;

    for (auto rcb_fr: rcb2fr) {
        Bond *rcb = rcb_fr.first; // ring closure bond
        const Ring &fr = rcb_fr.second; // corresponding fundamental ring 

        Ring::Bonds::size_type limit_size = largest_basis < fr.bonds().size() ?
          largest_basis : fr.bonds().size();
        
        // grow trees from each end of the rcb;  when leaves on
        // opposite trees match, a ring is indicated.  Track
        // the bonds "above" each leaf to avoid backtracking
        //
        // also, set_symmetric_difference only works on ordered
        // ranges, so use std::set instead of std::unordered_set
        std::MAP<Atom *, std::set<Bond *> > above_bonds[2];
        std::SET<Atom *> leaf_atoms[2];
        above_bonds[0][rcb->atoms()[0]].insert(rcb);
        above_bonds[1][rcb->atoms()[1]].insert(rcb);
        leaf_atoms[0].insert(rcb->atoms()[0]);
        leaf_atoms[1].insert(rcb->atoms()[1]);
        for (Ring::Bonds::size_type size = 2; size <= limit_size; ++size) {
            int side = size % 2;
            int opp_side = (side + 1) % 2;

            std::MAP<Atom *, std::set<Bond *> >& above = above_bonds[side];
            std::SET<Atom *>& leaves = leaf_atoms[side];
            std::SET<Atom *>& opp_leaves = leaf_atoms[opp_side];

            std::SET<Atom *> new_leaves;
            std::MAP<Atom *, std::set<Bond *> > new_above;

            for (auto leaf: leaves) {
                auto ei = leaf->neighbors().begin();
                auto bi = leaf->bonds().begin();
                for (; ei != leaf->neighbors().end(); ++ei, ++bi) {
                    Atom *end = *ei;
                    Bond *b = *bi;

                    if (above[leaf].find(b) != above[leaf].end())
                        continue;

                    if (!cross_residue && leaf->residue() != end->residue())
                        continue;
                    
                    if (ignore != nullptr
                    && ignore->find(end->residue()) != ignore->end())
                        continue;

                    new_above[end] = above[leaf];
                    (void)new_above[end].insert(b);
                    (void)new_leaves.insert(end);

                    if (opp_leaves.find(end) == opp_leaves.end())
                        continue;
                    
                    // forms a ring
                    Ring::Bonds ring_bonds;
                    std::set_symmetric_difference(
                      new_above[end].begin(),
                      new_above[end].end(),
                      above_bonds[opp_side][end].begin(),
                      above_bonds[opp_side][end].end(),
                      std::insert_iterator<Ring::Bonds>(
                      ring_bonds, ring_bonds.begin()));
                    if (ring_bonds.size() < 1)
                        continue;
                    (void)ring_bonds.insert(rcb);
                    (void)basis.insert(Ring(ring_bonds));
                }
            }

            leaves.swap(new_leaves);
            above.swap(new_above);
        }
    }

    std::SET<Bond *> basis_union;
    for (auto& r: basis)
        basis_union.insert(r.bonds().begin(), r.bonds().end());

    Rings msr; // msr == Minimum Spanning Rings
    std::SET<Bond *> msr_union;
    auto basis_iter = basis.begin();
    while (msr_union.size() < basis_union.size()) {
        Ring::Bonds::size_type ring_size = (*basis_iter).bonds().size();
        Rings new_coverage; // rings with bonds not yet in msr_union
        for (; basis_iter != basis.end(); ++basis_iter) {
            const Ring &r = *basis_iter;
            if (r.bonds().size() > ring_size) {
                break;
            }
            if (!std::includes(msr_union.begin(), msr_union.end(),
            r.bonds().begin(), r.bonds().end())) {
                new_coverage.insert(r);
            }
        }
        for (auto& r: new_coverage) {
            msr.insert(r);
            msr_union.insert(r.bonds().begin(), r.bonds().end());
        }
    }

    // construct the set of ring-closure bonds whose fundamental ring
    // in not in msr
    Ring::Bonds check_rcbs;
    for (auto rcb_fr: rcb2fr) {
        Bond *rcb = rcb_fr.first;
        const Ring &fr = rcb_fr.second;

        if (msr.find(fr) == msr.end())
            check_rcbs.insert(rcb);
    }

    // if some ring in msr contains only one bond in check_rcbs, remove
    // that bond from check_rcbs
    while (check_rcbs.size() > 0) {
        bool some_removed = true;
        while (check_rcbs.size() > 0 && some_removed) {
            some_removed = false;
            for (auto& r: msr) {
                Ring::Bonds::iterator bi = contains_exactly_one(r, &check_rcbs);
                
                if (bi == check_rcbs.end())
                    continue;

                // ring contains exactly one bond in check_rcbs
                check_rcbs.erase(bi);
                some_removed = true;
            }
        }
        if (check_rcbs.size() == 0)
            break;
        
        // hoo boy, now for some ugly stuff (see pg. 464, col. 2, 2nd
        // para. of the ref. for textual description...

        // find the rings in msr that have bonds in check_rcbs
        Rings have_check_bond;
        for (auto& r: msr) {
            Ring::Bonds::const_iterator found = std::find_first_of(
                r.bonds().begin(), r.bonds().end(),
                check_rcbs.begin(), check_rcbs.end());
            if (found == r.bonds().end())
                continue;
            have_check_bond.insert(r);
        }

        // form the set of rings that are linear combinations of the
        // these checkrcb-containing msr rings
        Rings linear = have_check_bond;
        for (Rings::const_iterator li = have_check_bond.begin();
        li != have_check_bond.end(); ++li) {
            const Ring &lr = *li;
            for (Rings::const_iterator nli = li;
            nli != have_check_bond.end(); ++nli) {
                if (li == nli)
                    continue;
                const Ring &nlr = *nli;
                Ring::Bonds ringSum;
                std::set_symmetric_difference(
                  lr.bonds().begin(), lr.bonds().end(),
                  nlr.bonds().begin(), nlr.bonds().end(),
                  std::insert_iterator<Ring::Bonds>(
                  ringSum, ringSum.begin()));
                if (ringSum.size() == 0
                || ringSum.size() == nlr.bonds().size() + lr.bonds().size())
                    continue;
                Ring lc(ringSum);
                linear.insert(lc);
            }
        }
        
        // if any linear combo ring contains exactly one checkrcb,
        // remove that bond from checkrcb
        for (auto& r: linear) {
            Ring::Bonds::iterator bi = contains_exactly_one(r, &check_rcbs);
            
            if (bi != check_rcbs.end())
                check_rcbs.erase(bi);
        }
        if (check_rcbs.size() == 0)
            break;
        
        // find the smallest rings that have bonds in check_rcbs but
        // that aren't linear combo rings.  Add them to msr.
        Rings add_to_msr;
        basis_iter = basis.begin();
        while (add_to_msr.size() == 0) {
            Ring::Bonds::size_type ring_size = (*basis_iter).bonds().size();
            if (ring_size > all_size_threshold) {
                check_rcbs.clear();
                break;
            }
            for (; basis_iter != basis.end(); ++basis_iter) {
                const Ring &r = *basis_iter;
                if (r.bonds().size() > ring_size) {
                    break;
                }
                Ring::Bonds::const_iterator bi =
                  std::find_first_of(r.bonds().begin(),
                  r.bonds().end(), check_rcbs.begin(),
                  check_rcbs.end());
                if (bi == r.bonds().end())
                    // no bond in common with check_rcbs
                    continue;
                if (linear.find(r) != linear.end())
                    // is a linear combo
                    continue;
                add_to_msr.insert(r);
            }
        }
        for (Rings::const_iterator ri = add_to_msr.begin();
        ri != add_to_msr.end(); ++ri) {
            const Ring &r = *ri;
            msr.insert(r);
        }
    }
    msr.swap(_rings);

    if (all_size_threshold > 0) {
        // return _all_ at least the given size

        // to optimize the cross_residue==false case, sort
        // rings by residue...

        Rings meets_size_criteria;
        for (auto& r: _rings) {
            if (r.bonds().size() <= all_size_threshold)
                meets_size_criteria.insert(r);
        }
        meets_size_criteria.swap(_rings);

        std::MAP<Residue *, Rings> ring_lists;
        if (cross_residue) {
            ring_lists[residues()[0].get()] = _rings;
        } else {
            for (auto& r: _rings) {
                ring_lists[(*r.bonds().begin())->atoms()[0]
                        ->residue()].insert(r);
            }
        }

        bool rings_added = true;
        Rings new_added;
        while (rings_added) {
            rings_added = false;
            Rings all_additional;
            for (auto r1res_rli: ring_lists) {
                Rings rs = r1res_rli.second;
                Rings additional;
                Residue *r1res;
                if (!cross_residue)
                    r1res = r1res_rli.first;
                for (Rings::const_iterator r1i = rs.begin();
                r1i != rs.end(); ++r1i) {
                    const Ring &r1 = *r1i;
                    if (cross_residue)
                        r1res = (*r1.bonds().begin())
                        ->atoms()[0]->residue();
                          // more efficient to access
                          // bonds in rings than atoms
                    Rings::const_iterator r2i, end;
                    if (new_added.size() == 0) {
                        // first time through,
                        // sum rings with themselves
                        r2i = r1i;
                        r2i++;
                        end = rs.end();
                    } else {
                        // on following passes, only
                        // sum rings with newly added
                        // rings
                        r2i = new_added.begin();
                        end = new_added.end();
                    }
                    for (; r2i != end; ++r2i) {
                        const Ring &r2 = *r2i;
                        Residue *r2res =
                          (*r2.bonds().begin())->
                          atoms()[0]->residue();
                        if (!cross_residue
                          && r1res != r2res)
                            continue;

                        Ring::Bonds sum;
                        std::set_symmetric_difference(
                          r1.bonds().begin(),
                          r1.bonds().end(),
                          r2.bonds().begin(),
                          r2.bonds().end(),
                          std::insert_iterator<
                          Ring::Bonds>(
                          sum, sum.begin()));
                        
                        if (sum.size() == 0 ||
                          sum.size() > all_size_threshold
                          || sum.size() ==
                          r1.bonds().size()
                          + r2.bonds().size())
                            continue;
                        
                        Ring candidate(sum);
                        if (rs.find(candidate) != rs.end())
                            // not new
                            continue;

                        // make sure new "ring" is not
                        // disjoint, and is not bridged
                        Bond *cur_bond = *candidate.bonds().begin();
                        Atom *start_atom = cur_bond->atoms()[0];
                        unsigned num_bonds = 1;
                        Atom *cur_atom = cur_bond->atoms()[1];
                        while (cur_atom != start_atom) {
                            Ring::Bonds intersection;
                            std::SET<Bond *> atom_bonds;
                            for (auto b: cur_atom->bonds()) {
                                atom_bonds.insert(b);
                            }
                            std::set_intersection(atom_bonds.begin(),
                              atom_bonds.end(), sum.begin(), sum.end(),
                              std::insert_iterator <Ring::Bonds>(intersection,
                              intersection.begin()));
                            if (intersection.size() != 2)
                                break;
                            for (auto b: intersection) {
                                if (b == cur_bond)
                                    continue;
                                cur_bond = b;
                                num_bonds++;
                                cur_atom = cur_bond->other_atom(cur_atom);
                                break;
                            }
                        }
                        if (cur_atom != start_atom || num_bonds != sum.size())
                            continue;

                        additional.insert(candidate);
                        rings_added = true;
                    }
                }
                rs.insert(additional.begin(), additional.end());
                ring_lists[r1res] = rs;
                all_additional.insert(additional.begin(), additional.end());

            }
            for (auto aa: all_additional) {
                _rings.insert(aa);
            }
            new_added.swap(all_additional);
        }
    }
}

static Ring::Bonds::iterator
contains_exactly_one(const Ring &ring, Ring::Bonds *bond_set)
{
    const Ring::Bonds &bonds = ring.bonds();

    Ring::Bonds::iterator first = std::find_first_of(
        bond_set->begin(), bond_set->end(), bonds.begin(), bonds.end());

    if (first == bond_set->end())
        return bond_set->end();
    
    // ring contains one bond, see if it contains another
    Ring::Bonds::iterator found = first;
    first++;
    Ring::Bonds::iterator second = std::find_first_of(
        first, bond_set->end(), bonds.begin(), bonds.end());

    if (second == bond_set->end())
        // no others; exactly one found
        return found;
    
    return bond_set->end();
}

} // namespace atomstruct
