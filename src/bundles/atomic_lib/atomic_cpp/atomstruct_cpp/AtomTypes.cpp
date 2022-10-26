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

#include <logger/logger.h>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "AtomicStructure.h"
#include "Bond.h"
#include "Coord.h"
#include "PBGroup.h"
#include "Residue.h"
#include "tmpl/TAexcept.h"

namespace atomstruct {

enum BondOrder { AMBIGUOUS, SINGLE, DOUBLE }; // need SINGLE==1, DOUBLE==2

static int free_oxygens(std::vector<Atom*>&,
                    std::map<Atom*, int>&, bool = false);
static bool aromatic_geometry(const Ring&);
static void make_assignments(std::set<Bond*>&,
    std::map<Bond*, BondOrder>&, std::map<Bond*, int>&,
    std::vector<std::map<Bond*, int>>*, bool allow_charged=false);
static bool is_N2plus(std::map<Bond*, int>*, const Atom::Bonds&);
static bool is_N3plus_okay(const std::vector<Atom*>&);
static void invert_uncertains(std::vector<Atom*>& uncertain,
    std::map<Atom*, Bond*>& uncertain2bond,
    std::map<Bond*, BondOrder>* connected);
static void uncertain_assign(std::vector<Atom*>& uncertain,
    std::map<Atom*, Bond*>& uncertain2bond,
    std::set<Bond*>& bonds, std::map<Bond*, BondOrder>& connected,
    std::vector<std::map<Bond*, int>>* assignments,
    std::vector<std::vector<Atom*>>* assigned_uncertains,
    bool allow_charged=false);
static void flip_assign(std::vector<Bond*>& flippable, std::set<Atom*>& atoms,
    std::set<Bond*>& bonds, std::map<Bond*, BondOrder>& connected,
    std::vector<std::map<Bond*, int>>* assignments,
    bool allow_charged=false);


template <class Item>
void
generate_permutations(std::vector<Item*>& items,
                std::vector<std::vector<Item*>>* permutations)
{
    std::vector<typename std::vector<Item*> > lastGen;
    std::vector<typename std::vector<Item*>::iterator> lastRems;
    //for (typename std::vector<Item *>::iterator ii = items.begin();
    for (auto ii = items.begin(); ii != items.end(); ++ii) {
        Item *i = *ii;
        std::vector<Item *> itemList;
        itemList.push_back(i);
        lastGen.push_back(itemList);
        lastRems.push_back(ii+1);
    }
    permutations->insert(permutations->end(), lastGen.begin(), lastGen.end());

    for (unsigned int i = 2; i <= items.size(); ++i) {
        std::vector<std::vector<Item *> > gen;
        std::vector<typename std::vector<Item *>::iterator> rems;
        auto gi = lastGen.begin();
        auto ri = lastRems.begin();
        for (; gi != lastGen.end(); ++gi, ++ri) {
            for (auto ii = *ri; ii != items.end(); ++ii) {
                auto perm = *gi;
                perm.push_back(*ii);
                gen.push_back(perm);
                rems.push_back(ii+1);
            }
        }
        permutations->insert(permutations->end(), gen.begin(), gen.end());
        lastGen.swap(gen);
        lastRems.swap(rems);
    }
}

static bool
is_N2plus(std::map<Bond*, int>* bestAssignment, const Atom::Bonds& bonds)
{
    int sum = 0, target = 4;
    for (auto b: bonds) {
        auto bai = bestAssignment->find(b);
        if (bai == bestAssignment->end())
            target -= 1;
        else
            sum += bai->second;
    }
    return sum == target;
}

static bool
is_Oarplus(std::map<Bond*, int>* bestAssignment, const Atom::Bonds& bonds)
{
    int sum = 0, target = 3;
    for (auto b: bonds) {
        auto bai = bestAssignment->find(b);
        if (bai == bestAssignment->end())
            return false;
        else
            sum += bai->second;
    }
    return sum == target;
}

static bool
is_N3plus_okay(const std::vector<Atom*>& neighbors)
{
    for (auto bondee: neighbors) {
        auto bondee_type = bondee->idatm_type();

        if (bondee_type != "C3" && bondee_type != "H" && bondee_type != "D") {
            return false;
        }
    }
    return true;
}

static void
make_assignments(std::set<Bond*>& bonds, std::map<Bond*,
    BondOrder>& connected, std::map<Bond*,int>& cur_assign,
    std::vector<std::map<Bond*, int>>* assignments, bool allow_charged)
{
    Bond* assign_target = *(bonds.begin());
    bonds.erase(bonds.begin());
    bool assign1okay = true, assign2okay = true;
    // see if this assignment completes the bonds of either connected
    // atom and which assignments work
    for (auto end: assign_target->atoms()) {
        bool complete = true;
        int sum = 0;
        auto& atom_bonds = end->bonds();
        // implied proton treated the same as ambiguous non-ring bond
        bool has_ambiguous = atom_bonds.size() == 2;
        for (auto b: atom_bonds) {
            if (b == assign_target)
                continue;
            if (bonds.find(b) != bonds.end()) {
                complete = false;
                break;
            }
            if (cur_assign.find(b) != cur_assign.end()) {
                sum += cur_assign[b];
                continue;
            }
            if (connected[b] == AMBIGUOUS) {
                sum++;
                has_ambiguous = true;
            } else {
                sum += connected[b];
            }
            if (connected[b] == DOUBLE)
                assign2okay = false;
        }
        if (atom_bonds.size() == 2) {
            if (sum == 2)
                assign2okay = false;
        } else {
            if (sum > 3) {
                assign1okay = assign2okay = false;
                break;
            }
            if (sum == 3)
                assign2okay = false;
        }
        if (!complete)
            continue;
        int element = end->element().number();
        if (element > 20)
            continue;
        int valence_electrons = (element - 2) % 8;
        if (valence_electrons >= 4) {
            int charge_mod = 0;
            valence_electrons += sum;
            if (allow_charged) {
                if (element == 7 && atom_bonds.size() == 3) {
                    charge_mod = 1;
                    has_ambiguous = true;
                } else if (element == 8 && atom_bonds.size() == 2) {
                    charge_mod = 1;
                    has_ambiguous = true;
                }
            }
            if (has_ambiguous) {
                if (valence_electrons < 6
                || valence_electrons > 7 + charge_mod)
                    assign1okay = false;
                if (valence_electrons < 5
                || valence_electrons > 6 + charge_mod)
                    assign2okay = false;
            } else {
                if (valence_electrons != 7)
                    assign1okay = false;
                if (valence_electrons != 6)
                    assign2okay = false;
            }
        } else {
            valence_electrons -= sum;
            if (has_ambiguous) {
                if (valence_electrons < 1
                || valence_electrons > 2)
                    assign1okay = false;
                if (valence_electrons < 2
                || valence_electrons > 3)
                    assign2okay = false;
            } else {
                if (valence_electrons != 1)
                    assign1okay = false;
                if (valence_electrons != 2)
                    assign2okay = false;
            }
        }
    }
    if (assign1okay) {
        cur_assign[assign_target] = 1;
        if (bonds.size() > 0) {
            make_assignments(bonds, connected, cur_assign,
                        assignments, allow_charged);
        } else {
            assignments->push_back(cur_assign);
        }
        cur_assign.erase(assign_target);
    }
    if (assign2okay) {
        cur_assign[assign_target] = 2;
        if (bonds.size() > 0) {
            make_assignments(bonds, connected, cur_assign,
                        assignments, allow_charged);
        } else {
            assignments->push_back(cur_assign);
        }
        cur_assign.erase(assign_target);
    }
    bonds.insert(assign_target);
}

static std::map<Bond*, int>*
find_best_assignment(std::vector<std::map<Bond*, int>>& assignments,
                    std::vector<const Ring*>& system_rings)
{
    if (assignments.size() == 0)
        return NULL;
    if (assignments.size() == 1)
        return &assignments[0];

    // prefer aromatic if possible (and avoid anti-aromatic)
    std::set<int> okay_assignments;
    std::map<int, int> num_plus;
    int best_aro = 0;
    for (unsigned int i = 0; i < assignments.size(); ++i) {
        std::map<Bond*, int>& assignment = assignments[i];
        std::map<Atom*, int> sum_orders, sum_bonds;
        for (auto b_order: assignment) {
            Bond* b = b_order.first;
            int order = b_order.second;
            for (auto a: b->atoms()) {
                sum_orders[a] += order;
                sum_bonds[a]++;
            }
        }
        // account for N2+
        for (auto a_sum: sum_orders) {
            Atom* a = a_sum.first;
            if (a->element() == Element::N) {
                if (is_N2plus(&assignment, a->bonds()))
                    num_plus[i] += 1;
            } else if (a->element() == Element::O) {
                if (is_Oarplus(&assignment, a->bonds()))
                    num_plus[i] += 1;
            }
        }
        int num_aro = 0;
        for (auto ring: system_rings) {
            int pi_electrons = 0;
            for (auto a: ring->atoms()) {
                int sum = sum_orders[a];
                int element = a->element().number();
                if (element > 20) {
                    // guessing not aromatic
                    pi_electrons = 0;
                    break;
                }
                int valence_electrons = (element - 2) % 8;
                if (valence_electrons < 3 || valence_electrons > 6) {
                    // aromatic never possible
                    pi_electrons = 0;
                    break;
                }
                if (valence_electrons == 4) {
                    if (sum == 2 && sum_bonds[a] == 2) {
                        // other bond is double
                        pi_electrons = 0;
                        break;
                    }
                    pi_electrons++;
                } else if (valence_electrons > 4) {
                    if (valence_electrons == 5 && sum != 3 && ring->size() == 5) {
                        // nitrogen needs to feel 3 bonds (see residue YG in 6tna)
                        pi_electrons = 0;
                        break;
                    }
                    pi_electrons += 2 - (sum != 2);
                } else if (a->bonds().size() == 2 && sum == 2) {
                    pi_electrons++;
                } else {
                    // aromatic not possible with this assignment
                    pi_electrons = 0;
                    break;
                }
            }
            if (pi_electrons % 4 == 2)
                num_aro += ring->atoms().size();
        }
        if (num_aro > 0 && num_aro >= best_aro) {
            if (num_aro > best_aro)
                okay_assignments.clear();
            okay_assignments.insert(i);
            best_aro = num_aro;
        }
    }
    if (okay_assignments.size() == 1)
        return &assignments[*okay_assignments.begin()];

    // lowest charge preferred
    std::set<int> next_okay;
    int low_charge = -1;
    std::vector<std::map<Bond *, int> >::iterator best_assignment, ai;
    for (ai=assignments.begin(); ai != assignments.end(); ++ai) {
        int index = ai - assignments.begin();
        if (okay_assignments.size() > 0
        && okay_assignments.find(index) == okay_assignments.end())
            continue;

        int cur_charge = num_plus[index];
        if (low_charge == -1 || cur_charge < low_charge) {
            next_okay.clear();
            next_okay.insert(index);
            low_charge = cur_charge;
        } else if (cur_charge == low_charge)
            next_okay.insert(index);
    }
    okay_assignments.swap(next_okay);
    if (okay_assignments.size() == 1)
        return &assignments[*okay_assignments.begin()];

    // evaluate by best fit to bond lengths
    float best_val = 0.0;
    for (ai=assignments.begin(); ai != assignments.end(); ++ai) {
        if (okay_assignments.size() > 0 && okay_assignments.find(
        ai - assignments.begin()) == okay_assignments.end())
            continue;
        float val = 0.0;
        std::map<Bond *, int>& assignment = *ai;
        int order_sum = 0;
        for (auto b_order: assignment) {
            val += b_order.first->sqlength() * b_order.second;
            order_sum += b_order.second;
        }
        val /= order_sum;
        if (best_val == 0.0 || val < best_val) {
            best_val = val;
            best_assignment = ai;
        }
    }
    return &(*best_assignment);
}

static int
free_oxygens(std::vector<Atom*>& neighbors,
    std::map<Atom*, int>& heavys, bool no_hyds)
{
    int free_oxygens = 0;
    for (auto bondee: neighbors) {
        if (bondee->element() == Element::O) {
            if (heavys[bondee] != 1)
                continue;
            if (no_hyds && bondee->neighbors().size() != 1)
                continue;
            free_oxygens++;
        }
    }

    return free_oxygens;
}

static bool
aromatic_geometry(const Ring& r)
{
    // algorithm from:
    //  Crystallographic Studies of Inter- and Intramolecular 
    //     Interactions Reflected in Aromatic Character of pi-Electron
    //     Systems
    //  J. Chem. Inf. Comput. Sci., 1993, 33, 70-78
    Real sum = 0.0;
    int bonds = 0;
    for (auto b: r.bonds()) {
        const Element& e1 = b->atoms()[0]->element();
        const Element& e2 = b->atoms()[1]->element();
        Coord c1 = b->atoms()[0]->coord();
        Coord c2 = b->atoms()[1]->coord();
        Real d = c1.distance(c2), delta;

        if (e1 == Element::C && e2 == Element::C) {
            delta = d - 1.38586;
        } else if ((e1 == Element::C || e2 == Element::C) &&
          (e1 == Element::N || e2 == Element::N)) {
            delta = d - 1.34148;
        } else
            continue;
        bonds++;
        sum += delta * delta;

    }
    if (bonds == 0)
        return false;
    
    Real homa = 1.0 - (792.0 * sum / bonds);

    if (homa >= 0.5) {
        // aromatic
        return true;
    } else if (bonds * homa < -35.0)
        return false;

    return true;
}

#ifdef REPORT_TIME
#include <ctime>
#endif
void
AtomicStructure::_compute_atom_types()
{
#ifdef REPORT_TIME
clock_t start_t = clock();
#endif

    // angle values used to discriminate between hybridization states
    const Real angle23val1 = 115.0;
    const Real angle23val1_tmax = 116.5;
    const Real angle23val1_tmin = 113.5;
    const Real angle23val2 = 122.0;
    const Real angle12val = 160.0;

    // bond length cutoffs from hybridization discrimination
    // p3... = pass 3 cutoffs; p4... = pass 4 cutoffs
    const Real p3c1c1 = 1.22 * 1.22;
    const Real p3c2c = 1.41 * 1.41;
    const Real p3c2n = 1.37 * 1.37;
    const Real p3n1c1 = 1.20 * 1.20;
    const Real p3n3c = 1.38 * 1.38;
    const Real p3n3n3 = 1.43 * 1.43;
    const Real p3n3n2 = 1.41 * 1.41;
    const Real p3n1o1 = 1.21 * 1.21;
    const Real p3c1o1 = 1.17 * 1.17;
    const Real p3o2c2 = 1.30 * 1.30;
    const Real p3o2as = 1.685 * 1.685;
    const Real p3o2o3 = 1.338 * 1.338;
    const Real p3s2c2 = 1.76 * 1.76;
    const Real p3s2as = 2.11 * 2.11;
    const Real p4c3c = 1.53 * 1.53;
    const Real p4c3n = 1.46 * 1.46;
    const Real p4c3o = 1.44 * 1.44;
    const Real p4n2c = 1.38 * 1.38;
    const Real p4n2n = 1.32 * 1.32;
    const Real p4c2c = 1.42 * 1.42;
    const Real p4c2n = 1.41 * 1.41;
    const Real p4ccnd = 1.45 * 1.45;

    const Real p7cn2nh = 1.3629;
    const Real p7nn2nh = 1.3337;
    const Real p7on2nh = 1.3485;

#ifdef TIME_PASSES
clock_t t0 = clock();
#endif
    // algorithm based on E.C. Meng / R.A. Lewis paper 
    // "Determination of Molecular Topology and Atomic Hybridization
    // States from Heavy Atom Coordinates", J. Comp. Chem., v12#7, 891-898
    // and on example code from idatm.f implementation by E.C. Meng

    // differences: No boron types.  Double-bonded Npls are split off
    //   as N2.  Sox split into Sxd (sulfoxide), and Son (sulfone).
    //   Carbons in aromatic rings are type Car.  Aromatic oxygens are Oar/Oar+.
    //   Negatively charged oxygens are O2- (planar) and O3- (tetrahedral)
    //   instead of O-.  Sp nitrogens bonded to two atoms are N1+.
    
    // Suspend atom-type change notifications, since they all change, and
    // ensure the suspension ends when this routine exits
    class SuspendNotification {
        AtomicStructure *as;
        public:
            SuspendNotification(AtomicStructure *s) { as = s; s->_atom_types_notify = false; }
            ~SuspendNotification() { as->_atom_types_notify = true; }
    };
    SuspendNotification suspender(this);

    const Atom::IdatmInfoMap& info_map = Atom::get_idatm_info_map();
#ifdef TIME_PASSES
clock_t t1 = clock();
std::cerr << "setup (fetch IDATM info map) took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // initialize idatm type in Atoms
    for (auto& a: atoms()) {
        a->set_computed_idatm_type(a->element().name());
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "setup (setting initial types) took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // if molecule is diamond/nanotube, skip atom typing since the
    // ring finding will take forever
    int num_bonds = this->num_bonds();
    size_t num_atoms = this->num_atoms();
    if (num_bonds - num_atoms > 100 && num_bonds / (float) num_atoms > 1.25)
        return;
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "setup (check nanotube) took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif


    // "pass 0.5": use templates for "infallible" typing of standard
    // residue types
    std::vector<Atom*> untyped_atoms;
    bool all_unassigned_are_H = true;
    std::vector<const Residue*> templated_residues;
    for (auto& r: residues()) {
        try {
            // Don't template-type residues with unexpected cross-residue 
            // bonds (e.g. residues bonded to PTD in 3kch)

            if (r->polymer_type() != PT_NONE) {
                for (auto ra: r->atoms()) {
                    if (ra->is_backbone(BBE_MIN))
                        continue;
                    for (auto nb: ra->neighbors()) {
                        if (nb->residue() != ra->residue()) {
                            // cysteine SG cross-residue bond okay...
                            if (nb->name() == "SG" && ra->name() == "SG"
                            && (nb->residue()->name() == "CYS" || nb->residue()->name() == "CYX")
                            && (ra->residue()->name() == "CYS" || ra->residue()->name() == "CYX"))
                                continue;
                            throw tmpl::TA_NoTemplate("Non-standard cross-residue bond");
                        }
                    }
                }
            }
            auto untemplated_atoms = r->template_assign(
                &Atom::set_computed_idatm_type,
                "idatm", "templates", "idatmres");
            templated_residues.push_back(r);
            if (untemplated_atoms.empty())
                continue;
            for (auto ra: untemplated_atoms) {
                if (ra->element().number() == 1) {
                    // type now so that we needn't (relatively slowly) loop through
                    // a large number of hydrogens later
                    bool bonded_to_carbon = false;
                    for (auto bondee: ra->neighbors()) {
                        if (bondee->element() == Element::C) {
                            bonded_to_carbon = true;
                            break;
                        }
                    }
                    ra->set_computed_idatm_type(bonded_to_carbon ?  "HC" : "H");
                } else {
                    untyped_atoms.push_back(ra);
                    all_unassigned_are_H = false;
                }
            }
        } catch (tmpl::TA_NoTemplate&) {
            for (auto ra: r->atoms()) {
                untyped_atoms.push_back(ra);
                if (ra->element().number() != 1)
                    all_unassigned_are_H = false;
            }
        } catch (...) {
            throw;
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 0.5 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif
    if (untyped_atoms.empty())
        return;     // All atoms assigned.

    // "pass 1":  type hydrogens / deuteriums and compute number of
    // heavy atoms connected to each atom
    std::map<Atom*, int> heavys; // number of heavy atoms bonded
    for (auto a: untyped_atoms) {
        const Element &element = a->element();

        if (element.number() == 1) {
            // sort out if it's a hydrogen or deuterium
            bool is_hyd = true;
            for (auto c: a->name()) {
                if (isalpha(c)) {
                    if (c == 'd' || c == 'D') {
                        is_hyd = false;
                    }
                    break;
                }
            }

            bool bonded_to_carbon = false;
            for (auto bondee: a->neighbors()) {
                if (bondee->element() == Element::C) {
                    bonded_to_carbon = true;
                    break;
                }
            }
              
            
            a->set_computed_idatm_type(bonded_to_carbon ? (is_hyd ?
              "HC" : "DC") : (is_hyd ? "H" : "D"));
        } else {
            int heavy_count = 0;
            for (auto bondee: a->neighbors()) {
                if (bondee->element().number() > 1) {
                    heavy_count++;
                }
            }
            heavys[a] = heavy_count;
        }
    }

    if (all_unassigned_are_H)
        return;

    untyped_atoms.clear();
    for (auto h_n: heavys)
        untyped_atoms.push_back(h_n.first);
    std::set<const Atom*>
        untyped_set(untyped_atoms.begin(), untyped_atoms.end());
    // since we need the heavy count for possibly-typed neighbor atoms in pass 5,
    // get the heavy count for all atoms
    for (auto a: atoms()) {
        if (heavys.find(a) == heavys.end()) {
            int heavy_count = 0;
            for (auto bondee: a->neighbors()) {
                if (bondee->element().number() > 1) {
                    heavy_count++;
                }
            }
            heavys[a] = heavy_count;
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 1 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 2": elements that are typed only by element type
    // and valences > 1
    std::map<Atom*, int> redo;
    std::set<Atom*> ambiguous_val2Cs;
    for (auto a: untyped_atoms) {
        const Element &element = a->element();

        // undifferentiated types
        if ((element >= Element::He && element <= Element::Be)
        || (element >= Element::Ne && element <= Element::Si)
        || element >= Element::Cl) {
            a->set_computed_idatm_type(element.name());
            continue;
        }

        // valence 4
        //  C must be sp3 (C3)
        //  N must be part of a quaternary amine (N3+) 
        //  P must be part of a phosphate (Pac), a P-oxide (Pox)
        //      or a quaternary phosphine (P3+)
        //  S must be part of a sulfate, sulfonate or sulfamate
        //      (Sac), or sulfone (Son)
        auto neighbors = a->neighbors();
        if (neighbors.size() == 4) {
            if (element == Element::C) {
                a->set_computed_idatm_type("C3");
            } else if (element == Element::N) {
                a->set_computed_idatm_type("N3+");
            } else if (element == Element::P) {
                int freeOxys = free_oxygens(neighbors, heavys);
                if (freeOxys >= 2)
                    a->set_computed_idatm_type("Pac");
                else if (freeOxys == 1)
                    a->set_computed_idatm_type("Pox");
                else
                    a->set_computed_idatm_type("P3+");
            } else if (element == Element::S) {
                int freeOxys = free_oxygens(neighbors, heavys);
                if (freeOxys >= 3) {
                    a->set_computed_idatm_type("Sac");
                } else if (freeOxys >= 1) {
                    a->set_computed_idatm_type("Son");
                } else {
                    a->set_computed_idatm_type("S");
                }
            }
        }

        // valence 3
        // calculate the three bond angles and average them;
        // since hydrogens may be missing, cannot count on valence
        // to determine the hybridization state.  Average bond angle
        // assists in discriminating hybridization
        //  C may be sp3 (C3), sp2 (C2), or part of a carboxylate
        //      (Cac)
        //  N may be sp3 (N3), sp2, or planar (as in amides and
        //      aniline deriviatives), or part of a nitro
        //      group (Ntr)
        //  S may be, depending on oxidation state, sulfoxide (Sxd)
        //      or S3+
        else if (neighbors.size() == 3) {
            Real avg_angle = 0.0;
            for (int n1 = 0; n1 < 3; ++n1) {
                for (int n2 = n1 + 1; n2 < 3; ++n2) {
                    avg_angle += a->coord().angle(neighbors[n1]->coord(),
                      neighbors[n2]->coord());
                }
            }
            avg_angle /= 3.0;

            if (element == Element::C) {
                bool c3 = false;
                if (avg_angle < angle23val1_tmin)
                    c3 = true;
                else if (avg_angle < angle23val1_tmax) {
                    Real min_sqdist = -1.0;
                    for (int n1 = 0; n1 < 3; ++n1) {
                        Real sqd = a->coord().sqdistance(neighbors[n1]->coord());
                        if (min_sqdist < 0.0 || sqd < min_sqdist)
                            min_sqdist = sqd;
                    }
                    if (min_sqdist > p4c3c)
                        c3 = true;
                    else if (min_sqdist > p4c2c && avg_angle < angle23val1)
                        c3 = true;
                }
                if (c3)
                    a->set_computed_idatm_type("C3");
                else
                    a->set_computed_idatm_type(free_oxygens(
                        neighbors, heavys, true) >= 2 ? "Cac" : "C2");
            } else if (element == Element::N) {
                if (avg_angle < angle23val1)
                    a->set_computed_idatm_type("N3");
                else
                    a->set_computed_idatm_type(free_oxygens(
                      neighbors, heavys) >= 2 ? "Ntr":"Npl");
            } else if (element == Element::S) {
                bool has_oxy = false;
                for (int i = 0; i < 3; ++i) {
                    if (neighbors[i]->element() == Element::O) {
                        has_oxy = true;
                        break;
                    }
                }
                a->set_computed_idatm_type(has_oxy ? "Sxd" : "S3+");
            }
        }

        // valence 2
        // calculate the bond angle and assign a tentative atom
        // type accordingly (a single angle is often not a good
        // indicator of type).  Mark these atoms for further
        // analysis by putting a non-zero value for them in the
        // 'redo' array.
        //  C may be sp3 (C3), sp2 (C2), or sp (C1)
        //  N may be sp3 (N3), sp2 or planar (Npl), or sp (N1+)
        //  O and S are sp3 (O3 and S3, respectively)
        else if (neighbors.size() == 2) {
            Point coord[2];
            int coord_ind = 0;
            for (std::vector<Atom *>::const_iterator bi =
              neighbors.begin(); bi != neighbors.end(); ++bi) {
                Atom *other = *bi;
                coord[coord_ind++] = other->coord();
            }
            Real ang = a->coord().angle(coord[0], coord[1]);

            if (element == Element::C) {
                if (ang < angle23val1) {
                    a->set_computed_idatm_type("C3");
                    redo[a] = 1;
                    if (ang > angle23val1_tmin)
                        ambiguous_val2Cs.insert(a);
                } else if (ang < angle12val) {
                    a->set_computed_idatm_type("C2");
                    if (ang < angle23val2) {
                        redo[a] = 3;
                    } else {
                        // allow ring bond-order code
                        // to change this assignment
                        redo[a] = -1;
                    }
                    if (ang < angle23val1_tmax)
                        ambiguous_val2Cs.insert(a);
                } else {
                    a->set_computed_idatm_type("C1");
                }
            } else if (element == Element::N) {
                if (ang < angle23val1) {
                    a->set_computed_idatm_type("N3");
                    redo[a] = 2;
                } else {
                    a->set_computed_idatm_type(
                      ang < angle12val ?  "Npl" : "N1+");
                }
            } else if (element == Element::O) {
                a->set_computed_idatm_type("O3");
            } else if (element == Element::S) {
                a->set_computed_idatm_type("S3");
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 2 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 3": determine types of valence 1 atoms.  These were typed
    // by element only in previous pass, but can be typed more accurately
    // now that the atoms they are bonded to have been typed.  Bond
    // lengths are used in this pass.  
    for (auto a: untyped_atoms) {

        auto neighbors = a->neighbors();
        if (neighbors.size() != 1)
            continue;
        
        Atom *bondee = *(neighbors.begin());
        Real sqlen = bondee->coord().sqdistance(a->coord());
        auto bondee_type = bondee->idatm_type();

        
        if (a->idatm_type() == "C") {
            if ((sqlen <= p3c1c1 && bondee_type == "C1")
            || (sqlen <= p3n1c1 && bondee->element() == Element::N)) {
                a->set_computed_idatm_type("C1");
            } else if (sqlen <= p3c2c &&
              bondee->element() == Element::C) {
                a->set_computed_idatm_type("C2");
            } else if (sqlen <= p3c2n &&
              bondee->element() == Element::N) {
                a->set_computed_idatm_type("C2");
            } else if (sqlen <= p3c1o1 &&
              bondee->element() == Element::O &&
              bondee->neighbors().size() == 1) {
                a->set_computed_idatm_type("C1-");
            } else {
                a->set_computed_idatm_type("C3");
            }
        } else if (a->idatm_type() == "N") {
            if (((sqlen <= p3n1c1 && (bondee_type == "C1" || bondee->element() == Element::N)) ||
              bondee_type == "N1+") || (sqlen < p3n1o1 &&
              bondee->element() == Element::O)) {
                a->set_computed_idatm_type("N1");
            } else if (sqlen > p3n3c &&
              (bondee_type == "C2" || bondee_type == "C3")) {
                a->set_computed_idatm_type("N3");
            } else if ((sqlen > p3n3n3 && bondee_type == "N3") ||
              (sqlen > p3n3n2 && bondee_type == "Npl")) {
                a->set_computed_idatm_type("N3");
            } else if (bondee->element() == Element::C ||
              bondee->element() == Element::N) {
                a->set_computed_idatm_type("Npl");
            } else {
                a->set_computed_idatm_type("N3");
            }
        } else if (a->idatm_type() == "O") {
            if (bondee_type == "Cac" || bondee_type == "Ntr" ||
              bondee_type == "N1+") {
                    a->set_computed_idatm_type("O2-");
            } else if (bondee_type == "Pac" || bondee_type == "Sac"
              || bondee_type == "N3+" || bondee_type == "Pox"
              || bondee_type == "Son" || bondee_type == "Sxd") {
                a->set_computed_idatm_type("O3-");

                // pKa of 3rd phosphate oxygen is 7...
                if (bondee_type != "Pac")
                    continue;
                int oxys = 0;
                for (auto bp: bondee->neighbors()) {
                    if (bp->element() == Element::O
                    && bp->neighbors().size() == 1)
                        oxys += 1;
                }
                if (oxys < 3)
                    continue;
                // if this bond is 0.05 A longer than
                // the other P-O bonds, assume OH
                Real len = bondee->coord().distance(a->coord());
                bool longer = true;
                for (auto bp: bondee->neighbors()) {
                    if (bp == a || bp->neighbors().size() > 1
                    || bp->element() != Element::O)
                        continue;
                    if (len < bondee->coord().distance(bp->coord()) + 0.05) {
                        longer = false;
                        break;
                    }
                }
                if (longer)
                    a->set_computed_idatm_type("O3");
            } else if (sqlen <= p3c1o1 &&
              bondee->element() == Element::C &&
              bondee->neighbors().size() == 1) {
                    a->set_computed_idatm_type("O1+");
            } else if (sqlen <= p3o2c2 &&
              bondee->element() == Element::C) {
                    a->set_computed_idatm_type("O2");
                if (untyped_set.find(bondee) != untyped_set.end())
                    bondee->set_computed_idatm_type("C2");
                redo[bondee] = 0;
            } else if (sqlen <= p3o2as &&
              bondee->element() == Element::As) {
                    a->set_computed_idatm_type("O2");
            } else if (sqlen <= p3o2o3 &&
              bondee->element() == Element::O &&
              bondee->neighbors().size() == 1) {
                // distinguish oxygen molecule from
                // hydrogen peroxide
                    a->set_computed_idatm_type("O2");
            } else if (sqlen <= p3n1o1 &&
              bondee->element() == Element::N &&
              bondee->neighbors().size() == 1) {
                    a->set_computed_idatm_type("O1");
            } else {
                    a->set_computed_idatm_type("O3");
            }
        } else if (a->idatm_type() == "S") {
            if (bondee->element() == Element::P) {
                    a->set_computed_idatm_type("S3-");
            } else if (bondee_type == "N1+") {
                    a->set_computed_idatm_type("S2");
            } else if (sqlen <= p3s2c2 &&
              bondee->element() == Element::C) {
                    a->set_computed_idatm_type("S2");
                if (untyped_set.find(bondee) != untyped_set.end())
                    bondee->set_computed_idatm_type("C2");
                redo[bondee] = 0;
            } else if (sqlen <= p3s2as &&
              bondee->element() == Element::As) {
                    a->set_computed_idatm_type("S2");
            } else {
                    a->set_computed_idatm_type("S3");
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 3 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 4": re-examine all atoms with non-zero 'redo' values and
    //   retype them if necessary
    for (auto a: untyped_atoms) {

        if (redo[a] == 0)
            continue;
        
        bool c3able = false;
        for (auto bondee: a->neighbors()) {
            const Element &bondee_element = bondee->element();
            Real sqlen = bondee->coord().sqdistance(a->coord());

            if (redo[a] == 1) {
                if ((sqlen <= p4c2c && bondee_element == Element::C)
                || (sqlen <= p4c2n && bondee_element == Element::N)) {
                    a->set_computed_idatm_type("C2");
                    break;
                }
                if ((sqlen > p4c3c && bondee_element == Element::C)
                || (sqlen > p4c3n && bondee_element == Element::N)
                || (sqlen > p4c3o && bondee_element == Element::O)) {
                    a->set_computed_idatm_type("C3");
                }
            } else if (redo[a] == 2) {
                if ((sqlen <= p4n2c && bondee_element == Element::C)
                || (sqlen <= p4n2n && bondee_element == Element::N)) {
                    // explicit hydrogen(s): N2
                    if (heavys[a] < 2)
                        a->set_computed_idatm_type("N2");
                    else
                        a->set_computed_idatm_type("Npl");
                    break;
                }
            } else {
                if ((sqlen <= p4c2c && bondee_element == Element::C)
                || (sqlen <= p4c2n && bondee_element == Element::N)) {
                    a->set_computed_idatm_type("C2");
                    c3able = false;
                    break;
                }
                if ((sqlen > p4c3c && bondee_element == Element::C)
                || (sqlen > p4c3n && bondee_element == Element::N)
                || (sqlen > p4c3o && bondee_element == Element::O)) {
                    c3able = true;
                }
                if (sqlen > p4ccnd && bondee_element == Element::C) {
                    c3able = true;
                }
            }
        }
        if (c3able)
            a->set_computed_idatm_type("C3");
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 4 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 4.5":  this pass is not in the IDATM paper but is a suggested
    //    improvement mentioned on page 897 of the paper:  find aromatic
    //    ring types.  The method is to:
    //
    //  1) Find all intraresidue rings (actually computed before pass 6)
    //  2) Check that all the atoms of the ring are planar types
    //  3) Check bond lengths around the ring; see if they are
    //      consistent with aromatic bond lengths
    std::set<const Residue*>
        mapped_residues(templated_residues.begin(), templated_residues.end());
    size_t too_many_rings = (residues().size() - mapped_residues.size()) * 20;
    int ring_limit = 3;
    Rings try_rings;
    _temporary_per_residue_rings(try_rings, ring_limit, &mapped_residues);
    if (try_rings.size() < too_many_rings) {
        // not something crazy like an averaged structure...
        ring_limit = 6;
        _temporary_per_residue_rings(try_rings, ring_limit, &mapped_residues);
        if (try_rings.size() < too_many_rings) {
            // not something crazy like a nanotube...
            ring_limit = 0;
        }
    }
    // try_rings is a copy, we want a reference...
    const Rings& rs = rings(false, ring_limit, &mapped_residues);
    // screen out rings with definite non-planar types
    std::set<const Ring*> planar_rings;
    for (auto& r: rs) {
        if (r.atoms().size() == 3) {
            for (auto a: r.atoms()) {
                if (a->element() == Element::C)
                    a->set_computed_idatm_type("C3");
            }
            continue;
        }
        bool planar_types = true;
        bool all_planar = true;
        int num_oxygens = 0;
        for (auto a: r.atoms()) {
            auto idatm_type = a->idatm_type();
            if (a->element() == Element::O)
                num_oxygens++;

            auto neighbors = a->neighbors();
            if (neighbors.size() > 3) {
                all_planar = planar_types = false;
                break;
            }

            if (idatm_type != "C2" && idatm_type != "Npl"
            && idatm_type != "Sar" && idatm_type != "O3"
            && idatm_type != "S3" && idatm_type != "N3"
            && idatm_type != "Oar" && idatm_type != "Oar+"
            && idatm_type != "P" && idatm_type != "Car"
            && idatm_type != "N2" && idatm_type != "N2+"
            && !(idatm_type == "C3" && neighbors.size() == 2)) {
                all_planar = planar_types = false;
                break;
            } else if (idatm_type == "O3" || idatm_type == "S3"
            || idatm_type == "N3" || idatm_type == "C3") {
                all_planar = false;
            }

        }

        if (!planar_types)
            continue;
        
        if (r.atoms().size() == 5 && num_oxygens > 1 && num_oxygens < 5)
            continue;
        if (all_planar || aromatic_geometry(r))
            planar_rings.insert(&r);
    }
    // find ring systems
    std::set<const Ring*> seen_rings;
    std::vector<std::set<Bond*> > fused_bonds;
    std::vector<std::set<Atom*> > fused_atoms;
    std::vector<std::vector<const Ring*> > component_rings;
    std::set<Atom *> ring_assigned_Ns;
    for (auto r: planar_rings) {
        if (seen_rings.find(r) != seen_rings.end())
            continue;
        std::set<Bond*> system_bonds;
        std::set<Atom*> system_atoms;
        std::vector<const Ring*> system_rings;
        std::vector<const Ring*> queue;
        queue.push_back(r);
        seen_rings.insert(r);
        while (queue.size() > 0) {
            const Ring* qr = queue.back();
            queue.pop_back();
            const Ring::Bonds &bonds = qr->bonds();
            const Ring::Atoms &atoms = qr->atoms();
            system_bonds.insert(bonds.begin(), bonds.end());
            system_atoms.insert(atoms.begin(), atoms.end());
            system_rings.push_back(qr);
            for (auto b: bonds) {
                for (auto br: b->rings(false, ring_limit, &mapped_residues)) {
                    if (seen_rings.find(br) != seen_rings.end())

                        continue;
                    if (planar_rings.find(br) == planar_rings.end())
                        continue;

                    queue.push_back(br);
                    seen_rings.insert(br);
                }
            }
        }
        fused_bonds.push_back(system_bonds);
        fused_atoms.push_back(system_atoms);
        component_rings.push_back(system_rings);
    }
        
    for (unsigned int i = 0; i < fused_bonds.size(); ++i) {
        std::set<Bond *> &bonds = fused_bonds[i];
        std::set<Atom *> &atoms = fused_atoms[i];
        std::vector<const Ring*> system_rings = component_rings[i];

        if (atoms.size() > 50) {
            // takes too long to do a massive fused-ring system;
            // assume aromatic
            for (auto fa: atoms) {
                if (fa->element() == Element::C) {
                    fa->set_computed_idatm_type("Car");
                } else if (fa->element() == Element::O) {
                    fa->set_computed_idatm_type("Oar");
                }
            }
            continue;
        }


        // find bonds directly connected to rings
        // and try to judge their order
        std::map<Bond*, BondOrder> connected;
        std::set<std::pair<Atom*, Bond*> > ring_neighbors;
        std::vector<std::pair<Bond*, Atom*> > possibly_ambiguous;
        for (auto a: atoms) {
            auto nai = a->neighbors().begin();
            auto nbi = a->bonds().begin();
            for (; nai != a->neighbors().end(); ++nai, ++nbi) {
                auto n = *nai;
                if (atoms.find(n) != atoms.end())
                    continue;
                auto nb = *nbi;
                ring_neighbors.insert(std::pair<Atom*, Bond*>(n, nb));

                if (ambiguous_val2Cs.find(n) != ambiguous_val2Cs.end()) {
                    connected[nb] = AMBIGUOUS;
                    continue;
                }
                Atom::IdatmInfoMap::const_iterator gi =
                    info_map.find(n->idatm_type());
                if (gi == info_map.end()) {
                    connected[nb] = SINGLE;
                    continue;
                }
                int geom = (*gi).second.geometry;
                if (geom != 3) {
                    connected[nb] = SINGLE;
                    if (geom == 4 && n->neighbors().size() == 1) {
                        possibly_ambiguous.push_back(
                            std::pair<Bond *, Atom*>(nb, n));
                    }
                    continue;
                }
                if (n->element() == Element::N) {
                    // aniline can be planar
                    connected[nb] = SINGLE;
                    continue;
                }
                // look at neighbors (and grandneighbors)
                bool outside_double = false;
                bool ambiguous = (redo[n] == -1);
                const std::vector<Atom*> nn = n->neighbors();
                if (nn.size() == 1) {
                    if (a->element() == Element::N
                    && n->element() == Element::O)
                        connected[nb] = SINGLE;
                    else {
                        connected[nb] = DOUBLE;
                        possibly_ambiguous.push_back(
                            std::pair<Bond*, Atom*>(nb, n));
                    }
                    continue;
                }
                for (auto n2: nn) {
                    if (n2 == a)
                        continue;
                    gi = info_map.find(n2->idatm_type());
                    if (gi == info_map.end())
                        continue;
                    int n2geom = (*gi).second.geometry;
                    if (n2geom != 3)
                        continue;
                    bool all_single = true;
                    for (auto n3: n2->neighbors()) {
                        if (n3 == n)
                            continue;
                        gi = info_map.find(n3->idatm_type());
                        if (gi == info_map.end())
                            continue;
                        int n3geom = (*gi).second.geometry;
                        if (n3geom != 3)
                            continue;
                        ambiguous = true;
                        all_single = false;
                    }
                    if (all_single) {
                        outside_double = true;
                        break;
                    }
                }
                if (outside_double)
                    connected[nb] = SINGLE;
                else if (ambiguous)
                    connected[nb] = AMBIGUOUS;
                else
                    connected[nb] = DOUBLE;
            }
        }
        std::map<Bond*, int> cur_assign;
        std::vector<std::map<Bond*, int>> assignments;
        std::vector<std::vector<Atom*>> assigned_uncertains;
        std::map<Atom*, Bond*> uncertain2bond;
        make_assignments(bonds, connected, cur_assign, &assignments);
        if (assignments.size() == 0)
            // try a charged ring
            make_assignments(bonds, connected, cur_assign, &assignments, true);
        else {
            // if there are no aromatic assignments for a ring and the ring
            // has a nitrogen/oxygen, append charged assignments
            bool add_charged = false;
            for (auto ring: system_rings) {
                bool has_NO = false;
                for (auto a: ring->atoms()) {
                    if (a->element() == Element::N
                    || a->element() == Element::O) {
                        has_NO = true;
                        break;
                    }
                }
                if (!has_NO)
                    continue;
                bool any_aro = false;
                for (auto& assignment: assignments) {
                    int bondSum = 0;
                    for (auto b: ring->bonds()) {
                        bondSum += assignment[b];
                    }
                    int size = ring->bonds().size();
                    if (bondSum == size + size/2) {
                        any_aro = true;
                        break;
                    }
                }
                if (!any_aro) {
                    add_charged = true;
                    break;
                }
            }
            if (add_charged) {
                auto prev_assignments = assignments;
                assignments.clear();
                make_assignments(bonds, connected, cur_assign, &assignments,
                    true);
                assignments.insert(assignments.end(), prev_assignments.begin(),
                    prev_assignments.end());
            }
        }
        if (assignments.size() == 0) {
            // see if flipping a possibly-ambiguous bond
            // allows an assignment to be made
            std::vector<std::pair<Real, Bond*> > sortable;
            for (auto b_a: possibly_ambiguous) {
                Bond *b = b_a.first;
                Atom *a = b_a.second;
                const Element& e = a->element();
                Real certainty;
                if (e == Element::O) {
                    certainty = b->sqlength() - p3o2c2;
                } else if (e == Element::S) {
                    certainty = b->sqlength() - p3s2c2;
                } else {
                    certainty = 0.0;
                }
                if (certainty < 0.0)
                    certainty = 0.0 - certainty;
                sortable.push_back(std::pair<Real, Bond*>(certainty, b));
            }
            std::sort(sortable.begin(), sortable.end());
            std::vector<Bond *> flippable;
            for (auto s: sortable) {
                flippable.push_back(s.second);
            }
            if (flippable.size() > 0) {
                flip_assign(flippable, atoms, bonds, connected, &assignments);
                if (assignments.size() == 0)
                    flip_assign(flippable, atoms, bonds,
                        connected, &assignments, true);
            }
        }

        if (assignments.size() == 0) {
            // if adjacent carbons were uncertain (i.e. had
            // "redo" values) try changing their type
            std::vector<Atom *> uncertain;
            for (auto rn_a_b: ring_neighbors) {
                Atom *rna = rn_a_b.first;
                Bond *rnb = rn_a_b.second;
                if (redo.find(rna) != redo.end() && redo[rna] != 0) {
                    uncertain.push_back(rna);
                    uncertain2bond[rna] = rnb;
                }
            }
            if (uncertain.size() > 0) {
                uncertain_assign(uncertain, uncertain2bond, bonds, connected,
                            &assignments, &assigned_uncertains);
                if (assignments.size() == 0)
                    uncertain_assign(uncertain, uncertain2bond, bonds,
                        connected, &assignments, &assigned_uncertains, true);
            }
        }
        if (assignments.size() == 0) {
            auto a = *atoms.begin();
            logger::warning(_logger, "Cannot find consistent set of bond"
                " orders for ring system containing atom ", a->name(),
                " in residue ", a->residue()->str());
            continue;
        }

        auto best_assignment = find_best_assignment(assignments, system_rings);
        if (best_assignment != NULL && assigned_uncertains.size() > 0) {
            unsigned int ba_index = std::find(assignments.begin(),
                assignments.end(), *best_assignment) - assignments.begin();
            invert_uncertains(assigned_uncertains[ba_index],
                            uncertain2bond, &connected);
        }

        // see if individual rings are aromatic -- if not
        // then assign types as per best assignment
        for (auto ring: system_rings) {
            int min_free_electrons = 0, max_free_electrons = 0;
            const Ring::Bonds &bonds = ring->bonds();
            std::set<Bond *> ring_bonds;
            ring_bonds.insert(bonds.begin(), bonds.end());
            const Ring::Atoms &atoms = ring->atoms();
            bool aro = true;
            for (auto a: atoms) {
                const Atom::Bonds &a_bonds = a->bonds();
                min_free_electrons++; // a few exceptions below
                if (a_bonds.size() == 2) {
                    int element = a->element().number();
                    if (element > 20) {
                        max_free_electrons += 2;
                        continue;
                    }
                    int valence = (element - 2) % 8;
                    if (valence < 3 || valence == 7) {
                        aro = false;
                        break;
                    }
                    if (valence < 5)
                        max_free_electrons++;
                    else if (valence == 5) {
                        if (best_assignment == NULL)
                            max_free_electrons += 2;
                        else {
                            int sum = (*best_assignment)[a_bonds[0]]
                                + (*best_assignment)[a_bonds[1]];
                            if (sum == 2) {
                                min_free_electrons++;
                                max_free_electrons += 2;
                            } else
                                max_free_electrons++;
                        }
                    } else {
                        if (best_assignment != NULL
                        && (element == 8 && is_Oarplus(best_assignment, a_bonds))) {
                                max_free_electrons++;
                        } else {
                            min_free_electrons++;
                            max_free_electrons += 2;
                        }
                    }
                } else if (a_bonds.size() == 3) {
                    BondOrder bo = AMBIGUOUS;
                    Bond *out_bond;
                    for (auto b: a_bonds) {
                        if (ring_bonds.find(b) != ring_bonds.end())
                            continue;
                        out_bond = b;
                        if (connected.find(b) != connected.end()) {
                            bo = connected[b];
                            break;
                        }
                        for (auto assignment: assignments) {
                            int b_assignment = assignment[b];
                            if (bo == AMBIGUOUS) {
                                bo = (BondOrder) b_assignment;
                            } else if (bo != b_assignment) {
                                bo = AMBIGUOUS;
                                break;
                            }
                        }
                        break;
                    }
                    int element = a->element().number();
                    if (element > 20) {
                        max_free_electrons += 2;
                        continue;
                    }
                    int valence = (element - 2) % 8;
                    if (valence < 4 || valence > 5) {
                        aro = false;
                        break;
                    }
                    if (valence == 4) {
                        if (bo == DOUBLE) {
                            if (out_bond->other_atom(a)->neighbors().size()>1) {
                                bool is_fused = false;
                                for (auto s_ring: system_rings) {
                                    if (s_ring == ring) {
                                        continue;
                                    }
                                    const Ring::Atoms &sr_atoms=s_ring->atoms();
                                    if (sr_atoms.find(a) != sr_atoms.end()) {
                                        is_fused = true;
                                        break;
                                    }
                                }
                                if (!is_fused) {
                                    aro = false;
                                    break;
                                }
                            } else {
                                aro = false;
                                break;
                            }
                        }
                        max_free_electrons += 1;
                    } else if (bo == DOUBLE ||
                        // ... or N2+ or Oar+
                    (best_assignment != NULL
                    && (element == 7 && is_N2plus(best_assignment, a_bonds))))
                        max_free_electrons += 1;
                    else {
                        min_free_electrons++;
                        max_free_electrons += 2;
                    }
                } else {
                    aro = false;
                    break;
                }
            }
            int aro_eval;
            if (aro) {
                aro_eval = (min_free_electrons-2) % 4;
                if ((aro_eval != 0) && (aro_eval +
                (max_free_electrons - min_free_electrons) < 4))
                    aro = false;
            }
            // assign aromatic types if aro (N2/Npl depends
            // on number of needed electrons)
            std::set<Atom *> protonatable_Ns;
            for (auto a: atoms) {
                const Element& e = a->element();
                if (e == Element::N) {
                    if (a->bonds().size() == 2)
                        protonatable_Ns.insert(a);
                    else if (best_assignment != NULL) {
                        // distinguish N2+/Npl
                        if (aro && is_N2plus(best_assignment, a->bonds())) {
                            // non-ring bond == 0
                            a->set_computed_idatm_type("N2+");
                            // type bonded isolated O as O3-
                            if (a->bonds().size() == 3) {
                                for (auto nb: a->neighbors()) {
                                    if (nb->element() == Element::O
                                    && nb->bonds().size() == 1)
                                        nb->set_computed_idatm_type("O3-");
                                }
                            }
                        }
                    }
                    continue;
                }
                if (!aro)
                    continue;
                if (e == Element::C) {
                    a->set_computed_idatm_type("Car");
                } else if (e == Element::O) {
                    if (is_Oarplus(best_assignment, a->bonds()))
                        a->set_computed_idatm_type("Oar+");
                    else
                        a->set_computed_idatm_type("Oar");
                } else if (e == Element::S) {
                    a->set_computed_idatm_type("Sar");
                } else if (e == Element::P) {
                    if (a->bonds().size() == 2)
                        if (aro_eval % 4 != 0)
                            aro_eval++;
                } else if (e.number() > 20) {
                    if (aro_eval % 4 != 0)
                        aro_eval++;
                }
            }
            if (best_assignment == NULL && aro) {
                // N2/Npl depends on number of needed electrons
                while (aro_eval % 4 != 0 && protonatable_Ns.size() > 0) {
                    aro_eval++;
                    Atom *longest_N = NULL;
                    float best_val = 0.0;
                    for (auto a: protonatable_Ns) {
                        float val = 0.0;
                        for (auto b: a->bonds()) {
                            val += b->length();
                        }
                        if (longest_N == NULL || val > best_val) {
                            longest_N = a;
                            best_val = val;
                        }
                        // avoid retyping in pass 7
                        redo[a] = -7;
                    }
                    longest_N->set_computed_idatm_type("Npl");
                    protonatable_Ns.erase(longest_N);
                    ring_assigned_Ns.insert(longest_N);

                }
                for (auto a: protonatable_Ns) {
                    a->set_computed_idatm_type("N2");
                    ring_assigned_Ns.insert(a);
                }
            } else if (best_assignment != NULL) {
                
                // decide if two-bond nitrogens are N2 or Npl
                for (auto a: protonatable_Ns) {
                    int bond_sum = 0;
                    for (auto b: a->bonds()) {
                        bond_sum += (*best_assignment)[b];
                    }
                    if (bond_sum == 2)
                        a->set_computed_idatm_type("Npl");
                    else
                        a->set_computed_idatm_type("N2");
                    ring_assigned_Ns.insert(a);
                }
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 4.5 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 5": change isolated sp2 carbons to sp3 since it is 
    //   impossible for an atom to be sp2 hybrizided if all its 
    //   neighbors are sp3 hybridized.  In addition, a carbon atom cannot
    //   be double bonded to a carboxylate carbon, phosphate phosphorus,
    //   sulfate sulfur, sulfone sulfur, sulfoxide sulfur, or sp1 carbon.
    //   Addition not in original idatm: Npl/N2+ also
    //
    //   This has now been streamlined to:  must be bonded to an sp2
    //   atom other than carboxylate carbon. Also, if the sp2 carbon
    //   is valence 3 and a neighbor is valence 1, then "trust" the sp2
    //   assignment and instead change the neighbor to sp2.
    for (auto a: untyped_atoms) {

        if (a->idatm_type() != "C2")
            continue;

        bool c2_possible = false;
        std::vector<Atom *> nb_valence1;
        int num_bonded_Npls = 0;
        for (auto bondee: a->neighbors()) {
            auto bondee_type = bondee->idatm_type();

            Atom::IdatmInfoMap::const_iterator i = info_map.find(bondee_type);
            if (i == info_map.end())
                continue;
            if ((*i).second.geometry == 3 && bondee_type != "Cac"
                    && bondee_type != "N2+"
                    // Npl with two bonds or less may be N2
                    && !(bondee_type == "Npl"
                    && bondee->neighbors().size() > 2
                    // because Ng+ isn't assigned until next pass
                    && heavys[bondee] > 1)) {
                c2_possible = true;
                break;
            } else if (bondee->neighbors().size() == 1)
                nb_valence1.push_back(bondee);
            if (bondee_type == "Npl")
                ++num_bonded_Npls;
        }
        if (num_bonded_Npls == 3)
            // guanidium
            c2_possible = true;

        if (!c2_possible) {
            if (a->neighbors().size() == 3 && nb_valence1.size() > 0) {
                Atom *best = NULL;
                float best_ratio;
                const char *best_sp2_type;
                for (auto nb: nb_valence1) {
                    const char *sp2_type;
                    Real test;
                    if (nb->element() == Element::C) {
                        sp2_type = "C2";
                        test = p3c2c;
                    } else if (nb->element() == Element::O) {
                        sp2_type = "O2";
                        test = p3o2c2;
                    } else if (nb->element() == Element::N) {
                        sp2_type = "N2";
                        test = p4n2c;
                    } else if (nb->element() == Element::S) {
                        sp2_type = "S2";
                        test = p3s2c2;
                    } else
                        continue;
                    Real sqlen = nb->coord().sqdistance(a->coord());
                    Real ratio = sqlen / test;
                    if (best == NULL || ratio < best_ratio) {
                        best = nb;
                        best_ratio = ratio;
                        best_sp2_type = sp2_type;
                    }
                }
                if (best != NULL)
                    best->set_computed_idatm_type(best_sp2_type);
                else
                    a->set_computed_idatm_type("C3");
            } else
                a->set_computed_idatm_type("C3");
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 5 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 6": 
    //   1) make decisions about the charge states of nitrogens.  If a
    //      nitrogen is bonded to sp3 carbons and/or hydrogens and/or
    //      deuteriums only, assume that it is positively charged (the pKa
    //      of its conjugate acid is probably high enough that the
    //      protonated form predominates at physiological pH).  If an sp2
    //      carbon is bonded to three planar nitrogens, it may be part of
    //      a guanidinium group.  Make the nitrogens positively charged
    //      (Ng+) if guanidine or similar structures can be ruled out (if
    //      'noplus' is false).
    //   2) make carboxyl oxygens negatively charged even if the proton is
    //      present (the pKa of the carboxyl group is probably low enough
    //      that the unprotonated form predominates at physiological pH).
    for (auto a: untyped_atoms) {
        
        const Element &element = a->element();
        if (element == Element::N && a->idatm_type() != "N3+") {
            if (is_N3plus_okay(a->neighbors()))
                a->set_computed_idatm_type("N3+");
            
        } else if (a->idatm_type() == "C2") {
            int num_Npls = 0;
            for (auto bondee: a->neighbors()) {
                if ((bondee->idatm_type() == "Npl"
                && untyped_set.find(bondee) != untyped_set.end())
                || bondee->idatm_type() == "Ng+")
                    // Ng+ possible through template
                    // typing
                    num_Npls++;
            }

            bool noplus = false;
            if (num_Npls >= 2) {
                for (auto bondee: a->neighbors()) {
                    if (bondee->idatm_type() != "Npl")
                        continue;
                    if (untyped_set.find(bondee) == untyped_set.end())
                        continue;
                    
                    if (bondee->rings(false,
                            ring_limit, &mapped_residues).size() > 0) {
                        noplus = true;
                        break;
                    }
                    bondee->set_computed_idatm_type("Ng+");
                }
            }
            if (noplus) {
                for (auto bondee: a->neighbors()) {
                    if (untyped_set.find(bondee) == untyped_set.end())
                        continue;
                    if (bondee->idatm_type() == "Ng+")
                        bondee->set_computed_idatm_type("Npl");
                }
            }
        } else if (a->idatm_type() == "Cac") {
            for (auto bondee: a->neighbors()) {
                if (untyped_set.find(bondee) == untyped_set.end())
                    continue;
                if (bondee->element() == Element::O && heavys[bondee] == 1) {
                    bondee->set_computed_idatm_type("O2-");
                }
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 6 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 7":  a non-IDATM pass:  split off heavy-atom-valence-2
    //  Npls that have no hydrogens as type N2.
    //  Discrimination criteria is the average bond length of the two 
    //  heavy-atom bonds (shorter implies more double-bond character,
    //  thereby no hydrogen).
    for (auto a: untyped_atoms) {

        if (a->idatm_type() != "Npl")
            continue;
        
        if (heavys[a] != 2)
            continue;

        if (ring_assigned_Ns.find(a) != ring_assigned_Ns.end())
            continue;
        
        if (a->neighbors().size() > 2)
            continue;

        Real threshold = 1.0;
        Real harm_len = 1.0;
        Atom *recipient = NULL;
        Real bratio = 1.0;
        for (auto bondee: a->neighbors()) {
            Real criteria;
            if (bondee->element() == Element::C) {
                criteria = p7cn2nh;
            } else if (bondee->element() == Element::N) {
                criteria = p7nn2nh;
            } else if (bondee->element() == Element::O) {
                if (bondee->neighbors().size() > 1)
                    continue;
                criteria = p7on2nh;
            } else {
                continue;
            }
            threshold *= criteria;
            Real len = bondee->coord().distance(a->coord());
            harm_len *= len;
            if (len > criteria)
                continue;
            Real ratio = len / criteria;
            if (ratio > bratio)
                continue;
            if (bondee->element() == Element::N && bondee->bonds().size() > 2)
                continue;
            if (bondee->idatm_type() == "Car")
                continue;

            bool double_okay = true;
            for (auto grand_bondee: bondee->neighbors()) {
                if (grand_bondee == a)
                    continue;
                auto gb_type = grand_bondee->idatm_type();

                Atom::IdatmInfoMap::const_iterator i = info_map.find(gb_type);
                if (i == info_map.end())
                    continue;
                int geom = (*i).second.geometry;
                if (geom > 1 && geom < 4 && heavys[grand_bondee] == 1) {
                        double_okay = false;
                        break;
                }
            }
            if (!double_okay)
                continue;
            recipient = bondee;
            bratio = ratio;
        }

        if (harm_len < threshold && recipient != NULL) {
            a->set_computed_idatm_type("N2");
            if (recipient->element() == Element::C) {
                recipient->set_computed_idatm_type("C2");
            } else if (recipient->element() == Element::N) {
                recipient->set_computed_idatm_type("N2");
            } else if (recipient->element() == Element::O) {
                recipient->set_computed_idatm_type("O2");
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 7 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 8":  another non-IDATM: change planar nitrogens bonded only
    //  SP3 atoms to N3 or N3+.  Change Npls/N3s bonded to sp2 atoms that
    //  are not in turn bonded to sp2 atoms (implying Npl doubled bonded)
    //  to N2, otherwise to Npl.  
    std::map<Atom *, std::vector<Atom *> > bonded_sp2s;
    for (auto a: untyped_atoms) {

        if (ring_assigned_Ns.find(a) != ring_assigned_Ns.end())
            continue;
        
        auto idatm_type = a->idatm_type();
        if (idatm_type != "Npl" && idatm_type != "N2" && idatm_type!="N3")
            continue;
        
        bool aro_ring = false;
        for (auto ring: a->rings(false, ring_limit, &mapped_residues)) {
            if (ring->aromatic()) {
                aro_ring = true;
                break;
            }
        }
        if (aro_ring)
            continue;

        // any sp2?
        if (idatm_type == "Npl" && a->neighbors().size() != 2)
            continue;

        std::vector<Atom *> bsp2list;
        for (auto bondee: a->neighbors()) {
            auto idatm_type = bondee->idatm_type();

            aro_ring = false;
            for (auto ring: bondee->rings(false, ring_limit, &mapped_residues)){
                if (ring->aromatic()) {
                    aro_ring = true;
                    break;
                }
            }
            if (aro_ring) {
                if (heavys[a] == 1) { // aniline
                    a->set_computed_idatm_type("Npl");
                    break;
                }
                continue;
            }

            Atom::IdatmInfoMap::const_iterator i = info_map.find(idatm_type);
            if (i == info_map.end() || (*i).second.geometry != 3
                    || bondee->idatm_type() == "Npl")
                continue;
            bsp2list.push_back(bondee);
        }
        bonded_sp2s[a] = bsp2list;
    }

    // order typing by easiest-figure-out (1 sp2 bonded) to hardest (3)
    // good test cases:  1CY in 3UM8; WRA in 1J3I
    for (unsigned int i = 1; i < 4; ++i) {
        for(auto a_sp2s: bonded_sp2s) {
            const std::vector<Atom *> &sp2s = a_sp2s.second;
            if (sp2s.size() != i)
                continue;
            Atom *a = a_sp2s.first;
            bool any_sp2 = false;

            for (auto bondee: sp2s) {
                any_sp2 = true;
                bool remote_sp2 = false;
                for (auto grand_bondee: bondee->neighbors()) {
                    if (grand_bondee == a)
                        continue;
                    Atom::IdatmInfoMap::const_iterator gi =
                    info_map.find(grand_bondee->idatm_type());
                    if (gi == info_map.end()
                    || (*gi).second.geometry == 3) {
                        if (grand_bondee->idatm_type() != "Car"
                        && grand_bondee->idatm_type() != "Npl"
                        ) {
                        //&& !(grand_bondee->idatm_type() == "Npl"
                        //& grand_bondee->bonds().size()
                        //= 2)) {
                            remote_sp2 = true;
                            break;
                        }
                    }
                }
                if (!remote_sp2) {
                    int hvys = heavys[a];
                    if (hvys > 1)
                        a->set_computed_idatm_type("N2");
                    else if (hvys == 1)
                        a->set_computed_idatm_type(is_N3plus_okay(
                            a->neighbors()) ? "N3+" : "N3");
                    else
                        a->set_computed_idatm_type("N3+");
                    break;
                }
                // a remote sp2 atom doesn't necessarily mean Npl
                // (see N1 in MX1 of 2aio), so no else clause
            }
            if (!any_sp2) {
                int hvys = heavys[a];
                if (hvys > 1)
                    a->set_computed_idatm_type("N3");
                else if (hvys == 1)
                    a->set_computed_idatm_type(is_N3plus_okay(
                        a->neighbors()) ? "N3+" : "N3");
                else
                    a->set_computed_idatm_type("N3+");
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 8 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 9":  another non-IDATM pass and analogous to pass 8:
    //  change O3 bonded only to non-Npl sp2 atom not in turn bonded
    //  to non-Npl sp2 to O2.
    for (auto a: untyped_atoms) {

        auto idatm_type = a->idatm_type();
        if (idatm_type != "O3")
            continue;
        
        if (a->neighbors().size() != 1)
            continue;

        // any sp2?
        Atom *bondee = a->neighbors()[0];
        auto bondee_type = bondee->idatm_type();

        bool aro_ring = false;
        for (auto b_ring: bondee->rings(false, ring_limit, &mapped_residues)) {
            if (b_ring->aromatic()) {
                aro_ring = true;
                break;
            }
        }
        if (aro_ring) {
            // can't be O2
            continue;
        }

        Atom::IdatmInfoMap::const_iterator i = info_map.find(bondee_type);
        if (i == info_map.end() || (*i).second.geometry != 3)
            continue;
        bool remote_sp2 = false;
        for (auto grand_bondee: bondee->neighbors()) {
            if (grand_bondee == a)
                continue;
            Atom::IdatmInfoMap::const_iterator gi =
                info_map.find(grand_bondee->idatm_type());
            if (gi == info_map.end() || (*gi).second.geometry == 3) {
                if (grand_bondee->idatm_type() != "Car"
                && grand_bondee->idatm_type() != "Npl") {
                    remote_sp2 = true;
                    break;
                }
            }
        }
        if (!remote_sp2)
            a->set_computed_idatm_type("O2");
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 9 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 10":  another non-IDATM pass. Ensure nitrate ions are N2/O2-
    for (auto a: untyped_atoms) {

        if (a->element() != Element::N)
            continue;
        
        if (a->neighbors().size() != 2)
            continue;

        bool bonders_okay = true;
        for (auto bondee: a->neighbors()) {
            if (bondee->element() != Element::O
            || bondee->neighbors().size() != 1) {
                bonders_okay = false;
                break;
            }
        }

        if (bonders_okay) {
            a->set_computed_idatm_type("N2");
            for (auto bondee: a->neighbors()) {
                bondee->set_computed_idatm_type("O2-");
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 10 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // "pass 11":  another non-IDATM pass. Change S3 sulfurs with < 2 bonds and
    // coordinating 3+ metal ions to S3-, regardless of templating
    //
    // First count coordinations
    auto pbg = pb_mgr().get_group(Structure::PBG_METAL_COORDINATION);
    if (pbg != nullptr) {
        std::map<Atom*,int> coord_count;
        for (auto pb: pbg->pseudobonds()) {
            for (auto a: pb->atoms()) {
                if (a->idatm_type() == "S3" && a->bonds().size() < 2) {
                    if (++coord_count[a] > 2)
                        a->set_computed_idatm_type("S3-");
                }
            }
        }
    }
#ifdef TIME_PASSES
t1 = clock();
std::cerr << "pass 11 took " << (t1 - t0) / (float)CLOCKS_PER_SEC << " seconds\n";
t0 = t1;
#endif

    // since the rings() "ignore" arg pointed to a local variable, the
    // rings cannot be reused...
    _recompute_rings = true;
#ifdef REPORT_TIME
clock_t end_t = clock();
std::cerr << "Tracking non-templated took " << (end_t - start_t) / (float)CLOCKS_PER_SEC << " seconds\n";
#endif
}

static void
invert_uncertains(std::vector<Atom*> &uncertain,
    std::map<Atom*, Bond*> &uncertain2bond,
    std::map<Bond*, BondOrder> *connected)
{
    for (auto a: uncertain) {
        Bond *b = uncertain2bond[a];
        (*connected)[b] = (BondOrder) (3 - (*connected)[b]);
        if (a->idatm_type() == "C3")
            a->set_computed_idatm_type("C2");
        else if (a->idatm_type() == "C2")
            a->set_computed_idatm_type("C3");
        else if (a->idatm_type() == "Npl")
            a->set_computed_idatm_type("N2");
        else if (a->idatm_type() == "N2")
            a->set_computed_idatm_type("Npl");
        else
            logger::error(a->structure()->logger(),
                "Unknown invert atom type: ", a->idatm_type(),
                " for atom ", a->str());
    }
}

static void
uncertain_assign(std::vector<Atom*> &uncertain,
    std::map<Atom*, Bond*> &uncertain2bond,
    std::set<Bond*> &bonds, std::map<Bond *, BondOrder> &connected,
    std::vector<std::map<Bond*, int> > *assignments,
    std::vector<std::vector<Atom*>> *assigned_uncertains,
    bool allow_charged)
{
    std::map<Bond*, int> cur_assign;
    std::vector<std::vector<Atom*>> permutations;
    generate_permutations<Atom>(uncertain, &permutations);
    // if we find an assignment involving changing N atoms, have to
    // try all permutations that involve changing no more than N atoms
    unsigned int limit = uncertain.size();
    for (auto& uncertain: permutations) {
        if (uncertain.size() > limit)
            break;
        invert_uncertains(uncertain, uncertain2bond, &connected);
        unsigned int num_prev_assigned = assignments->size();
        make_assignments(bonds, connected, cur_assign, assignments,
            allow_charged);
        int increase = assignments->size() - num_prev_assigned;
        if (increase) {
            limit = uncertain.size();
            for (int i = 0; i < increase; ++i) {
                assigned_uncertains->push_back(uncertain);
            }
        }
        invert_uncertains(uncertain, uncertain2bond, &connected);
    }
}

static void
flip_assign(std::vector<Bond*>& flippable, std::set<Atom*>& atoms,
    std::set<Bond*>& bonds, std::map<Bond*, BondOrder>& connected,
    std::vector<std::map<Bond*, int>>* assignments, bool allow_charged)
{
    std::map<Bond*, int> cur_assign;
    std::vector<std::vector<Bond*>> permutations;
    generate_permutations<Bond>(flippable, &permutations);
    for (auto& flip: permutations) {
        for (auto b: flip) {
            connected[b] = (BondOrder) (3 - connected[b]);
        }
        make_assignments(bonds, connected, cur_assign, assignments,
            allow_charged);
        if (assignments->size() > 0) {
            for (auto b: flip) {
                for (auto a: b->atoms()) {
                    if (atoms.find(a) != atoms.end())
                        continue;
                    const Element& e = a->element();
                    if (e == Element::O) {
                        if (connected[b] == 1)
                            a->set_computed_idatm_type("O3");
                        else
                            a->set_computed_idatm_type("O2");
                    } else if (e == Element::S) {
                        if (connected[b] == 1)
                            a->set_computed_idatm_type("S3");
                        else
                            a->set_computed_idatm_type("S2");
                    }
                }
            }
            break;
        }
        for (auto b: flip) {
            connected[b] = (BondOrder) (3 - connected[b]);
        }
    }
}

} // namespace atomstruct
