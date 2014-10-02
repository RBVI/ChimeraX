// vim: set expandtab ts=4 sw=4:

#include <algorithm>
#include <map>
#include <stdlib.h>

#include "Atom.h"
#include "Residue.h"
#include "AtomicStructure.h"
#include "tmpl/Residue.h"
#include "tmpl/Atom.h"
#include "tmpl/residues.h"
#include "MolResId.h"
#include "connect.h"

namespace atomstruct {

using basegeom::Coord;

// standard_residues contains the names of residues that should have PDB ATOM records.
static std::set<std::string, std::less<std::string> >    standard_residues;

static void
init_standard_residues()
{
    standard_residues.insert("A");
    standard_residues.insert("ALA");
    standard_residues.insert("ARG");
    standard_residues.insert("ASN");
    standard_residues.insert("ASP");
    standard_residues.insert("ASX");
    standard_residues.insert("C");
    standard_residues.insert("CYS");
    standard_residues.insert("DA");
    standard_residues.insert("DC");
    standard_residues.insert("DG");
    standard_residues.insert("DT");
    standard_residues.insert("G");
    standard_residues.insert("GLN");
    standard_residues.insert("GLU");
    standard_residues.insert("GLX");
    standard_residues.insert("GLY");
    standard_residues.insert("HIS");
    standard_residues.insert("I");
    standard_residues.insert("ILE");
    standard_residues.insert("LEU");
    standard_residues.insert("LYS");
    standard_residues.insert("MET");
    standard_residues.insert("PHE");
    standard_residues.insert("PRO");
    standard_residues.insert("SER");
    standard_residues.insert("T");
    standard_residues.insert("THR");
    standard_residues.insert("TRP");
    standard_residues.insert("TYR");
    standard_residues.insert("U");
    standard_residues.insert("VAL");
}

//TODO: these 3 funcs need to be wrapped also
bool
standard_residue(const std::string& name)
{
    if (standard_residues.empty())
        init_standard_residues();
    return standard_residues.find(name) != standard_residues.end();
}

void
add_standard_residue(const std::string& name)
{
    if (standard_residues.empty())
        init_standard_residues();
    standard_residues.insert(name);
}

void
remove_standard_residue(const std::string& name)
{
    if (standard_residues.empty())
        init_standard_residues();
    standard_residues.erase(name);
}

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
        } else
            (void) a->structure()->new_bond(a, oa);
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
            if (!a->connects_to(b))
                (void) a->structure()->new_bond(a, b);
        }
    }
    // For each atom that wasn't connected (i.e. not in template),
    // connect it by distance
    if (!some_connectivity_unknown)
        return;
    connect_residue_by_distance(r, &known_connectivity);
}

static std::map<Element, unsigned long>  _saturationMap = {
    {Element::H, 1},
    {Element::O, 2}
};
static bool
saturated(Atom* a)
{
    auto info = _saturationMap.find(a->element());
    if (info == _saturationMap.end())
        return a->bonds().size() >= 4;
    return a->bonds().size() >= (*info).second;

}

// find_closest:
//    Find closest heavy atom to given heavy atom with residue that has
//    the same alternate location identifier (or none) and optionally return
static Atom *
find_closest(Atom* a, Residue* r, float* ret_dist_sq, bool nonSaturated=false)
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
    if (fsave != NULL) {
        if (!any_length && bonded_dist(fsave, tsave) == 0.0)
            return;
        add_bond(fsave, tsave);
    }
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

// connect_structure:
//    Connect atoms in structure by template if one is found, or by distance.
//    Adjacent residues are connected if appropriate.
void
connect_structure(AtomicStructure* as, std::vector<Residue *>* start_residues,
    std::vector<Residue *>* end_residues, std::set<Atom *>* conect_atoms,
    std::set<MolResId>* mod_res)
{
    // walk the residues, connecting residues as appropriate and
    // connect the atoms within the residue
    Residue *link_res = NULL, *prev_res = NULL, *first_res = NULL;
    Atom *link_atom;
    std::string link_atom_name("");
    for (AtomicStructure::Residues::const_iterator ri = as->residues().begin();
    ri != as->residues().end(); ++ri) {
        Residue *r = (*ri).get();

        if (!first_res)
            first_res = r;
        const tmpl::Residue *tr;
        if (mod_res->find(MolResId(r)) != mod_res->end())
            // residue in MODRES record;
            // don't try to use template connectivity
            tr = NULL;
        else
            tr = tmpl::find_template_residue(r->name(),
                std::find(start_residues->begin(),
                start_residues->end(), r) != start_residues->end(),
                std::find(end_residues->begin(),
                end_residues->end(), r) != end_residues->end());
        if (tr != NULL)
            connect_residue_by_template(r, tr, conect_atoms);
        else
            connect_residue_by_distance(r, conect_atoms);

        // connect up previous residue
        if (link_res != NULL) {
            if (tr == NULL || tr->chief() == NULL) {
                add_bond_nearest_pair(link_res, r);
            } else {
                bool made_connection = false;
                // don't definitely connect a leading HET residue
                bool definitely_connect = (link_res != first_res
                    || link_atom_name != "");
                Atom *chief = r->find_atom(tr->chief()->name());
                if (chief != NULL) {
                    // 1vqn, chain 5, is a nucleic/amino acid
                    // hybrid with the na/aa connectivity in
                    // CONECT records; prevent also adding a
                    // chief-link bond
                    if (saturated(chief)) {
                        made_connection = true;
                    } if (link_atom != NULL) {
                        if (!saturated(link_atom))
                            add_bond(link_atom, chief);
                        made_connection = true;
                    } else {
                        made_connection = hookup(chief, link_res, definitely_connect);
                    }
                }
                if (!made_connection && definitely_connect)
                    add_bond_nearest_pair(link_res, r);
            }
        } else if (r->atoms().size() > 1 && prev_res != NULL
                && prev_res->chain_id() == r->chain_id()
                && r->is_het() && conect_atoms->find(
                (*r->atoms().begin())) == conect_atoms->end()) {
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
        for (AtomicStructure::Residues::const_iterator ri=as->residues().begin()
        ; ri != as->residues().end(); ++ri) {
            Residue *r = (*ri).get();
            if (standard_residue(r->name()) || r->name() == "UNK")
                continue;
            if (!r->is_het()) {
                break_long = true;
                break;
            }
        }
    }
    if (break_long) {
        std::vector<Bond *> break_these;
        for (AtomicStructure::Bonds::const_iterator bi = as->bonds().begin();
        bi != as->bonds().end(); ++bi) {
            Bond *b = (*bi).get();
            const Bond::Atoms & atoms = b->atoms();
            Residue *r1 = atoms[0]->residue();
            Residue *r2 = atoms[1]->residue();
            if (r1 == r2)
                continue;
            if (standard_residue(r1->name()) && standard_residue(r2->name()))
                continue;
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
    } else {
        // turn long inter-residue bonds into "missing structure" pseudobonds
        std::vector<Bond*> long_bonds;
        for (auto& b: as->bonds()) {
            Atom* a1 = b->atoms()[0];
            Atom* a2 = b->atoms()[1];
            Residue* r1 = a1->residue();
            Residue* r2 = a2->residue();
            if (r1 == r2)
                continue;
            if (r1->chain_id() == r2->chain_id()
            && abs(r1->position() - r2->position()) < 2)
                continue;
            auto idealBL = Element::bond_length(a1->element(), a2->element());
            if (b->sqlength() >= 3.0625 * idealBL * idealBL)
                // 3.0625 == 1.75 squared
                // (allows ASP 223.A OD2 <-> PLP 409.A N1 bond in 1aam
                // and SER 233.A OG <-> NDP 300.A O1X bond in 1a80
                // to not be classified as missing seqments)
                long_bonds.push_back(b.get());
        }
        if (long_bonds.size() > 0) {
            auto pbg = as->pb_mgr().get_group(as->PBG_MISSING_STRUCTURE,
                AS_PBManager::GRP_NORMAL);
            for (auto lb: long_bonds) {
                pbg->newPseudoBond(lb->atoms());
                as->delete_bond(lb);
            }
        }
    }
}

}  // namespace atomstruct
