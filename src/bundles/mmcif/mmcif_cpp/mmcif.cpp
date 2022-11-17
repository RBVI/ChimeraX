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

#include "_mmcif.h"
#include "mmcif.h"
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/Pseudobond.h>
#include <atomstruct/PBGroup.h>
#include <pdb/connect.h>
#include <atomstruct/tmpl/restmpl.h>
#include <logger/logger.h>
#include <arrays/pythonarray.h>	// Use python_voidp_array()
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>
#include <set>

#undef CLOCK_PROFILING
#ifdef CLOCK_PROFILING
#include <ctime>
#endif

// The PDB has a limited form of struct_sheet_hbond called
// pdbx_struct_sheet_hbond that does the simple case "where only a single
// hydrogen bond is used to register the two residue ranges".
// This is insufficient to actually be useful.  So the working code is
// ifdef'd out to be used if struct_sheet_hbond support is implemented.
#undef SHEET_HBONDS

using std::hash;
using std::map;
using std::multiset;
using std::set;
using std::string;
using std::unordered_map;
using std::vector;

using atomstruct::AtomicStructure;
using atomstruct::Structure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using element::Element;
using atomstruct::MolResId;
using atomstruct::Coord;
using atomstruct::Real;
using atomstruct::StructureSeq;

using atomstruct::AtomName;
using atomstruct::ChainID;
using atomstruct::ResName;
using atomstruct::PolymerType;

namespace {

set<ResName> standard_residues {
    "ALA", "ASX", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
    "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP",
    "UNK", "TYR", "GLX",
    "A", "G", "C", "U", "I", "N",  // RNA
    "DA", "DG", "DC", "DT", "DI", "DN", // DNA
};

bool is_standard_residue(const ResName& name)
{
    return standard_residues.find(name) != standard_residues.end();
}

// Symbolic names for readcif arguments
static const bool Required = true;  // column is required

inline void
canonicalize_atom_name(AtomName* aname, bool* asterisks_translated)
{
    for (size_t i = aname->length(); i > 0; ) {
        --i;
        // use prime instead of asterisk
        if ((*aname)[i] == '*') {
            (*aname)[i] = '\'';
            *asterisks_translated = true;
        }
    }
}

Atom* closest_atom_by_template(tmpl::Atom* ta, const tmpl::Residue* tr, const Residue* r)
{
    // Find closest (in hops) existing (heavy) atom to given atom in template
    // Assumes atom cooresponding to ta is not in residue
    const vector<Atom*>& atoms = r->atoms();
    if (atoms.size() == 1) {
        // optimization for C-alpha traces
        return atoms[0];
    }
    set<tmpl::Atom*> visited;
    vector<tmpl::Atom*> to_visit;
    to_visit.reserve(tr->atoms_map().size());  // upper bound
    visited.insert(ta);
    for (auto n: ta->neighbors()) {
        if (n->element().number() != 1)
            to_visit.push_back(n);
    }
    for (auto i = to_visit.begin(); i != to_visit.end(); ++i) {
        ta = *i;
        Atom* a = r->find_atom(ta->name());
        if (a)
            return a;
        visited.insert(ta);
        for (auto n: ta->neighbors()) {
            if (n->element().number() != 1 && visited.find(n) == visited.end())
                to_visit.push_back(n);
        }
    }
    return nullptr;
}

} // namespace

namespace mmcif {

typedef vector<string> StringVector;
typedef vector<unsigned> UIntVector;

// DEBUG code
template <typename t> std::ostream&
operator<<(std::ostream& os, const vector<t>& v)
{
    for (auto&& i = v.begin(), e = v.end(); i != e; ++i) {
        os << *i;
        if (i != e - 1)
            os << ", ";
    }
    return os;
}

#ifdef CLOCK_PROFILING
class ClockProfile {
    const char *tag;
    clock_t start_t;
public:
    ClockProfile(const char *t): tag(t), start_t(clock()) {
    }
    ~ClockProfile() {
        clock_t end_t = clock();
        std::cout << tag << "; " << (end_t - start_t) / (float) CLOCKS_PER_SEC
            << "\n";
    }
};
#endif

bool reasonable_bond_length(Atom* a1, Atom* a2, float distance = std::numeric_limits<float>::quiet_NaN())
{
    Real idealBL = Element::bond_length(a1->element(), a2->element());
    Real sqlength;
    if (!std::isnan(distance))
        sqlength = distance * distance;
    else
        sqlength = a1->coord().sqdistance(a2->coord());
    // 3.0625 == 1.75 squared
    // (allows ASP 223.A OD2 <-> PLP 409.A N1 bond in 1aam
    // and SER 233.A OG <-> NDP 300.A O1X bond in 1a80
    // to not be classified as missing seqments)
    return (sqlength < 3.0625f * idealBL * idealBL);
}


struct ExtractMolecule: public readcif::CIFFile
{
    static const char* builtin_categories[];
    PyObject* _logger;
    ExtractMolecule(PyObject* logger, const StringVector& generic_categories, bool coordsets,
        bool atomic);
    ~ExtractMolecule();
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();
    void connect_polymer_pair(Residue* r0, Residue* r1, bool gap, bool nstd_okay, int model_num);
    void connect_residue_by_template(Residue* r, const tmpl::Residue* tr, int model_num);
    const tmpl::Residue* find_template_residue(const ResName& name, bool start = false, bool stop = false);
    void parse_audit_conform();
    void parse_audit_syntax();
    void parse_atom_site();
    void parse_atom_site_anisotrop();
    void parse_struct_conn();
    void parse_struct_conf();
    void parse_struct_sheet_range();
#ifdef SHEET_HBONDS
    void parse_struct_sheet_order();
    void parse_pdbx_struct_sheet_hbond();
#endif
    void parse_entity();
    void parse_entity_poly();
    void parse_entity_poly_seq();
    void parse_entry();
    void parse_pdbx_database_PDB_obs_spr();
    void parse_generic_category();
    // for inline resiude templates
    void parse_chem_comp();
    void parse_chem_comp_bond();

    std::map<string, StringVector> generic_tables;
    vector<Structure*> all_molecules;
    map<int, Structure*> molecules;
    // serial_num -> atom, alt_id for atom_site_anisotrop
    std::map <long, std::pair<Atom*, char>> atom_lookup;
    map<ChainID, string> chain_entity_map;  // [label_asym_id] -> entity_id
    map<string, string> entity_description;  // entity_id: description
    struct ResidueKey {
        string entity_id;
        long seq_id;
        ResName mon_id;
        ResidueKey(const string& e, long n, const ResName& m):
            entity_id(e), seq_id(n), mon_id(m) {}
        bool operator==(const ResidueKey& k) const {
            return seq_id == k.seq_id && entity_id == k.entity_id
                && mon_id == k.mon_id;
        }
        bool operator<(const ResidueKey& k) const {
            if (seq_id < k.seq_id)
                return true;
            if (seq_id != k.seq_id)
                return false;
            if (entity_id < k.entity_id)
                return true;
            if (entity_id != k.entity_id)
                return false;
            return mon_id < k.mon_id;
        }
    };
    struct hash_ResidueKey {
        size_t operator()(const ResidueKey& k) const {
            return hash<string>()(k.entity_id)
                ^ hash<long>()(k.seq_id)
                ^ hash<ResName>()(k.mon_id);
        }
    };
    typedef unordered_map<ResidueKey, Residue*, hash_ResidueKey> ResidueMap;
    typedef unordered_map<ChainID, ResidueMap> ChainResidueMap;
    unordered_map<int, ChainResidueMap> all_residues;
    struct PolySeq {
        long seq_id;
        ResName mon_id;
        bool hetero;
        PolySeq(long s, const ResName& m, bool h):
            seq_id(s), mon_id(m), hetero(h) {}
        bool operator<(const PolySeq& p) const {
            return this->seq_id < p.seq_id;
        }
    };
    typedef multiset<PolySeq> EntityPolySeq;
    struct EntityPoly {
        multiset<PolySeq> seq;
        bool nstd;
        PolymerType ptype;
        EntityPoly(bool nstd, PolymerType pt = PolymerType::PT_NONE): nstd(nstd), ptype(pt) {}
    };
    map<string /* entity_id */, EntityPoly> poly;
    set<string /* entity_id */> non_poly;
    int first_model_num;
    string entry_id;
    tmpl::Molecule* my_templates;
    bool found_missing_poly_seq;
    map<string, bool> has_poly_seq;   // entity_id: bool
    set<ResName> empty_residue_templates;
    set<ResName> missing_residue_templates;
    bool coordsets;  // use coordsets (trajectory) instead of separate models (NMR)
    bool atomic;  // use AtomicStructure if true, else Structure
    bool guess_fixed_width_categories;
    bool verbose;  // whether to give extra warning messages
    int hydrogens_missing_in_template;
#ifdef SHEET_HBONDS
    typedef map<string, map<std::pair<string, string>, string>> SheetOrder;
    SheetOrder sheet_order;
    typedef map<std::pair<string, string>, vector<Residue*>> StrandInfo;
    StrandInfo strand_info;
#endif
    Residue* find_residue(int model_num, const ChainID& chain_id, long position, const ResName& name);
    Residue* find_residue(const ChainResidueMap& crm, const ChainID& chain_id, ResidueKey& rk);
};

const char* ExtractMolecule::builtin_categories[] = {
    "chimerax_audit_syntax",
    "audit_conform", "audit_syntax", "entity", "entity_poly", "entity_poly_seq",
    "atom_site", "atom_site_anisotrop",
    "struct_conn", "struct_conf", "struct_sheet_range",
#ifdef SHEET_HBONDS
    "struct_sheet_order", "pdbx_struct_sheet_hbond",
#endif
    "chem_comp", "chem_comp_bond",
};
#define MIXED_CASE_BUILTIN_CATEGORIES 0

ExtractMolecule::ExtractMolecule(PyObject* logger, const StringVector& generic_categories, bool coordsets, bool atomic):
    _logger(logger), first_model_num(INT_MAX), my_templates(nullptr),
    found_missing_poly_seq(false), coordsets(coordsets), atomic(atomic),
    guess_fixed_width_categories(false), verbose(false), hydrogens_missing_in_template(0)
{
    empty_residue_templates.insert("UNL");  // Unknown ligand
    empty_residue_templates.insert("UNX");  // Unknown atom or ion
    register_category("audit_conform",
        [this] () {
            parse_audit_conform();
        });
    register_category("chimerax_audit_syntax",
        [this] () {
            parse_audit_syntax();
        });
    register_category("audit_syntax",
        [this] () {
            parse_audit_syntax();
        });
    register_category("entry",
        [this] () {
            parse_entry();
        });
    register_category("pdbx_database_PDB_obs_spr",
        [this] () {
            parse_pdbx_database_PDB_obs_spr();
        }, { "entry" });
    register_category("entity",
        [this] () {
            parse_entity();
        });
    register_category("entity_poly",
        [this] () {
            parse_entity_poly();
        });
    register_category("entity_poly_seq",
        [this] () {
            parse_entity_poly_seq();
        }, { "entity_poly" });
    register_category("atom_site",
        [this] () {
            parse_atom_site();
        }, { "entity", "entity_poly_seq" });
    register_category("atom_site_anisotrop",
        [this] () {
            parse_atom_site_anisotrop();
        }, { "atom_site" });
    register_category("struct_conn",
        [this] () {
            parse_struct_conn();
        }, { "atom_site" });
    register_category("struct_conf",
        [this] () {
            parse_struct_conf();
        }, { "struct_conn",  "entity_poly_seq" });
    register_category("struct_sheet_range",
        [this] () {
            parse_struct_sheet_range();
        }, { "struct_conn" });
#ifdef SHEET_HBONDS
    register_category("struct_sheet_order",
        [this] () {
            parse_struct_sheet_order();
        });
    register_category("pdbx_struct_sheet_hbond",
        [this] () {
            parse_pdbx_struct_sheet_hbond();
        }, { "atom_site", "struct_sheet_order", "struct_sheet_range" });
#endif
    register_category("chem_comp",
        [this] () {
            parse_chem_comp();
        });
    register_category("chem_comp_bond",
        [this] () {
            parse_chem_comp_bond();
        }, { "chem_comp" });
    // must be last:
    for (auto& c: generic_categories) {
#if MIXED_CASE_BUILTIN_CATEGORIES==0
        const string& category_ci = c;
#else
        string category_ci(c);
        for (auto& c: category_ci)
            c = tolower(c);
#endif
        if (std::find(std::begin(builtin_categories), std::end(builtin_categories), category_ci) != std::end(builtin_categories)) {
            logger::warning(_logger, "Can not override builtin parsing for "
                            "category: ", c);
            continue;
        }
        register_category(c,
            [this] () {
                parse_generic_category();
            });
    }
}

ExtractMolecule::~ExtractMolecule()
{
    if (verbose) {
        if (has_PDBx_fixed_width_columns())
            logger::info(_logger, "Used PDBx fixed column width tables to speed up reading mmCIF file");
        else
            logger::info(_logger, "No PDBx fixed column width tables");
        if (PDBx_keywords())
            logger::info(_logger, "Used PDBx keywords to speed up reading mmCIF file");
        else
            logger::info(_logger, "No PDBx keywords");
    }
    if (my_templates)
        delete my_templates;
}

void
ExtractMolecule::reset_parse()
{
    molecules.clear();
    atom_lookup.clear();
    chain_entity_map.clear();
    entity_description.clear();
    all_residues.clear();
#ifdef SHEET_HBONDS
    sheet_order.clear();
    strand_info.clear();
#endif
    entry_id.clear();
    generic_tables.clear();
    poly.clear();
    non_poly.clear();
    if (my_templates) {
        delete my_templates;
        my_templates = nullptr;
    }
    found_missing_poly_seq = false;
    has_poly_seq.clear();
    guess_fixed_width_categories = false;
    hydrogens_missing_in_template = 0;
}

inline Residue*
ExtractMolecule::find_residue(int model_num, const ChainID& chain_id, long position, const ResName& name) {
    // Document all of the steps to find a Residue given its
    // pdbx_PDB_model_num, label_asym_id, label_seq_id, label_comp_id values.
    // Typically, all_residues and the entity_id are predetermined
    // and don't need to be recomputed each time.
    const auto cemi = chain_entity_map.find(chain_id);
    if (cemi == chain_entity_map.end())
        return nullptr;
    const auto ari = all_residues.find(model_num);
    if (ari == all_residues.end())
        return nullptr;
    const auto ci = ari->second.find(chain_id);
    if (ci == ari->second.end())
        return nullptr;
    const ResidueMap& residue_map = ci->second;
    const string& entity_id = cemi->second;
    auto rmi = residue_map.find(ResidueKey(entity_id, position, name));
    if (rmi == residue_map.end())
        return nullptr;
    return rmi->second;
}

inline Residue*
ExtractMolecule::find_residue(const ChainResidueMap& crm, const ChainID& chain_id, ResidueKey& rk)
{
    const auto ci = crm.find(chain_id);
    if (ci == crm.end())
        return nullptr;
    const ResidueMap& residue_map = ci->second;
    const auto rmi = residue_map.find(rk);
    if (rmi == residue_map.end())
        return nullptr;
    return rmi->second;
}

const tmpl::Residue*
ExtractMolecule::find_template_residue(const ResName& name, bool start, bool stop)
{
    if (my_templates) {
        auto tr = my_templates->find_residue(name);
        if (tr && tr->atoms_map().size() > 0)
            return tr;
    }
    if (missing_residue_templates.find(name) != missing_residue_templates.end())
        return nullptr;
    auto tr =  mmcif::find_template_residue(name, start, stop);
    if (tr == nullptr) {
        // skipped warning if already given for this molecule
        logger::warning(_logger,
            "Unable to fetch template for '", name, "': might have incorrect bonds");
        missing_residue_templates.insert(name);
    }
    return tr;
}

void
ExtractMolecule::connect_polymer_pair(Residue* r0, Residue* r1, bool gap, bool nstd_okay, int model_num)
{
    // Connect adjacent residues that have the same type
    // and have link & chief atoms (i.e., peptides and nucleotides)
    const auto& r0name = r0->name();
    bool r0std = is_standard_residue(r0name);
    const auto& r1name = r1->name();
    bool r1std = is_standard_residue(r1name);
    if (!r0std || !r1std) {
        // One of the residues is non-standard, so there should be an explicit bond
        if (r0->connects_to(r1))
            return;
        // TODO: warning: missing expected bond
    }

    auto tr0 = find_template_residue(r0name);
    auto tr1 = find_template_residue(r1name);
    bool same_type = tr0 && tr1 && tr0->polymer_type() != PolymerType::PT_NONE
                     && (tr1->polymer_type() == tr0->polymer_type());
    if (r0std && r1std && !same_type) {
        // standard residues, but of different types, so there should be an explicit bond
        if (r0->connects_to(r1))
            return;
        // TODO: warning: missing expected bond
    }
    Atom* a0 = nullptr;
    Atom* a1 = nullptr;
    const char* conn_type;
    if (!same_type) {
        conn_type = "connection between ";
    } else {
        // peptide or nucletide
        auto ta0 = tr0 ? tr0->link() : nullptr;
        if (ta0) {
            a0 = r0->find_atom(ta0->name());
            if (a0 == nullptr) {
                // find closest heavy atom to ta0 in template that exists
                a0 = closest_atom_by_template(ta0, tr0, r0);
                if (a0)
                    gap = true;
            }
        }
        auto ta1 = tr1 ? tr1->chief() : nullptr;
        if (ta1) {
            a1 = r1->find_atom(ta1->name());
            if (a1 == nullptr) {
                // find closest heavy atom to ta1 in template that exists
                a1 = closest_atom_by_template(ta1, tr1, r1);
                if (a1)
                    gap = true;
            }
        }
        if (a0 == nullptr && a1 != nullptr) {
            std::swap(a0, a1);
            std::swap(r0, r1);
        }
        conn_type = "linking atoms for ";
    }
    if (a0 == nullptr) {
        pdb_connect::find_nearest_pair(r0, r1, &a0, &a1);
        if (a0 == nullptr || a0->element() != Element::C || a0->name() != "CA") {
            // suppress warning for CA traces and when missing templates
            if (model_num == first_model_num && !gap && tr0 && tr1)
                logger::warning(_logger, "Expected gap or ", conn_type,
                                r0->str(), " and ", r1->str());
        }
    } else if (a1 == nullptr) {
        a1 = pdb_connect::find_closest(a0, r1, nullptr, true);
        if (a1 == nullptr || a1->element() != Element::C || a1->name() != "CA") {
            // suppress warning for CA traces and when missing templates
            if (model_num == first_model_num && !gap && tr0 && tr1)
                logger::warning(_logger,
                                "Expected gap or linking atom in ",
                                r1->str(), " for ", r0->str());
        }
    }
    if (a1 == nullptr) {
        logger::warning(_logger, "Unable to connect ", r0->str(), " and ", r1->str());
        return;
    }
#if 0
    if (gap && reasonable_bond_length(a0, a1)) {
        logger::warning(_logger, "Eliding gap between ", r0-str(), " and ", r1->str());
        gap = false;    // bad data
    }
#endif
    if (same_type && !gap && !reasonable_bond_length(a0, a1) && r0->connects_to(r1)) {
        if (!nstd_okay)
            logger::warning(_logger,
                "Apparent non-polymeric linkage between ", r0->str(), " and ", r1->str());
        return;
    }
    if (gap || (!Bond::polymer_bond_atoms(a0, a1) && !reasonable_bond_length(a0, a1))) {
        // gap or CA trace
        auto as = r0->structure();
        auto pbg = as->pb_mgr().get_group(as->PBG_MISSING_STRUCTURE,
            atomstruct::AS_PBManager::GRP_NORMAL);
        pbg->new_pseudobond(a0, a1);
    } else if (!a0->connects_to(a1))
        (void) a0->structure()->new_bond(a0, a1);
}

// connect_residue_by_template:
//    Connect bonds in residue according to the given template.  Takes into
//    account alternate atom locations.
void
ExtractMolecule::connect_residue_by_template(Residue* r, const tmpl::Residue* tr, int model_num)
{
    auto& atoms = r->atoms();
    if (atoms.size() <= 1)
        return;

    // Confirm all atoms in residue are in template, if not connect by distance
    for (auto&& a: atoms) {
        tmpl::Atom *ta = tr->find_atom(a->name());
        if (!ta) {
            if (tr->atoms_map().size() == 0) {
                if (model_num == first_model_num
                && empty_residue_templates.find(r->name()) == empty_residue_templates.end()) {
                    empty_residue_templates.insert(r->name());
                    logger::warning(_logger, "Empty ", r->name(),
                                    " residue template");
                }
                // Fill in missing connectivity
                pdb_connect::connect_residue_by_distance(r);
                return;
            }
            bool connected = false;
            auto bonds = a->bonds();
            for (auto&& b: bonds) {
                if (b->other_atom(a)->residue() == r) {
                    connected = true;
                }
            }
            // TODO: worth checking if there is a metal coordination bond?
            if (!connected) {
                if (model_num == first_model_num) {
                    bool show_message = true;
                    if (a->element().number() == Element::H) {
                        const int threshold = 10;
                        hydrogens_missing_in_template += 1;
                        show_message = threshold > hydrogens_missing_in_template;
                        if (threshold == hydrogens_missing_in_template) {
                            logger::warning(_logger, "Too many hydrogens missing from "
                                            " residue template(s) to warn about ");
                        }
                    }
                    if (show_message)
                        logger::warning(_logger, "Atom ", a->name(),
                                        " is not in the residue template for ", r->str());
                }
                pdb_connect::connect_residue_by_distance(r);
                return;
            }
            // atom is connected, so assume template is still appropriate
        }
    }

    // foreach atom in residue
    //    connect up like atom in template
    for (auto&& a: atoms) {
        tmpl::Atom *ta = tr->find_atom(a->name());
        bool found_bond = false;
        bool has_heavy_neighbors = false;
        for (auto&& tmpl_nb: ta->neighbors()) {
            has_heavy_neighbors |= tmpl_nb->element().number() > Element::H;
            Atom *b = r->find_atom(tmpl_nb->name());
            if (b == nullptr)
                continue;
            found_bond = true;
            if (!a->connects_to(b))
                (void) a->structure()->new_bond(a, b);
        }
        if (!found_bond && has_heavy_neighbors)
            logger::warning(_logger, "Atom ", a->name(),
                            " has no neighbors to form bonds with according to residue template for ",
                            r->str());
    }
}

void
ExtractMolecule::finished_parse()
{
    if (molecules.empty())
        return;

    if (my_templates) {
        // small optimization (1% for 3j3q)
        bool has_atoms = false;
        for (auto& tri: my_templates->residues_map()) {
            auto tr = tri.second;
            if (tr->atoms_map().size() > 0) {
                has_atoms = true;
                break;
            }
        }
        if (!has_atoms) {
            // none of the templates have atoms, so get rid of them to speed things up
            delete my_templates;
            my_templates = nullptr;
        }
    }

    for (auto& mi: molecules) {
        auto model_num = mi.first;
        auto mol = mi.second;

        // fill in coord set for Monte-Carlo trajectories if necessary
        // (the last coord set might be too small)
        if (coordsets && mol->coord_sets().size() > 1) {
            CoordSet *acs = mol->active_coord_set();
            const CoordSet *prev_cs = mol->find_coord_set(acs->id() - 1);
            if (prev_cs != nullptr && acs->coords().size() < prev_cs->coords().size())
                acs->fill(prev_cs);
        }

        // Connect residues in entity_poly_seq.
        // Because some positions are heterogeneous, delay connecting
        // until next group of residues is found.
        // typedef unordered_set<Residue*, hash_ResidueKey> ResidueSet;
        typedef set<Residue*> ResidueSet;
        ResidueSet start_residues, stop_residues;
        bool no_polymer = true;
        for (auto& chain: all_residues[model_num]) {
            ResidueMap& residue_map = chain.second;
            if (residue_map.size() <= 1)
                continue;
            auto ri = residue_map.begin();
            const string& entity_id = ri->first.entity_id;
            if (non_poly.find(entity_id) != non_poly.end())
                continue;
            if (poly.find(entity_id) == poly.end())
                continue;
            const PolySeq* lastp = nullptr;
            bool gap = false;
            vector<Residue*> previous, current;
            ChainID auth_chain_id;
            auto& entity_poly = poly.at(entity_id);
            bool nstd = entity_poly.nstd;
            vector<ResName> seqres;
            vector<Residue *> residues;
            seqres.reserve(entity_poly.seq.size());
            residues.reserve(entity_poly.seq.size());
            no_polymer = no_polymer && entity_poly.seq.empty();
            auto& entity_poly_seq = poly.at(entity_id).seq;
            bool first = true;
            Residue* stop_residue = nullptr;
            for (auto pi = entity_poly_seq.begin(); pi != entity_poly_seq.end();) {
                auto p = *pi;
                auto pit = entity_poly_seq.equal_range(p);
                // count might be more than one if there is microheterogenatity 
                // or guessed sequence has duplicate seq_id's.  Only look at
                // one residue with a given seq_id
                auto count = std::distance(pit.first, pit.second);
                ResidueMap::iterator ri = residue_map.end();
                multiset<PolySeq>::iterator pi2;
                for (pi2 = pit.first; pi2 != pit.second; ++pi2) {
                    auto& p2 = *pi2;
                    ri = residue_map.find(ResidueKey(entity_id, p2.seq_id, p2.mon_id));
                    if (ri == residue_map.end())
                        continue;
                    p = p2;
                    break;
                }
                if (pi2 != pit.second) {
                    for (++pi2; pi2 != pit.second; ++pi2) {
                        // delete duplicates and microheterogeneity
                        auto& p2 = *pi2;
                        auto ri2 = residue_map.find(ResidueKey(entity_id, p2.seq_id, p2.mon_id));
                        if (ri2 == residue_map.end())
                            continue;
                        string c_id;
                        if (auth_chain_id == " ")
                            c_id = "' '";
                        else
                            c_id = auth_chain_id;
                        if (model_num == first_model_num) {
                            if (model_num != first_model_num)
                                ;  // only warn for first model
                            else if (p2.hetero)
                                logger::warning(_logger, "Ignoring microheterogeneity for label_seq_id ",
                                                p.seq_id, " in chain ", c_id);
                            else
                                logger::warning(_logger, "Skipping residue with duplicate label_seq_id ",
                                                p.seq_id, " in chain ", c_id);
                        }
                        Residue* r = ri2->second;
                        residue_map.erase(ri2);
                        mol->delete_residue(r);
                    }
                }
                if (ri == residue_map.end()) {
                    if (!lastp || lastp->seq_id != p.seq_id) {
                        seqres.push_back(p.mon_id);
                        residues.push_back(nullptr);
                        stop_residue = nullptr;
                    }
                    if (current.empty()) {
                        pi = pit.second;
                        first = false;
                        continue;
                    }
                    if (!previous.empty())
                        connect_polymer_pair(previous[0], current[0], gap, nstd, model_num);
                    previous = std::move(current);
                    current.clear();
                    if (!lastp || lastp->seq_id != p.seq_id) {
                        gap = true;
                    }
                    lastp = &*pi;
                    pi = pit.second;
                    first = false;
                    continue;
                }
                Residue* r = ri->second;
                seqres.push_back(p.mon_id);
                residues.push_back(r);
                if (first)
                    start_residues.insert(r);
                else
                    stop_residue = r;
                if (auth_chain_id.empty())
                    auth_chain_id = r->chain_id();
                if (!previous.empty() && !current.empty()) {
                    connect_polymer_pair(previous[0], current[0], gap, nstd, model_num);
                    gap = false;
                }
                if (!current.empty()) {
                    previous = std::move(current);
                    current.clear();
                }
                current.push_back(r);
                lastp = &*pi;
                pi = pit.second;
                first = false;
            }
            if (stop_residue != nullptr)
                stop_residues.insert(stop_residue);
            if (!previous.empty() && !current.empty())
                connect_polymer_pair(previous[0], current[0], gap, nstd, model_num);
            if (has_poly_seq.find(entity_id) == has_poly_seq.end())
                found_missing_poly_seq = true;
            else {
                if (entity_poly.ptype == PolymerType::PT_NONE)
                    mol->set_input_seq_info(auth_chain_id, seqres);
                else
                    mol->set_input_seq_info(auth_chain_id, seqres, &residues, entity_poly.ptype);
                if (mol->input_seq_source.empty())
                    mol->input_seq_source = "mmCIF entity_poly_seq table";
            }
        }

        // connect residues in molecule with all_residues information
        bool has_metal = false;
        for (auto&& r : mol->residues()) {
            bool start = start_residues.find(r) != start_residues.end();
            bool stop = stop_residues.find(r) != stop_residues.end();
            auto tr = find_template_residue(r->name(), start, stop);
            if (tr == nullptr) {
                if (model_num == first_model_num) {
                    // Warning already given about missing template
                    // logger::warning(_logger, "Missing or invalid residue template for ", r->str());
                    has_metal = true;   // it's okay to do extra work
                }
                pdb_connect::connect_residue_by_distance(r);
            } else {
                has_metal = has_metal || tr->has_metal();
                connect_residue_by_template(r, tr, model_num);
            }
        }

        if (found_missing_poly_seq && !no_polymer && model_num == first_model_num)
            logger::warning(_logger, "Missing or incomplete entity_poly_seq table.  Inferred polymer connectivity.");
        if (has_metal)
            pdb_connect::find_and_add_metal_coordination_bonds(mol);
        if (found_missing_poly_seq)
            pdb_connect::find_missing_structure_bonds(mol);
    }

    // export mapping of label chain ids to entity ids.
    StringVector chain_mapping;
    chain_mapping.reserve(chain_entity_map.size() * 2);
    for (auto i: chain_entity_map) {
        chain_mapping.emplace_back(i.first);
        chain_mapping.emplace_back(i.second);
    }
    generic_tables["struct_asym"] = { "struct_asym", "id", "entity_id" };
    generic_tables["struct_asym data"] = chain_mapping;

    // multiple molecules means there were multiple models,
    // so copy per-model information
    for (auto& im: molecules) {
        auto m = im.second;
        all_molecules.push_back(m);
        m->metadata = generic_tables;
        m->use_best_alt_locs();
        auto& chains = m->chains();
        for (auto& chain: chains) {
            auto label_asym_id = chain->res_map().begin()->first->mmcif_chain_id();
            auto cmi = chain_entity_map.find(label_asym_id);
            if (cmi == chain_entity_map.end())
                continue;
            auto edi = entity_description.find(cmi->second);
            if (edi == entity_description.end())
                continue;
            chain->set_description(edi->second);
        }
    }
    reset_parse();
}

void
ExtractMolecule::data_block(const string& /*name*/)
{
    if (!molecules.empty())
        finished_parse();
    else
        reset_parse();
}

void
ExtractMolecule::parse_entry()
{
    CIFFile::ParseValues pv;
    pv.reserve(1);
    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                entry_id = string(start, end - start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping entry category: ", e.what());
        return;
    }
    parse_row(pv);
    generic_tables["entry"] = { "entry", "id" };
    generic_tables["entry data"] = { entry_id };
}

void
ExtractMolecule::parse_pdbx_database_PDB_obs_spr()
{
    if (entry_id.empty())
        return;

    string id, pdb_id, replace_pdb_id;
    CIFFile::ParseValues pv;
    pv.reserve(3);
    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                id = string(start, end - start);
            });
        pv.emplace_back(get_column("pdb_id", Required),
            [&] (const char* start, const char* end) {
                pdb_id = string(start, end - start);
            });
        pv.emplace_back(get_column("replace_pdb_id", Required),
            [&] (const char* start, const char* end) {
                replace_pdb_id = string(start, end - start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping pdbx_database_PDB_obs_spr category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        if (id != "OBSLTE")
            continue;
        if (replace_pdb_id == entry_id)
            logger::warning(_logger, "PDB entry ", replace_pdb_id, " has been replaced by ",
                         pdb_id);
    }
}

void
ExtractMolecule::parse_generic_category()
{
    const string& category = this->category();
    const StringVector& colnames = this->colnames();
    string category_ci = category;
    for (auto& c: category_ci)
        c = tolower(c);
    StringVector colinfo;
    colinfo.reserve(colnames.size() + 1);
    colinfo.push_back(category);
    colinfo.insert(colinfo.end(), colnames.begin(), colnames.end());
    generic_tables[category_ci] = colinfo;
    StringVector& data = parse_whole_category();
    generic_tables[category_ci + " data"].swap(data);
}

void
ExtractMolecule::parse_chem_comp()
{
    ResName id;
    string  mon_nstd_flag;
    string  type;
    string  name;
    string  pdbx_synonyms;
    bool    ambiguous = false;
    StringVector col_names = { "id", "mon_nstd_flag", "type", "name", "pdbx_synonyms" };
    StringVector data;

    CIFFile::ParseValues pv;
    pv.reserve(4);
    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                id = ResName(start, end - start);
            });
        pv.emplace_back(get_column("mon_nstd_flag"),
            [&] (const char* start, const char* end) {
                mon_nstd_flag = string(start, end - start);
            });
        pv.emplace_back(get_column("type", Required),
            [&] (const char* start, const char* end) {
                type = string(start, end - start);
            });
        pv.emplace_back(get_column("name"),
            [&] (const char* start, const char* end) {
                name = string(start, end - start);
            });
        pv.emplace_back(get_column("pdbx_synonyms"),
            [&] (const char* start, const char* end) {
                pdbx_synonyms = string(start, end - start);
            });
        pv.emplace_back(get_column("pdbx_ambiguous_flag"),
            [&] (const char* start) {
                ambiguous = *start == 'Y' || *start == 'y';
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping chem_comp category: ", e.what());
        return;
    }

    if (my_templates == nullptr)
        my_templates = new tmpl::Molecule();
    while (parse_row(pv)) {
        data.push_back(id.c_str());
        data.push_back(mon_nstd_flag);
        data.push_back(type);
        data.push_back(name);
        data.push_back(pdbx_synonyms);

        tmpl::Residue* tr = my_templates->find_residue(id);
        if (tr)
            continue;

        tr = my_templates->new_residue(id.c_str());
        tr->pdbx_ambiguous = ambiguous;
        // convert type to lowercase
        for (auto& c: type) {
            if (isupper(c))
                c = tolower(c);
        }
        bool is_peptide = type.find("peptide") != string::npos;
        if (is_peptide)
            tr->polymer_type(PolymerType::PT_AMINO);
        else {
            bool is_nucleotide = type.compare(0, 3, "dna") == 0
                || type.compare(0, 3, "rna") == 0;
            if (is_nucleotide)
                tr->polymer_type(PolymerType::PT_NUCLEIC);
        }
    }
    StringVector colinfo;
    colinfo.reserve(col_names.size() + 1);
    colinfo.emplace_back("chem_comp");
    colinfo.insert(colinfo.end(), col_names.begin(), col_names.end());
    generic_tables["chem_comp"] = colinfo;
    generic_tables["chem_comp data"].swap(data);
}

void
ExtractMolecule::parse_chem_comp_bond()
{
    if (!my_templates)
        return;

    ResName  rname;
    AtomName aname1, aname2;

    CIFFile::ParseValues pv;
    pv.reserve(4);
    try {
        pv.emplace_back(get_column("comp_id", Required),
            [&] (const char* start, const char* end) {
                rname = ResName(start, end - start);
            });
        pv.emplace_back(get_column("atom_id_1", Required),
            [&] (const char* start, const char* end) {
                aname1 = AtomName(start, end - start);
            });
        pv.emplace_back(get_column("atom_id_2", Required),
            [&] (const char* start, const char* end) {
                aname2 = AtomName(start, end - start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping chem_comp_bond category: ", e.what());
        return;
    }
    // pretend all atoms are the same element. we only need connectivity,
    // so we are ignoring the chem_comp_atom table
    const Element& elem = Element::get_element("C");
    while (parse_row(pv)) {
        tmpl::Residue* tr = my_templates->find_residue(rname);
        if (!tr)
            continue;
        tmpl::Atom* a1 = tr->find_atom(aname1);
        if (!a1) {
            a1 = my_templates->new_atom(aname1, elem);
            tr->add_atom(a1);
        }
        tmpl::Atom* a2 = tr->find_atom(aname2);
        if (!a2) {
            a2 = my_templates->new_atom(aname2, elem);
            tr->add_atom(a2);
        }
        if (a1 != a2)
            my_templates->new_bond(a1, a2);
        else
            logger::info(_logger, "error in chem_comp_bond near line ",
                         line_number(), ": atom can not connect to itself");
    }

    // sneak in chief and link atoms
    for (auto& ri: my_templates->residues_map()) {
        tmpl::Residue* tr = ri.second;
        if (tr->polymer_type() == PolymerType::PT_AMINO) {
            tr->chief(tr->find_atom("N"));
            tr->link(tr->find_atom("C"));
        } else if (tr->polymer_type() == PolymerType::PT_NUCLEIC) {
            tr->chief(tr->find_atom("P"));
            tr->link(tr->find_atom("O3'"));
        }
    }
}

void
ExtractMolecule::parse_audit_conform()
{
    // Looking for a way to tell if the mmCIF file was written
    // in the PDBx/mmCIF stylized format.  The following technique
    // is not guaranteed to work, but we'll use it for now.
    string dict_name;
    float dict_version = 0;

    CIFFile::ParseValues pv;
    pv.reserve(2);
    try {
        pv.emplace_back(get_column("dict_name"),
            [&dict_name] (const char* start, const char* end) {
                dict_name = string(start, end - start);
            });
        pv.emplace_back(get_column("dict_version"),
            [&dict_version] (const char* start) {
                dict_version = strtof(start, NULL);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping audit_conform category: ", e.what());
        return;
    }
    parse_row(pv);
    if (dict_name == "mmcif_pdbx.dic" && dict_version > 4) {
        set_PDBx_keywords(true);
        guess_fixed_width_categories = true;
    }
    // If dict_name is core_std.dic, then it's a small molecule cif
    // if dict_name doesn't start with mmcif, the give a warning
}

void
ExtractMolecule::parse_audit_syntax()
{
    // Looking for a way to tell if the mmCIF file was written
    // in the PDBx/mmCIF stylized format.  The following technique
    // is not guaranteed to work, but we'll use it for now.
    bool case_sensitive = false;
    vector<string> fixed_width;
    fixed_width.reserve(12);

    CIFFile::ParseValues pv;
    pv.reserve(2);
    try {
        pv.emplace_back(get_column("case_sensitive_flag"),
            [&] (const char* start) {
                case_sensitive = *start == 'Y' || *start == 'y';
            });
        pv.emplace_back(get_column("fixed_width"),
            [&] (const char* start, const char* end) {
                for (const char *cp = start; cp < end; ++cp) {
                    if (isspace(*cp))
                        continue;
                    start = cp;
                    while (cp < end && !isspace(*cp))
                        ++cp;
                    fixed_width.push_back(string(start, cp - start));
                }
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping audit_syntax category: ", e.what());
        return;
    }
    parse_row(pv);
    set_PDBx_keywords(case_sensitive);
    guess_fixed_width_categories = false;
    for (auto& category: fixed_width)
        set_PDBx_fixed_width_columns(category);
}

void
ExtractMolecule::parse_atom_site()
{
    // x, y, z are not required by mmCIF, but are by us

    readcif::CIFFile::ParseValues pv;
    pv.reserve(20);

    string entity_id;             // label_entity_id
    ChainID chain_id;             // label_asym_id
    ChainID auth_chain_id;        // auth_asym_id
    long position;                // label_seq_id
    long auth_position = INT_MAX; // auth_seq_id
    char ins_code = ' ';          // pdbx_PDB_ins_code
    char alt_id = '\0';           // label_alt_id
    AtomName atom_name;           // label_atom_id
#if 0
    AtomName auth_atom_name;      // auth_atom_id
#endif
    ResName residue_name;         // label_comp_id
    ResName auth_residue_name;    // auth_comp_id
    char symbol[3];               // type_symbol
    long serial_num = 0;          // id
    double x, y, z;               // Cartn_[xyz]
    double occupancy = DBL_MAX;   // occupancy
    double b_factor = DBL_MAX;    // B_iso_or_equiv
    int model_num = 0;            // pdbx_PDB_model_num
    long user_position;           // auth_position if given else position

    if (guess_fixed_width_categories)
        set_PDBx_fixed_width_columns("atom_site");

    // If it has fractional coordinates, then it is a coreCIF file
    bool is_corecif = true;
    try {
        get_column("fract_x", Required);
    } catch (std::runtime_error& e) {
        is_corecif = false;
    }
    if (is_corecif) {
        throw std::runtime_error("is a small molecule (coreCIF) file");
    }

    try {
        pv.emplace_back(get_column("id"),
            [&] (const char* start) {
                serial_num = readcif::str_to_int(start);
            });

        pv.emplace_back(get_column("label_entity_id"),
            [&] (const char* start, const char* end) {
                entity_id = string(start, end - start);
                if (entity_id.size() == 1 && (*start == '.' || *start == '?'))
                    entity_id.clear();
            });

        pv.emplace_back(get_column("label_asym_id", Required),
            [&] (const char* start, const char* end) {
                chain_id = ChainID(start, end - start);
            });
        pv.emplace_back(get_column("auth_asym_id"),
            [&] (const char* start, const char* end) {
                auth_chain_id = ChainID(start, end - start);
                if (auth_chain_id.size() == 1 && (*start == '.' || *start == '?'))
                    auth_chain_id.clear();
            });
        pv.emplace_back(get_column("pdbx_PDB_ins_code"),
            [&] (const char* start, const char* end) {
                if (end == start + 1 && (*start == '.' || *start == '?'))
                    ins_code = ' ';
                else {
                    // TODO: check if more than one character
                    ins_code = *start;
                }
            });
        pv.emplace_back(get_column("label_seq_id", Required),
            [&] (const char* start) {
                position = readcif::str_to_int(start);
            });
        pv.emplace_back(get_column("auth_seq_id"),
            [&] (const char* start) {
                if (*start == '.' || *start == '?')
                    auth_position = INT_MAX;
                else
                    auth_position = readcif::str_to_int(start);
            });

        pv.emplace_back(get_column("label_alt_id"),
            [&] (const char* start, const char* end) {
                if (end == start + 1
                && (*start == '.' || *start == '?' || *start == ' '))
                    alt_id = '\0';
                else {
                    // TODO: what about more than one character?
                    alt_id = *start;
                }
            });
        pv.emplace_back(get_column("type_symbol", Required),
            [&] (const char* start) {
                symbol[0] = *start;
                symbol[1] = *(start + 1);
                if (readcif::is_whitespace(symbol[1]))
                    symbol[1] = '\0';
                else
                    symbol[2] = '\0';
            });
        pv.emplace_back(get_column("label_atom_id", Required),
            [&] (const char* start, const char* end) {
                // deal with Coot's braindead leading and trailing
                // spaces in atom names
                while (isspace(*start))
                    ++start;
                while (end > start && isspace(*(end - 1)))
                    --end;
                atom_name = AtomName(start, end - start);
            });
#if 0
        pv.emplace_back(get_column("auth_atom_id"),
            [&] (const char* start, const char* end) {
                auth_atom_name = AtomName(start, end - start);
                if (auth_atoms_name.size() == 1 && (*start == '.' || *start == '?'))
                    auth_atom_name.clear();
            });
#endif
        pv.emplace_back(get_column("label_comp_id", Required),
            [&] (const char* start, const char* end) {
                residue_name = ResName(start, end - start);
            });
        pv.emplace_back(get_column("auth_comp_id"),
            [&] (const char* start, const char* end) {
                auth_residue_name = ResName(start, end - start);
                if (auth_residue_name.size() == 1 && (*start == '.' || *start == '?'))
                    auth_residue_name.clear();
            });
        // x, y, z are not required by mmCIF, but are by us
        pv.emplace_back(get_column("Cartn_x", Required),
            [&] (const char* start) {
                x = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("Cartn_y", Required),
            [&] (const char* start) {
                y = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("Cartn_z", Required),
            [&] (const char* start) {
                z = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("occupancy"),
            [&] (const char* start) {
                if (*start == '?')
                    occupancy = DBL_MAX;
                else
                    occupancy = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("B_iso_or_equiv"),
            [&] (const char* start) {
                if (*start == '?')
                    b_factor = DBL_MAX;
                else
                    b_factor = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("pdbx_PDB_model_num"),
            [&] (const char* start) {
                model_num = readcif::str_to_int(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping atom_site category: ", e.what());
        return;
    }

    long atom_serial = 0;
    Residue* cur_residue = nullptr;
    Structure* mol = nullptr;
    int cur_model_num = INT_MAX;
    // residues are uniquely identified by (entity_id, seq_id, comp_id)
    string cur_entity_id;
    int cur_seq_id = INT_MAX;
    int cur_auth_seq_id = INT_MAX;
    ChainID cur_chain_id;
    ResName cur_comp_id;
    bool missing_seq_id_warning = false;
    bool missing_entity_id_warning = false;
    for (;;) {
        if (!parse_row(pv))
            break;
        if (model_num != cur_model_num) {
            if (first_model_num == INT_MAX)
                first_model_num = model_num;
            cur_model_num = model_num;
            cur_residue = nullptr;
            if (!coordsets) {
                if (atomic) {
                    mol = molecules[cur_model_num] = new AtomicStructure(_logger);
                } else {
                    mol = molecules[cur_model_num] = new Structure(_logger);
                }
            } else {
                if (mol == nullptr) {
                    if (atomic) {
                        mol = new AtomicStructure(_logger);
                    } else {
                        mol = new Structure(_logger);
                    }
                    molecules[model_num] = mol;
                    CoordSet *cs = mol->new_coord_set(model_num);
                    mol->set_active_coord_set(cs);
                } else {
                    // make additional CoordSets same size as others
                    size_t cs_size = mol->active_coord_set()->coords().size();
                    if (cur_model_num > mol->active_coord_set()->id() + 1) {
                        // fill in coord sets for Monte-Carlo trajectories
                        const CoordSet *acs = mol->active_coord_set();
                        for (int fill_in_ID = acs->id() + 1; fill_in_ID < cur_model_num; ++fill_in_ID) {
                            CoordSet *cs = mol->new_coord_set(fill_in_ID, cs_size);
                            cs->fill(acs);
                        }
                    }
                    CoordSet *cs = mol->new_coord_set(cur_model_num, cs_size);
                    mol->set_active_coord_set(cs);
                }
            }
            mol->set_res_numbering_valid(RN_CANONICAL, true);
        }

        bool missing_entity_id = entity_id.empty();
        if (missing_entity_id)
            entity_id = chain_id;  // no entity_id, use chain id
        bool missing_position = position == 0;
        if (missing_position)
            position = auth_position;
        if (auth_position == INT_MAX)
            user_position = position;
        else
            user_position = auth_position;

        if (cur_residue == nullptr
        || cur_entity_id != entity_id
        || cur_seq_id != position
        || cur_auth_seq_id != auth_position
        || cur_chain_id != chain_id
        || cur_comp_id != residue_name) {
            ResName rname;
            ChainID cid;
            if (!auth_residue_name.empty())
                rname = auth_residue_name;
            else
                rname = residue_name;
            if (!auth_chain_id.empty())
                cid = auth_chain_id;
            else
                cid = chain_id;
            bool make_new_residue = true;
            if (coordsets) {
                auto& res_map = all_residues[first_model_num][chain_id];
                if (!res_map.empty()) {
                    auto ri = res_map.find(ResidueKey(entity_id, position, residue_name));
                    if (ri != res_map.end()) {
                        make_new_residue = false;
                        cur_residue = ri->second;
                    }
                }
            }
            if (make_new_residue) {
                cur_residue = mol->new_residue(rname, cid, user_position, ins_code);
                cur_residue->set_mmcif_chain_id(chain_id);
                cur_residue->set_number(RN_CANONICAL, position);
            }
            cur_entity_id = entity_id;
            cur_seq_id = position;
            cur_auth_seq_id = auth_position;
            cur_chain_id = chain_id;
            cur_comp_id = residue_name;
            if (has_poly_seq.find(entity_id) == has_poly_seq.end()
            && non_poly.find(entity_id) == non_poly.end()) {
                // sequence not given in entity_poly_seq and not in a known non-polymeric entity
                auto tr = find_template_residue(residue_name);
                if (tr && tr->polymer_type() != PolymerType::PT_NONE) {
                    // only save polymer residues
                    if (position == INT_MAX) {
                        if (!missing_seq_id_warning) {
                            logger::warning(_logger, "Unable to infer polymer connectivity due to "
                                            "unspecified label_seq_id for residue \"",
                                            residue_name, "\" near line ", line_number());
                           missing_seq_id_warning = true;
                        }
                    } else {
                        if (poly.find(entity_id) == poly.end()) {
                            if (missing_entity_id) {
                                if (!missing_entity_id_warning) {
                                    logger::warning(_logger, "Missing entity information.  "
                                                    "Treating each chain as a separate entity.");
                                    missing_entity_id_warning = true;
                                }
                            } else {
                                logger::warning(_logger, "Unknown polymer entity '", entity_id,
                                                "' near line ", line_number());
                            }
                            // fake polymer entity to cut down on secondary warnings
                            poly.emplace(entity_id, false);
                        }
                        auto& entity_poly_seq = poly.at(entity_id).seq;
                        PolySeq p(position, residue_name, false);
                        auto pit = entity_poly_seq.equal_range(p);
                        bool found = false;
                        for (auto& i = pit.first; i != pit.second; ++i) {
                            auto& p2 = *i;
                            if (p2.mon_id == p.mon_id) {
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                            entity_poly_seq.emplace(p);
                    }
                }
            }
            chain_entity_map[chain_id] = entity_id;
            all_residues[model_num][chain_id]
                [ResidueKey(entity_id, position, residue_name)] = cur_residue;
        }

        if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
            logger::warning(_logger, "Skipping atom \"", atom_name,
                            "\" near line ", line_number(),
                            ": missing coordinates");
            continue;
        }
        canonicalize_atom_name(&atom_name, &mol->asterisks_translated);

        bool make_new_atom = true;
        Atom* a;
        if (alt_id && cur_residue->count_atom(atom_name) == 1) {
            make_new_atom = false;
            a = cur_residue->find_atom(atom_name);
            a->set_alt_loc(alt_id, true);
        } else if (coordsets && cur_model_num != first_model_num) {
            a = cur_residue->find_atom(atom_name);
            if (a != nullptr)
                make_new_atom = false;
        }
        if (make_new_atom) {
            const Element& elem = Element::get_element(symbol);
            a = mol->new_atom(atom_name.c_str(), elem);
            cur_residue->add_atom(a);
            if (alt_id)
                a->set_alt_loc(alt_id, true);
            if (serial_num)
                atom_serial = serial_num;
            else
                ++atom_serial;
            a->set_serial_number(atom_serial);
        }
        Coord c(x, y, z);
        a->set_coord(c);
        if (b_factor != DBL_MAX)
            a->set_bfactor(b_factor);
        if (occupancy != DBL_MAX)
            a->set_occupancy(occupancy);
        if (serial_num)
            atom_lookup[serial_num] = {a, alt_id};
    }
}

void
ExtractMolecule::parse_atom_site_anisotrop()
{
    readcif::CIFFile::ParseValues pv;
    pv.reserve(20);

    if (guess_fixed_width_categories)
        set_PDBx_fixed_width_columns("atom_site_anisotrop");

    long serial_num = 0;          // id
    double u11, u12, u13, u22, u23, u33;

    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start) {
                serial_num = readcif::str_to_int(start);
            });
        pv.emplace_back(get_column("U[1][1]", Required),
            [&] (const char* start) {
                u11 = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("U[1][2]", Required),
            [&] (const char* start) {
                u12 = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("U[1][3]", Required),
            [&] (const char* start) {
                u13 = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("U[2][2]", Required),
            [&] (const char* start) {
                u22 = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("U[2][3]", Required),
            [&] (const char* start) {
                u23 = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("U[3][3]", Required),
            [&] (const char* start) {
                u33 = readcif::str_to_float(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping atom_site_anistrop category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        const auto& ai = atom_lookup.find(serial_num);
        if (ai == atom_lookup.end())
            continue;
        Atom *a = ai->second.first;
        char alt_id = ai->second.second;
        if (alt_id)
            a->set_alt_loc(alt_id, false);
        a->set_aniso_u(u11, u12, u13, u22, u23, u33);
    }
}

void
ExtractMolecule::parse_struct_conn()
{
    if (molecules.empty())
        return;

    // these strings are concatenated to make the column headers needed
    #define P1 "ptnr1"
    #define P2 "ptnr2"
    #define ASYM_ID "_label_asym_id"
    #define COMP_ID "_label_comp_id"
    #define SEQ_ID "_label_seq_id"
    #define AUTH_SEQ_ID "_auth_seq_id"
    #define ATOM_ID "_label_atom_id"
    #define ALT_ID "_label_alt_id" // pdbx
    #define SYMMETRY "_symmetry"

    // bonds from struct_conn records
    ChainID chain_id1, chain_id2;           // ptnr[12]_label_asym_id
    long position1, position2;              // ptnr[12]_label_seq_id
    long auth_position1 = INT_MAX,          // ptnr1_auth_seq_id
         auth_position2 = INT_MAX;          // ptnr2_auth_seq_id
    //char alt_id1 = '\0', alt_id2 = '\0';    // pdbx_ptnr[12]_label_alt_id
    AtomName atom_name1, atom_name2;        // ptnr[12]_label_atom_id
    ResName residue_name1, residue_name2;   // ptnr[12]_label_comp_id
    string conn_id("(missing id)");         // id
    string conn_type;                       // conn_type_id
    string symmetry1, symmetry2;            // ptnr[12]_symmetry
    float distance = std::numeric_limits<float>::quiet_NaN();  // pdbx_dist_value
    std::set<std::pair<Atom*, Atom*>>   metal_bonds, hydrogen_bonds;

    CIFFile::ParseValues pv;
    pv.reserve(32);
    try {
        pv.emplace_back(get_column("id"),
            [&] (const char* start, const char* end) {
                conn_id = string(start, end - start);
            });
        pv.emplace_back(get_column("conn_type_id", Required),
            [&] (const char* start, const char* end) {
                conn_type = string(start, end - start);
                for (auto& c: conn_type) {
                    if (isupper(c))
                        c = tolower(c);
                }
            });
        pv.emplace_back(get_column(P1 ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id1 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(P1 SEQ_ID, Required),
            [&] (const char* start) {
                position1 = readcif::str_to_int(start);
            });
        pv.emplace_back(get_column(P1 AUTH_SEQ_ID),
            [&] (const char* start) {
                if (*start == '.' || *start == '?')
                    auth_position1 = INT_MAX;
                else
                    auth_position1 = readcif::str_to_int(start);
            });
#if 0
        pv.emplace_back(get_column("pdbx_" P1 ALT_ID),
            [&] (const char* start, const char* end) {
                if (end == start + 1
                && (*start == '.' || *start == '?' || *start == ' '))
                    alt_id1 = '\0';
                else {
                    // TODO: what about more than one character?
                    alt_id1 = *start;
                }
            });
#endif
        pv.emplace_back(get_column(P1 ATOM_ID, Required),
            [&] (const char* start, const char* end) {
                atom_name1 = AtomName(start, end - start);
            });
        pv.emplace_back(get_column(P1 COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name1 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(P1 SYMMETRY),
            [&] (const char* start, const char* end) {
                symmetry1 = string(start, end - start);
            });

        pv.emplace_back(get_column(P2 ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id2 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(P2 SEQ_ID, Required),
            [&] (const char* start) {
                position2 = readcif::str_to_int(start);
            });
        pv.emplace_back(get_column(P2 AUTH_SEQ_ID),
            [&] (const char* start) {
                if (*start == '.' || *start == '?')
                    auth_position2 = INT_MAX;
                else
                    auth_position2 = readcif::str_to_int(start);
            });
#if 0
        pv.emplace_back(get_column("pdbx_" P2 ALT_ID),
            [&] (const char* start, const char* end) {
                if (end == start + 1
                && (*start == '.' || *start == '?' || *start == ' '))
                    alt_id2 = '\0';
                else {
                    // TODO: what about more than one character?
                    alt_id2 = *start;
                }
            });
#endif
        pv.emplace_back(get_column(P2 ATOM_ID, Required),
            [&] (const char* start, const char* end) {
                atom_name2 = AtomName(start, end - start);
            });
        pv.emplace_back(get_column(P2 COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name2 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(P2 SYMMETRY),
            [&] (const char* start, const char* end) {
                symmetry2 = string(start, end - start);
            });
        pv.emplace_back(get_column("pdbx_dist_value"),
            [&] (const char* start) {
                distance = readcif::str_to_float(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping struct_conn category: ", e.what());
        return;
    }

#if 0
    // now need one per model_num
    atomstruct::Proxy_PBGroup* metal_pbg = nullptr;
    atomstruct::Proxy_PBGroup* hydro_pbg = nullptr;
    atomstruct::Proxy_PBGroup* missing_pbg = nullptr;
#endif
    // connect residues in molecule with all_residues information
    while (parse_row(pv)) {
        if (symmetry1 != symmetry2)
            continue;
        if (atom_name1 == "?" || atom_name2 == "?")
            continue;
        bool normal = false;
        bool metal = false;
        bool hydro = false;
        // TODO: survey PDB mmCIF files and test in descending prevalence
        if (conn_type.compare(0, 6, "covale") == 0 || conn_type == "disulf")
            normal = true;
        else if (conn_type == "hydrog")
            hydro = true;
        else if (conn_type == "metalc")
            metal = true;
        if (!normal && !metal && !hydro)
            continue;   // skip modres and unknown connection types
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end())
            continue;
        if (position1 == 0)
            position1 = auth_position1;
        if (position2 == 0)
            position2 = auth_position2;
        auto entity1 = cemi->second;
        ResidueKey rk1(entity1, position1, residue_name1);
        cemi = chain_entity_map.find(chain_id2);
        if (cemi == chain_entity_map.end())
            continue;
        auto entity2 = cemi->second;
        ResidueKey rk2(entity2, position2, residue_name2);
        for (auto& mi: molecules) {
            auto model_num = mi.first;
            auto mol = mi.second;
            auto ari = all_residues.find(model_num);
            if (ari == all_residues.end())
                continue;
            auto& crm = ari->second;
            Residue* r1 = find_residue(crm, chain_id1, rk1);
            if (!r1) {
                logger::warning(_logger, "Missing first residue in struct_conn \"",
                                conn_id, "\"");
                continue;
            }
            Atom* a1 = r1->find_atom(atom_name1);
            if (!a1) {
                logger::warning(_logger, "Missing first atom in struct_conn \"",
                                conn_id, "\"");
                continue;
            }
            Residue* r2 = find_residue(crm, chain_id2, rk2);
            if (!r2) {
                logger::warning(_logger, "Missing second residue in struct_conn \"",
                                conn_id, "\"");
                continue;
            }
            Atom* a2 = r2->find_atom(atom_name2);
            if (!a2) {
                logger::warning(_logger, "Missing second atom in struct_conn \"",
                                conn_id, "\"");
                continue;
            }
            if (metal) {
                // make sure only once metal coordination bond is created between atoms
                if (a2 < a1)
                    std::swap(a1, a2);
                auto key = std::make_pair(a1, a2);
                if (metal_bonds.find(key) != metal_bonds.end())
                    continue;  // might be connected to alternate position
                auto metal_pbg = mol->pb_mgr().get_group(mol->PBG_METAL_COORDINATION,
                        atomstruct::AS_PBManager::GRP_PER_CS);
                for (auto& cs: mol->coord_sets()) {
                    metal_pbg->new_pseudobond(a1, a2, cs);
                }
                metal_bonds.insert(key);
                continue;
            }
            if (hydro) {
                // make sure only once hydrogen bond is created between atoms
                if (a2 < a1)
                    std::swap(a1, a2);
                auto key = std::make_pair(a1, a2);
                if (hydrogen_bonds.find(key) != hydrogen_bonds.end())
                    continue;  // might be connected to alternate position
                auto hydro_pbg = mol->pb_mgr().get_group(mol->PBG_HYDROGEN_BONDS,
                        atomstruct::AS_PBManager::GRP_PER_CS);
                for (auto& cs: mol->coord_sets()) {
                    hydro_pbg->new_pseudobond(a1, a2, cs);
                }
                hydrogen_bonds.insert(key);
                continue;
            }
            if (!reasonable_bond_length(a1, a2, distance)) {
                auto missing_pbg = mol->pb_mgr().get_group(
                        mol->PBG_MISSING_STRUCTURE,
                        atomstruct::AS_PBManager::GRP_NORMAL);
                missing_pbg->new_pseudobond(a1, a2);
                continue;
            }
            try {
                mol->new_bond(a1, a2);
            } catch (std::invalid_argument&) {
                // already bonded, might be connected to alternate position
            }
        }
    }
    #undef P1
    #undef P2
    #undef ASYM_ID
    #undef COMP_ID
    #undef SEQ_ID
    #undef AUTH_SEQ_ID
    #undef ATOM_ID
    #undef ALT_ID
    #undef INS_CODE
    #undef SYMMETRY
}

void
ExtractMolecule::parse_struct_conf()
{
    if (molecules.empty())
        return;

    // these strings are concatenated to make the column headers needed
    #define BEG "beg"
    #define END "end"
    #define ASYM_ID "_label_asym_id"
    #define COMP_ID "_label_comp_id"
    #define SEQ_ID "_label_seq_id"
    string conf_type;                       // conf_type_id
    string id;                              // id
    ChainID chain_id1, chain_id2;            // (beg|end)_label_asym_id
    long position1, position2;              // (beg|end)_label_seq_id
    ResName residue_name1, residue_name2;    // (beg|end)_label_comp_id

    CIFFile::ParseValues pv;
    pv.reserve(14);
    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                id = string(start, end - start);
            });
        pv.emplace_back(get_column("conf_type_id", Required),
            [&] (const char* start, const char* end) {
                conf_type = string(start, end - start);
            });

        pv.emplace_back(get_column(BEG ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id1 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(BEG COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name1 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(BEG SEQ_ID, Required),
            [&] (const char* start) {
                position1 = readcif::str_to_int(start);
            });

        pv.emplace_back(get_column(END ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id2 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(END COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name2 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(END SEQ_ID, Required),
            [&] (const char* start) {
                position2 = readcif::str_to_int(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping struct_conf category: ", e.what());
        return;
    }

    #undef BEG
    #undef END
    #undef ASYM_ID
    #undef COMP_ID
    #undef SEQ_ID
    #undef INS_CODE

    int helix_id = 0;
    int strand_id = 0;
    map<ChainID, int> strand_ids;
    ChainID last_chain_id;
    while (parse_row(pv)) {
        if (conf_type.empty())
            continue;
        if (chain_id1 != chain_id2) {
            logger::warning(_logger, "Start and end residues of struct_conf \"", id,
                          "\" are in different chains near line ", line_number());
            continue;
        }
        // Only expect helixes and turns, strands were in mmCIF v. 2,
        // but are not in mmCIF v. 4.
        bool is_helix = conf_type[0] == 'H' || conf_type[0] == 'h';
        bool is_strnd = conf_type[0] == 'S' || conf_type[0] == 's';
        if (!is_helix && !is_strnd) {
            // ignore turns
            continue;
        }
        if (is_helix)
            ++helix_id;
        else if (is_strnd) {
            auto si = strand_ids.find(chain_id1);
            if (si == strand_ids.end()) {
                strand_ids[chain_id1] = 1;
                strand_id = 1;
            } else {
                strand_id = ++(si->second);
            }
        }

        auto ari = all_residues[first_model_num].find(chain_id1);
        if (ari == all_residues[first_model_num].end()) {
            logger::warning(_logger, "Invalid residue range for struct_conf \"",
                            id, "\": invalid chain \"", chain_id1,
                            "\", near line ", line_number());
            continue;
        }
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid residue range for struct_conf \"",
                            id, "\": invalid chain \"", chain_id1,
                            "\", near line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly.find(entity_id);
        if (psi == poly.end()) {
            logger::warning(_logger, "Invalid residue range for struct_conf \"",
                            id, "\": invalid entity \"", entity_id,
                            "\", near line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second.seq;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for struct_conf \"",
                            id, "\": ends before it starts"
                            ", near line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end()) {}
            logger::warning(_logger,
                            "Bad residue range for struct_conf \"", id,
                            "\" near line ", line_number());
            continue;
        }
        for (auto& mi: molecules) {
            auto model_num = mi.first;
            auto ari = all_residues.find(model_num);
            if (ari == all_residues.end())
                continue;
            auto& crm = ari->second;
            const auto& crmi = crm.find(chain_id1);
            if (crmi == crm.end())
                continue;
            const ResidueMap& residue_map = crmi->second;
            for (auto pi = init_ps; pi != end_ps; ++pi) {
                auto ri = residue_map.find(ResidueKey(entity_id, pi->seq_id,
                                                      pi->mon_id));
                if (ri == residue_map.end())
                    continue;
                Residue *r = ri->second;
                if (is_helix) {
                    r->set_is_helix(true);
                    r->set_ss_id(helix_id);
                } else {
                    if (chain_id1 != last_chain_id) {
                        strand_id = 1;
                        last_chain_id = chain_id1;
                    }
                    r->set_is_strand(true);
                    r->set_ss_id(strand_id);
                }
            }
        }
    }
}

void
ExtractMolecule::parse_struct_sheet_range()
{
    if (molecules.empty())
        return;
    //
    // these strings are concatenated to make the column headers needed
    #define BEG "beg"
    #define END "end"
    #define ASYM_ID "_label_asym_id"
    #define COMP_ID "_label_comp_id"
    #define SEQ_ID "_label_seq_id"
    string sheet_id;                        // sheet_id
    string id;                              // id
    ChainID chain_id1, chain_id2;           // (beg|end)_label_asym_id
    long position1, position2;              // (beg|end)_label_seq_id
    ResName residue_name1, residue_name2;   // (beg|end)_label_comp_id

    CIFFile::ParseValues pv;
    pv.reserve(14);
    try {
        pv.emplace_back(get_column("sheet_id", Required),
            [&] (const char* start, const char* end) {
                sheet_id = string(start, end - start);
            });
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                id = string(start, end - start);
            });

        pv.emplace_back(get_column(BEG ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id1 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(BEG COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name1 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(BEG SEQ_ID, Required),
            [&] (const char* start) {
                position1 = readcif::str_to_int(start);
            });

        pv.emplace_back(get_column(END ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id2 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(END COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name2 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(END SEQ_ID, Required),
            [&] (const char* start) {
                position2 = readcif::str_to_int(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping struct_sheet_range category: ", e.what());
        return;
    }

    #undef BEG
    #undef END
    #undef ASYM_ID
    #undef COMP_ID
    #undef SEQ_ID
    #undef INS_CODE

    map<ChainID, int> strand_ids;
    while (parse_row(pv)) {
        if (chain_id1 != chain_id2) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": different chains"
                            ", near line ", line_number());
            continue;
        }

        auto ari = all_residues[first_model_num].find(chain_id1);
        if (ari == all_residues[first_model_num].end()) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": invalid chain \"",
                            chain_id1, "\", near line ", line_number());
            continue;
        }
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": invalid chain \"",
                            chain_id1, "\", near line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly.find(entity_id);
        if (psi == poly.end()) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": invalid entity \"",
                            entity_id, "\", near line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second.seq;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": ends before it starts"
                            ", near line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end())
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\" near line ", line_number());
            continue;
        }
#ifdef SHEET_HBONDS
        auto& sheet_residues = strand_info[std::make_pair(sheet_id, id)];
        sheet_residues.reserve(std::distance(init_ps, end_ps));
#endif
        int strand_id;
        auto si = strand_ids.find(chain_id1);
        if (si == strand_ids.end()) {
            strand_ids[chain_id1] = 1;
            strand_id = 1;
        } else {
            strand_id = ++(si->second);
        }
        for (auto& mi: molecules) {
            auto model_num = mi.first;
            auto ari = all_residues.find(model_num);
            if (ari == all_residues.end())
                continue;
            auto& crm = ari->second;
            const auto& crmi = crm.find(chain_id1);
            if (crmi == crm.end())
                continue;
            const ResidueMap& residue_map = crmi->second;
            for (auto pi = init_ps; pi != end_ps; ++pi) {
                auto ri = residue_map.find(ResidueKey(entity_id, pi->seq_id,
                                                      pi->mon_id));
                if (ri == residue_map.end()) {
#ifdef SHEET_HBONDS
                    sheet_residues.push_back(nullptr);
#endif
                    continue;
                }
                Residue *r = ri->second;
                r->set_is_strand(true);
                r->set_ss_id(strand_id);
#ifdef SHEET_HBONDS
                sheet_residues.push_back(r);
#endif
            }
        }
    }
}

#ifdef SHEET_HBONDS
void
ExtractMolecule::parse_struct_sheet_order()
{
    string sheet_id;
    string id1, id2;
    string sense;

    CIFFile::ParseValues pv;
    pv.reserve(5);
    try {
        pv.emplace_back(get_column("sheet_id", Required),
            [&] (const char* start, const char* end) {
                sheet_id = string(start, end - start);
            });
        pv.emplace_back(get_column("range_id_1", Required),
            [&] (const char* start, const char* end) {
                id1 = string(start, end - start);
            });
        pv.emplace_back(get_column("range_id_2", Required),
            [&] (const char* start, const char* end) {
                id2 = string(start, end - start);
            });
        pv.emplace_back(get_column("sense", Required),
            [&] (const char* start, const char* end) {
                sense = string(start, end - start);
                for (auto& c: sense) {
                    if (isupper(c))
                        c = tolower(c);
                }
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping struct_sheet_range category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        auto range = std::make_pair(id1, id2);
        sheet_order[sheet_id][range] = sense;
    }
}

void
ExtractMolecule::parse_pdbx_struct_sheet_hbond()
{
    if (molecules.empty())
        return;

    #define RANGE1 "range_1"
    #define RANGE2 "range_2"
    #define ATOM_ID "_label_atom_id"
    #define COMP_ID "_label_comp_id"
    #define ASYM_ID "_label_asym_id"
    #define SEQ_ID "_label_seq_id"

    // hbonds from pdbx_struct_sheet_hbond records
    string sheet_id;
    string id1, id2;
    ChainID chain_id1, chain_id2;           // range_[12]_label_asym_id
    long position1, position2;              // range_[12]_label_seq_id
    AtomName atom_name1, atom_name2;        // range_[12]_label_atom_id
    ResName residue_name1, residue_name2;   // range_[12]_label_comp_id

    CIFFile::ParseValues pv;
    pv.reserve(32);
    try {
        pv.emplace_back(get_column("sheet_id", Required),
            [&] (const char* start, const char* end) {
                sheet_id = string(start, end - start);
            });
        pv.emplace_back(get_column("range_id_1", Required),
            [&] (const char* start, const char* end) {
                id1 = string(start, end - start);
            });
        pv.emplace_back(get_column("range_id_2", Required),
            [&] (const char* start, const char* end) {
                id2 = string(start, end - start);
            });
        pv.emplace_back(get_column(RANGE1 ATOM_ID, Required),
            [&] (const char* start, const char* end) {
                atom_name1 = AtomName(start, end - start);
            });
        pv.emplace_back(get_column(RANGE1 COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name1 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(RANGE1 ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id1 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(RANGE1 SEQ_ID, Required),
            [&] (const char* start) {
                position1 = readcif::str_to_int(start);
            });
        pv.emplace_back(get_column(RANGE2 ATOM_ID, Required),
            [&] (const char* start, const char* end) {
                atom_name2 = AtomName(start, end - start);
            });
        pv.emplace_back(get_column(RANGE2 COMP_ID, Required),
            [&] (const char* start, const char* end) {
                residue_name2 = ResName(start, end - start);
            });
        pv.emplace_back(get_column(RANGE2 ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id2 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column(RANGE2 SEQ_ID, Required),
            [&] (const char* start) {
                position2 = readcif::str_to_int(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping pdbx_struct_sheet_hbond category: ", e.what());
        return;
    }
    #undef RANGE1
    #undef RANGE2
    #undef ATOM_ID
    #undef COMP_ID
    #undef ASYM_ID
    #undef SEQ_ID

    static AtomName xray_atom_names[2] = { "O", "N" };
    static AtomName nmr_atom_names[2] = { "O", "H" };
    atomstruct::Proxy_PBGroup* hydro_pbg = nullptr;
    // TODO: all models
    auto mol = all_residues[first_model_num].begin()->second.begin()->second->structure();
    while (parse_row(pv)) {
        if (hydro_pbg == nullptr)
            hydro_pbg = mol->pb_mgr().get_group(mol->PBG_HYDROGEN_BONDS,
                atomstruct::AS_PBManager::GRP_PER_CS);

        Residue* r1 = find_residue(chain_id1, position1, residue_name1);
        if (r1 == nullptr) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find residue /",
                            chain_id1, ":", position1, " ", residue_name1);
            continue;
        }
        Residue* r2 = find_residue(chain_id2, position2, residue_name2);
        if (r2 == nullptr) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find residue /",
                            chain_id2, ":", position2, " ", residue_name2);
            continue;
        }

        // make initial bond
        Atom* a1 = r1->find_atom(atom_name1); 
        if (a1 == nullptr) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find atom ",
                            atom_name1, " in ", r1->str());
        }
        Atom* a2 = r2->find_atom(atom_name2); 
        if (a2 == nullptr) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find atom ",
                            atom_name2, " in ", r2->str());
        }
        if (a1 != nullptr && a2 != nullptr) {
            for (auto& cs: mol->coord_sets()) {
                hydro_pbg->new_pseudobond(a1, a2, cs);
            }
        }

        // walk strands and create hbonds
        AtomName* atom_names;
        if (atom_name1 == "H" || atom_name2 == "H")
            atom_names = nmr_atom_names;
        else
            atom_names = xray_atom_names;
        auto range = std::make_pair(id1, id2);
        const string& sense = sheet_order[sheet_id][range];
        auto& strand1 = strand_info[std::make_pair(sheet_id, id1)];
        if (strand1.empty()) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find strand ",
                            id1, " in sheet ", sheet_id);
            continue;
        }
        auto& strand2 = strand_info[std::make_pair(sheet_id, id2)];
        if (strand1.empty()) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find strand ",
                            id2, " in sheet ", sheet_id);
            continue;
        }
        int n1 = std::find(atom_names, atom_names + 2, atom_name1) - atom_names;
        int n2 = std::find(atom_names, atom_names + 2, atom_name2) - atom_names;
        if (n1 == 2) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: unexpected atom name ", atom_name1);
            continue;
        }
        if (n2 == 2) {
            logger::warning(_logger, "pdbx_stuct_sheet_hbond: unexpected atom name ", atom_name2);
            continue;
        }
        if (sense == "parallel") {
            auto i1 = std::find(strand1.begin(), strand1.end(), r1);
            auto i2 = std::find(strand2.begin(), strand2.end(), r2);
            while (i1 != strand1.end() && i2 != strand2.end()) {
                // TODO: fill in
                ++i1;
                ++i2;
            }
        } else { // anti-parallel
            auto i1 = std::find(strand1.rbegin(), strand1.rend(), r1);
            auto i2 = std::find(strand2.begin(), strand2.end(), r2);
            logger::info(_logger, "pdbx_stuct_sheet_hbond: lengths ", std::distance(i1, strand1.rend()),
                         ' ', std::distance(i2, strand2.end()));
            while (i1 != strand1.rend() && i2 != strand2.end()) {
                Residue* r1 = *i1;
                Residue* r2 = *i2;
                Atom* a1 = r1->find_atom(atom_names[n1 % 2]);
                if (a1 == nullptr) {
                    logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find atom ",
                                    atom_names[n1 % 2], " in ", r1->str());
                }
                Atom* a2 = r2->find_atom(atom_names[n2 % 2]);
                if (a2 == nullptr) {
                    logger::warning(_logger, "pdbx_stuct_sheet_hbond: can't find atom ",
                                    atom_names[n2 % 2], " in ", r2->str());
                }
                if (a1 != nullptr && a2 != nullptr) {
                    for (auto& cs: mol->coord_sets()) {
                        hydro_pbg->new_pseudobond(a1, a2, cs);
                    }
                }
                ++n1;
                ++n2;
                a1 = r1->find_atom(atom_names[n1 % 2]);
                a2 = r2->find_atom(atom_names[n2 % 2]);
                if (a1 != nullptr && a2 != nullptr) {
                    for (auto& cs: mol->coord_sets()) {
                        hydro_pbg->new_pseudobond(a1, a2, cs);
                    }
                }

                // advance by two residues
                ++i1;
                ++i2;
                if (i1 == strand1.rend() || i2 == strand2.end())
                    break;
                ++i1;
                ++i2;
            }
        }
    }
}
#endif

void
ExtractMolecule::parse_entity()
{
    // keep track of non-polymer entities, so we don't try to connect them
    // type must be one of "branched", "macrolide", "non-polymer", "polymer", "water"
    // so only look at first letter
    CIFFile::ParseValues pv;
    pv.reserve(2);

    string entity_id;
    string description;
    char type;

    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                entity_id = string(start, end - start);
            });
        pv.emplace_back(get_column("pdbx_description"),
            [&] (const char* start, const char* end) {
                description = string(start, end - start);
                if (description.size() == 1 && (description[0] == '?' || description[0] == '/'))
                    description = "";
            });
        pv.emplace_back(get_column("type", Required),
            [&] (const char* start) {
                type = *start;
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping entity category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        if (type != 'p' && type != 'P')
            non_poly.emplace(entity_id);
        entity_description[entity_id] = description;
    }
}

void
ExtractMolecule::parse_entity_poly()
{
    CIFFile::ParseValues pv;
    pv.reserve(4);

    string entity_id;
    string type;
    bool nstd_monomer = false;

    try {
        pv.emplace_back(get_column("entity_id", Required),
            [&] (const char* start, const char* end) {
                entity_id = string(start, end - start);
            });
        pv.emplace_back(get_column("type"),
            [&] (const char* start, const char* end) {
                type = string(start, end - start);
            });
        pv.emplace_back(get_column("nstd_monomer"),
            [&] (const char* start) {
                nstd_monomer = *start == 'Y' || *start == 'y';
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping entity_poly category: ", e.what());
        return;
    }

    static const string peptide("polypeptide");
    static const string dna("polydeoxyribonucleotide");
    static const string rna("polyribonucleotide");

    while (parse_row(pv)) {
        // convert type to lowercase
        for (auto& c: type)
            c = tolower(c);
        PolymerType pt = PolymerType::PT_NONE;
        if (type.compare(0, peptide.size(), peptide) == 0)
            pt = PolymerType::PT_AMINO;
        else if (type.compare(0, dna.size(), dna) == 0
        || type.compare(0, rna.size(), rna) == 0)
            pt = PolymerType::PT_NUCLEIC;
        if (poly.find(entity_id) == poly.end())
            poly.emplace(entity_id, EntityPoly(nstd_monomer, pt));
        else
            logger::warning(_logger, "Duplicate polymer '", entity_id,
                            "' near line ", line_number());
    }
}

void
ExtractMolecule::parse_entity_poly_seq()
{
    // have to save all of entity_poly_seq because the same entity
    // can appear in more than one chain
    string entity_id;
    long seq_id = 0;
    ResName mon_id;
    bool hetero = false;

    CIFFile::ParseValues pv;
    pv.reserve(4);
    try {
        pv.emplace_back(get_column("entity_id", Required),
            [&] (const char* start, const char* end) {
                entity_id = string(start, end - start);
            });
        pv.emplace_back(get_column("num", Required),
            [&] (const char* start) {
                seq_id = readcif::str_to_int(start);
            });
        pv.emplace_back(get_column("mon_id", Required),
            [&] (const char* start, const char* end) {
                mon_id = ResName(start, end - start);
            });
        pv.emplace_back(get_column("hetero"),
            [&] (const char* start) {
                hetero = *start == 'Y' || *start == 'y';
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping entity_poly_seq category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        if (poly.find(entity_id) == poly.end()) {
            logger::warning(_logger, "Unknown polymer entity '", entity_id,
                            "' near line ", line_number());
            // fake polymer entity to cut down on secondary warnings
            poly.emplace(entity_id, false);
        }
        has_poly_seq[entity_id] = true;
        poly.at(entity_id).seq.emplace(seq_id, mon_id, hetero);
    }
}

static PyObject*
structure_pointers(ExtractMolecule &e)
{
    int count = 0;
    for (auto m: e.all_molecules) {
        if (m->atoms().size() > 0) {
            count += 1;
        }
    }

    void **sa;
    PyObject *s_array = python_voidp_array(count, &sa);
    int i = 0;
    for (auto m: e.all_molecules)
        if (m->atoms().size() > 0)
            sa[i++] = static_cast<void *>(m);

    return s_array;
}

PyObject*
parse_mmCIF_file(const char *filename, PyObject* logger, bool coordsets, bool atomic)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_file");
#endif
    ExtractMolecule extract(logger, StringVector(), coordsets, atomic);
    extract.parse_file(filename);
    return structure_pointers(extract);
}

PyObject*
parse_mmCIF_file(const char *filename, const StringVector& generic_categories,
                 PyObject* logger, bool coordsets, bool atomic)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_file2");
#endif
    ExtractMolecule extract(logger, generic_categories, coordsets, atomic);
    extract.parse_file(filename);
    return structure_pointers(extract);
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file, PyObject* logger, bool coordsets, bool atomic)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_buffer");
#endif
    ExtractMolecule extract(logger, StringVector(), coordsets, atomic);
    extract.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(extract);
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file,
   const StringVector& generic_categories, PyObject* logger, bool coordsets, bool atomic)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_buffer2");
#endif
    ExtractMolecule extract(logger, generic_categories, coordsets, atomic);
    extract.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(extract);
}


struct ExtractTables: public readcif::CIFFile
{
    struct Done: std::exception {};
    ExtractTables(const StringVector& categories, bool all_blocks=false);
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();
    void parse_category();

    PyObject* data;
    PyObject* all_data;
    string block_name;
    bool all_blocks;
};

ExtractTables::ExtractTables(const StringVector& categories, bool all_blocks):
    all_blocks(all_blocks)
{
    for (auto& c: categories) {
        register_category(c,
            [this] () {
                parse_category();
            });
    }
    reset_parse();
}

void
ExtractTables::reset_parse()
{
    data = nullptr;
    all_data = nullptr;
    block_name.clear();
}

void
ExtractTables::data_block(const string& name)
{
    if (data)
        finished_parse();
    if (!all_blocks) {
        // can only handle one data block with categories in it
        if (data)
            throw Done();
    }
    block_name = name;
}

void
ExtractTables::finished_parse()
{
    if (!all_blocks)
        return;
    if (!all_data)
        all_data = PyList_New(0);
    PyObject* results = PyTuple_New(2);
    PyObject* name = PyUnicode_DecodeUTF8(block_name.data(), block_name.size(), "replace");
    PyTuple_SET_ITEM(results, 0, name);
    if (!data)
        data = PyList_New(0);
    PyTuple_SET_ITEM(results, 1, data);
    data = nullptr;
    PyList_Append(all_data, results);
    Py_DECREF(results);
}

void
ExtractTables::parse_category()
{
    // this routine leaks memory for the PyStructSequence description
    const string& category = this->category();
    const StringVector& colnames = this->colnames();
    size_t num_colnames = colnames.size();

    PyObject* fields = PyTuple_New(num_colnames);
    if (!fields)
        throw std::runtime_error("Python Error");
    for (size_t i = 0; i < num_colnames; ++i) {
        PyObject* o = PyUnicode_DecodeUTF8(colnames[i].data(), colnames[i].size(), "replace");
        if (!o) {
            PyObject *type, *value, *traceback;
            PyErr_Fetch(&type, &value, &traceback);
            Py_DECREF(fields);
            PyErr_Restore(type, value, traceback);
            throw std::runtime_error("Python Error");
        }
        PyTuple_SET_ITEM(fields, i, o);
    }

    PyObject* items = PyList_New(0);
    parse_whole_category(
        [&] (const char* start, const char* end) {
            PyObject* o = PyUnicode_DecodeUTF8(start, end - start, "replace");
            if (!o || PyList_Append(items, o) < 0) {
                PyObject *type, *value, *traceback;
                PyErr_Fetch(&type, &value, &traceback);
                Py_XDECREF(o);
                Py_DECREF(fields);
                Py_DECREF(items);
                PyErr_Restore(type, value, traceback);
                throw std::runtime_error("Python Error");
            }
        });

    PyObject* field_items = PyTuple_New(2);
    if (!field_items) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        Py_DECREF(fields);
        Py_DECREF(items);
        PyErr_Restore(type, value, traceback);
        throw std::runtime_error("Python Error");
    }
    PyTuple_SET_ITEM(field_items, 0, fields);
    PyTuple_SET_ITEM(field_items, 1, items);

    PyObject* o = PyUnicode_DecodeUTF8(category.data(), category.size(), "replace");
    if (!o) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        Py_DECREF(field_items);
        PyErr_Restore(type, value, traceback);
        throw std::runtime_error("Python Error");
    }
    if (!data)
        data = PyDict_New();
    if (PyDict_SetItem(data, o, field_items) < 0) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        Py_DECREF(field_items);
        PyErr_Restore(type, value, traceback);
        throw std::runtime_error("Python Error");
    }
}

PyObject*
extract_CIF_tables(const char* filename,
                     const std::vector<std::string> &categories, bool all_data_blocks)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("extract_CIF tables");
#endif
    ExtractTables extract(categories, all_data_blocks);
    try {
        extract.parse_file(filename);
    } catch (ExtractTables::Done&) {
        // normal early termination
    }
    if (all_data_blocks) {
        if (extract.all_data == nullptr)
            Py_RETURN_NONE;
        return extract.all_data;
    } else {
        if (extract.data == nullptr)
            Py_RETURN_NONE;
        return extract.data;
    }
}

void
non_standard_bonds(const Bond **bonds, size_t num_bonds, bool selected_only, bool displayed_only, Bonds& disulfide, Bonds& covalent)
{
    const Bond** end_bonds = bonds + num_bonds;
    for (const Bond** bi = bonds; bi < end_bonds; ++bi) {
        const Bond *b = *bi;
        const auto atoms = b->atoms();
        const Atom* a0 = atoms[0];
        const Atom* a1 = atoms[1];
        const Residue* r0 = a0->residue();
        const Residue* r1 = a1->residue();
        if (r0 == r1)
            continue;  // intra-residue bonds should be in template
        if (selected_only && (!a0->selected() || !a1->selected()))
            continue;
        if (displayed_only && (!a0->display() || !a1->display()))
            continue;
        if (a0->element() == Element::S && a1->element() == Element::S) {
            disulfide.push_back(b);
            continue;
        }
        const Chain* c0 = r0->chain();
        const Chain* c1 = r1->chain();
        if (!c0 || c0 != c1) {
            // atoms in different chains
            covalent.push_back(b);
            continue;
        }
        if (!is_standard_residue(r0->name()) || !is_standard_residue(r1->name())) {
            // link to non-standard residue
            covalent.push_back(b);
            continue;
        }
        // check for non-implicit bond
        if (b->polymeric_start_atom() == nullptr) {
            // non-polymeric bond
            covalent.push_back(b);
            continue;
        }
        // check for non-adjacent bond
        const StructureSeq::ResMap& res_map = c0->res_map();
        StructureSeq::SeqPos p0, p1;
        try {
            p0 = res_map.at(const_cast<Residue*>(r0));
            p1 = res_map.at(const_cast<Residue*>(r1));
        } catch (std::out_of_range&) {
            // should never happen because residues are in same chain
            continue;
        }
        if (std::abs((ssize_t) (p1 - p0)) != 1) {
            // not adjacent (circular)
            covalent.push_back(b);
            continue;
        }
    }
}


bool
whitespace(Py_UCS4 ch)
{
    static Py_UCS4 whitespace[] = {
        // unicode Bidi-WS from https://en.wikipedia.org/wiki/Whitespace_character
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0,
        0x1680, 0x2000, 0x2001, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x205f, 0x3000
    };
    return std::end(whitespace) != std::find(std::begin(whitespace), std::end(whitespace), ch);
}

PyObject*
quote_value(PyObject* value, int max_len)
{
    // Return CIF 1.1 data value version of string
    // max_len is for mimicing the output from the PDB (see #2230)
    PyObject* str = PyObject_Str(value);
    if (!str)
        return NULL;

    if (PyBool_Check(value) || PyLong_Check(value) || PyFloat_Check(value))
        return str;

    Py_ssize_t len = PyUnicode_GET_LENGTH(str);
    if (len == 0) {
        Py_DECREF(str);
        return PyUnicode_FromString("''");
    }

    Py_UCS4 ch;
    int kind = PyUnicode_KIND(str);
    void* data = PyUnicode_DATA(str);
    ch = PyUnicode_READ(kind, data, 0);
    bool sing_quote = ch == '\'';
    bool dbl_quote = ch == '"';
    bool line_break = ch == '\n';
    bool special = ch == ' ' || ch == '_' || ch == '$' || ch == '[' || ch == ';';

    if (!(special || sing_quote || dbl_quote || line_break)) {
        // check for conflict with reserved words
        if (PyUnicode_READ(kind, data, len - 1) == '_') {
            // check if a reserved word: "data_", "loop_", "save_", "stop_", or "global_"
            if (len == 5 && (ch == 'd' || ch == 'D')) {
                // check if starts with "data_"
                ch = PyUnicode_READ(kind, data, 1);
                if (ch == 'a' || ch == 'A') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 't' || ch == 'T') {
                        ch = PyUnicode_READ(kind, data, 3);
                        special = (ch == 'a' || ch == 'A');
                    }
                }
            } else if (len == 5 && (ch == 'l' || ch == 'L')) {
                // check if "loop_"
                ch = PyUnicode_READ(kind, data, 1);
                if (ch == 'o' || ch == 'O') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 'o' || ch == 'O') {
                        ch = PyUnicode_READ(kind, data, 3);
                        special = (ch == 'p' || ch == 'P');
                    }
                }
            } else if (len == 5 && (ch == 's' || ch == 'S')) {
                // check if "stop_" or "save_"
                ch = PyUnicode_READ(kind, data, 1);
                if (ch == 't' || ch == 'T') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 'o' || ch == 'O') {
                        ch = PyUnicode_READ(kind, data, 3);
                        special = (ch == 'p' || ch == 'P');
                    }
                } else if (ch == 'a' || ch == 'A') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 'v' || ch == 'V') {
                        ch = PyUnicode_READ(kind, data, 3);
                        special = (ch == 'e' || ch == 'E');
                    }
                }
            } else if (len == 7 && (ch == 'g' || ch == 'G')) {
                // check if "global_"
                ch = PyUnicode_READ(kind, data, 1);
                if (ch == 'l' || ch == 'L') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 'o' || ch == 'O') {
                        ch = PyUnicode_READ(kind, data, 3);
                        if (ch == 'b' || ch == 'B') {
                            ch = PyUnicode_READ(kind, data, 4);
                            if (ch == 'a' || ch == 'A') {
                                ch = PyUnicode_READ(kind, data, 5);
                                special = (ch == 'l' || ch == 'L');
                            }
                        }
                    }
                }
            }
        } else if (len > 5 && PyUnicode_READ(kind, data, 4) == '_') {
            if (ch == 'd' || ch == 'D') {
                // check if starts with "data_"
                ch = PyUnicode_READ(kind, data, 1);
                if (ch == 'a' || ch == 'A') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 't' || ch == 'T') {
                        ch = PyUnicode_READ(kind, data, 3);
                        special = (ch == 'a' || ch == 'A');
                    }
                }
            } else if (ch == 's' || ch == 'S') {
                // check if starts with "save_"
                ch = PyUnicode_READ(kind, data, 1);
                if (ch == 'a' || ch == 'A') {
                    ch = PyUnicode_READ(kind, data, 2);
                    if (ch == 'v' || ch == 'V') {
                        ch = PyUnicode_READ(kind, data, 3);
                        special = (ch == 'e' || ch == 'E');
                    }
                }
            }
        }
    }

    for (auto i = 1; i < len; ++i) {
        ch = PyUnicode_READ(kind, data, i);
        if (i < len - 1) {
            if (ch == '"') {
                if (whitespace(PyUnicode_READ(kind, data, i + 1)))
                    dbl_quote = true;
                else
                    special = true;
                continue;
            } else if (ch == '\'') {
                if (whitespace(PyUnicode_READ(kind, data, i + 1)))
                    sing_quote = true;
                else
                    special = true;
                continue;
            }
        }
        if (whitespace(ch)) {
            if (ch == '\n')
                line_break = true;
            else
                special = true;
        }
    }
    PyObject* result;
    if (line_break || (sing_quote && dbl_quote) || (max_len && len > max_len))
        result = PyUnicode_FromFormat("\n;%U\n;\n", str);
    else if (dbl_quote)
        result = PyUnicode_FromFormat("'%U'", str);
    else if (sing_quote || special)
        result = PyUnicode_FromFormat("\"%U\"", str);
    else
        return str;
    Py_DECREF(str);
    return result;
}

} // namespace mmcif
