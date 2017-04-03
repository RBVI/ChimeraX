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
#include <atomstruct/connect.h>
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

using atomstruct::AtomName;
using atomstruct::ChainID;
using atomstruct::ResName;

namespace {

// Symbolic names for readcif arguments
static const bool Required = true;  // column is required

inline void
canonicalize_atom_name(AtomName* aname, bool* asterisks_translated)
{
    for (int i = aname->length(); i > 0; ) {
        --i;
        // use prime instead of asterisk
        if ((*aname)[i] == '*') {
            (*aname)[i] = '\'';
            *asterisks_translated = true;
        }
    }
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

void connect_residue_by_template(Residue* r, const tmpl::Residue* tr);

bool reasonable_bond_length(Atom* a1, Atom* a2, float distance = 0)
{
    float idealBL = Element::bond_length(a1->element(), a2->element());
    float sqlength;
    if (distance > 0)
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
    ExtractMolecule(PyObject* logger, const StringVector& generic_categories);
    ~ExtractMolecule();
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();
    void connect_residue_pairs(vector<Residue*> a, vector<Residue*> b, bool gap);
    const tmpl::Residue* find_template_residue(const ResName& name);
    void parse_audit_conform();
    void parse_atom_site();
    void parse_atom_site_anisotrop();
    void parse_struct_conn();
    void parse_struct_conf();
    void parse_struct_sheet_range();
    void parse_entity_poly_seq();
    void parse_entry();
    void parse_pdbx_database_PDB_obs_spr();
    void parse_generic_category();
    // for inline resiude templates
    void parse_chem_comp();
    void parse_chem_comp_bond();

    std::map<string, StringVector> generic_tables;
    vector<AtomicStructure*> all_molecules;
    map<int, AtomicStructure*> molecules;
    struct AtomKey {
        long position;
        long auth_position;   // needed in PDB mmCIF files for uniqueness
        AtomName atom_name;
        ResName residue_name;
        ChainID chain_id;
        char ins_code;
        char alt_id;
        AtomKey(const ChainID& c, long p, long ap, char i, char a,
                    const AtomName& n, const ResName& r):
            position(p), auth_position(ap), atom_name(n), residue_name(r),
            chain_id(c), ins_code(i), alt_id(a) {}
        bool operator==(const AtomKey& k) const {
            return position == k.position && auth_position == k.auth_position
                && atom_name == k.atom_name && residue_name == k.residue_name
                && chain_id == k.chain_id && ins_code == k.ins_code
                && alt_id == k.alt_id;
        }
        bool operator<(const AtomKey& k) const {
            if (position < k.position)
                return true;
            if (position != k.position)
                return false;
            if (auth_position < k.auth_position)
                return true;
            if (auth_position != k.auth_position)
                return false;
            if (atom_name < k.atom_name)
                return true;
            if (atom_name != k.atom_name)
                return false;
            if (residue_name < k.residue_name)
                return true;
            if (residue_name != k.residue_name)
                return false;
            if (chain_id < k.chain_id)
                return true;
            if (chain_id != k.chain_id)
                return false;
            if (alt_id < k.alt_id)
                return true;
            if (alt_id != k.alt_id)
                return false;
            return ins_code < k.ins_code;
        }
    };
    struct hash_AtomKey {
        size_t operator()(const AtomKey& k) const {
            return hash<ChainID>()(k.chain_id)
                ^ hash<long>()(k.position)
                ^ hash<long>()(k.auth_position)
                ^ hash<char>()(k.ins_code)
                ^ hash<char>()(k.alt_id)
                ^ hash<AtomName>()(k.atom_name)
                ^ hash<ResName>()(k.residue_name);
        }
    };
    unordered_map<AtomKey, Atom*, hash_AtomKey> atom_map;
    map<ChainID, string> chain_entity_map;
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
    unordered_map<ChainID, ResidueMap> all_residues;
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
    map<string /* entity_id */, EntityPolySeq> poly_seq;
    int first_model_num;
    string entry_id;
    tmpl::Molecule* my_templates;
    bool missing_poly_seq;
    bool has_pdbx;
};

const char* ExtractMolecule::builtin_categories[] = {
    "audit_conform", "atom_site", "entity_poly_seq"
    "struct_conn", "struct_conf", "struct_sheet_range",
};

std::ostream& operator<<(std::ostream& out, const ExtractMolecule::AtomKey& k) {
    out << k.chain_id << ':' << k.residue_name << '.' << k.position
        << '(' << k.auth_position << ')'
        << int(k.ins_code) << '@' << k.atom_name << '.' << int(k.alt_id);
    return out;
}

ExtractMolecule::ExtractMolecule(PyObject* logger, const StringVector& generic_categories):
    _logger(logger), first_model_num(INT_MAX), my_templates(nullptr),
    missing_poly_seq(false), has_pdbx(false)
{
    register_category("audit_conform",
        [this] () {
            parse_audit_conform();
        });
    register_category("entry",
        [this] () {
            parse_entry();
        });
    register_category("pdbx_database_PDB_obs_spr",
        [this] () {
            parse_pdbx_database_PDB_obs_spr();
        }, { "entry" });
    register_category("entity_poly_seq",
        [this] () {
            parse_entity_poly_seq();
        });
    register_category("atom_site",
        [this] () {
            parse_atom_site();
        }, { "entity_poly_seq" });
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
    for (auto& c: generic_categories) {
        if (std::find(std::begin(builtin_categories), std::end(builtin_categories), c) != std::end(builtin_categories)) {
            logger::warning(_logger, "Can not overriden builtin parsing for "
                            "category: ", c);
            continue;
        }
        register_category(c,
            [this] () {
                parse_generic_category();
            });
    }
    register_category("chem_comp",
        [this] () {
            parse_chem_comp();
        });
    register_category("chem_comp_bond",
        [this] () {
            parse_chem_comp_bond();
        }, { "chem_comp" });
}

ExtractMolecule::~ExtractMolecule()
{
    if (has_PDBx_fixed_width_columns())
        logger::info(_logger, "Used PDBx fixed column width tables to speed up reading mmCIF file");
    else
        logger::info(_logger, "No PDBx fixed column width tables");
    if (PDBx_keywords())
        logger::info(_logger, "Used PDBx keywords to speed up reading mmCIF file");
    else
        logger::info(_logger, "No PDBx keywords");
    if (my_templates)
        delete my_templates;
}

void
ExtractMolecule::reset_parse()
{
    molecules.clear();
    atom_map.clear();
    chain_entity_map.clear();
    all_residues.clear();
    entry_id.clear();
    generic_tables.clear();
    if (my_templates) {
        delete my_templates;
        my_templates = nullptr;
    }
    has_pdbx = false;
}

const tmpl::Residue*
ExtractMolecule::find_template_residue(const ResName& name)
{
    if (my_templates) {
        tmpl::Residue* tr = my_templates->find_residue(name);
        if (tr && tr->atoms_map().size() > 0)
            return tr;
    }
    return mmcif::find_template_residue(name);
}

void
ExtractMolecule::connect_residue_pairs(vector<Residue*> a, vector<Residue*> b, bool gap)
{
    // Connect adjacent residues that have the same type
    // and have link & chief atoms (i.e., peptides and nucleotides)
    for (auto&& r0: a) {
        auto tr0 = find_template_residue(r0->name());
        if (tr0 == nullptr)
            continue;
        auto ta0 = tr0->link();
        if (ta0 == nullptr)
            continue;
        for (auto&& r1: b) {
            auto tr1 = find_template_residue(r1->name());
            // only connect residues of the same type
            if (tr1 == nullptr || tr1->description() != tr0->description())
                continue;
            auto ta1 = tr1->chief();
            if (ta1 == nullptr)
                continue;
            Atom* a0 = r0->find_atom(ta0->name());
            Atom* a1 = r1->find_atom(ta1->name());
            if (a0 == nullptr && a1 != nullptr) {
                std::swap(a0, a1);
                std::swap(r0, r1);
            }
            if (a0 == nullptr) {
                find_nearest_pair(r0, r1, &a0, &a1);
            } else if (a1 == nullptr)
                a1 = find_closest(a0, r1, nullptr, true);
            if (a1 == nullptr)
                continue;
            if (gap && reasonable_bond_length(a0, a1))
                gap = false;    // bad data
            if (gap || !Bond::polymer_bond_atoms(a0, a1)) {
                auto as = r0->structure();
                auto pbg = as->pb_mgr().get_group(as->PBG_MISSING_STRUCTURE,
                    atomstruct::AS_PBManager::GRP_NORMAL);
                pbg->new_pseudobond(a0, a1);
            } else if (!a0->connects_to(a1))
                (void) a0->structure()->new_bond(a0, a1);
        }
    }
}

void
copy_nmr_info(Structure* from, Structure* to, PyObject* _logger)
{
    if (from->num_atoms() != to->num_atoms())
        logger::warning(_logger, "Mismatched number of atoms (",
            from->num_atoms(), " vs. ", to->num_atoms(), ")");
    // copy bonds, pseudobonds, secondary structure
    // -- Assumes atoms were added in the exact same order

    to->metadata = from->metadata;

    // Bonds:
    auto& bonds = from->bonds();
    auto& to_atoms = to->atoms();
    size_t to_size = to_atoms.size();
    for (auto&& b: bonds) {
        auto bond_atoms = b->atoms();
        auto a0_index = bond_atoms[0]->coord_index();
        auto a1_index = bond_atoms[1]->coord_index();
        if (a0_index >= to_size || a1_index >= to_size)
            continue;
        to->new_bond(to_atoms[a0_index], to_atoms[a1_index]);
    }

    // Pseudobonds:
    auto metal_pbg = from->pb_mgr().get_group(from->PBG_METAL_COORDINATION);
    if (metal_pbg != nullptr) {
        auto to_pbg = to->pb_mgr().get_group(to->PBG_METAL_COORDINATION,
            atomstruct::AS_PBManager::GRP_PER_CS);
        for (auto&& b: metal_pbg->pseudobonds()) {
            auto bond_atoms = b->atoms();
            auto a0_index = bond_atoms[0]->coord_index();
            auto a1_index = bond_atoms[1]->coord_index();
            if (a0_index >= to_size || a1_index >= to_size)
                continue;
            to_pbg->new_pseudobond(to_atoms[a0_index], to_atoms[a1_index]);
        }
    }
    auto hydro_pbg = from->pb_mgr().get_group(from->PBG_HYDROGEN_BONDS);
    if (hydro_pbg != nullptr) {
        auto to_pbg = to->pb_mgr().get_group(to->PBG_HYDROGEN_BONDS,
            atomstruct::AS_PBManager::GRP_PER_CS);
        for (auto&& b: hydro_pbg->pseudobonds()) {
            auto bond_atoms = b->atoms();
            auto a0_index = bond_atoms[0]->coord_index();
            auto a1_index = bond_atoms[1]->coord_index();
            if (a0_index >= to_size || a1_index >= to_size)
                continue;
            to_pbg->new_pseudobond(to_atoms[a0_index], to_atoms[a1_index]);
        }
    }

    // "seqres":
    auto& info = from->input_seq_info();
    for (auto& i: info)
        to->set_input_seq_info(i.first, i.second);

    // Secondary Structure:
    auto& residues = from->residues();
    auto& to_residues = to->residues();
    size_t num_residues = std::min(residues.size(), to_residues.size());
    for (size_t i = 0; i < num_residues; ++i) {
        to_residues[i]->set_is_strand(residues[i]->is_strand());
        to_residues[i]->set_is_helix(residues[i]->is_helix());
        to_residues[i]->set_ss_id(residues[i]->ss_id());
    }
}

void
ExtractMolecule::finished_parse()
{
    if (molecules.empty())
        return;

    // connect residues in molecule with all_residues information
    auto mol = all_residues.begin()->second.begin()->second->structure();
    for (auto&& r : mol->residues()) {
        auto tr = find_template_residue(r->name());
        if (tr == nullptr) {
            connect_residue_by_distance(r);
        } else {
            connect_residue_by_template(r, tr);
        }
    }

    // Connect residues in entity_poly_seq.
    // Because some positions are heterogeneous, delay connecting
    // until next group of residues is found.
    for (auto&& chain: all_residues) {
        const ResidueMap& residue_map = chain.second;
        auto ri = residue_map.begin();
        const string& entity_id = ri->first.entity_id;
        if (poly_seq.find(entity_id) == poly_seq.end())
            continue;
        const PolySeq* lastp = nullptr;
        bool gap = false;
        vector<Residue*> previous, current;
        ChainID auth_chain_id;
        auto& entity_poly_seq = poly_seq[entity_id];
        for (auto& p: entity_poly_seq) {
            auto ri = residue_map.find(ResidueKey(entity_id, p.seq_id, p.mon_id));
            if (ri == residue_map.end()) {
                if (current.empty())
                    continue;
                if (!previous.empty())
                    connect_residue_pairs(previous, current, gap);
                previous = std::move(current);
                current.clear();
                gap = true;
                continue;
            }
            Residue* r = ri->second;
            if (auth_chain_id.empty())
                auth_chain_id = r->chain_id();
            if (lastp && lastp->seq_id == p.seq_id) {
                if (!lastp->hetero)
                    logger::warning(_logger, "Duplicate entity_id/seq_id ",
                        p.seq_id, " without hetero");
                current.push_back(r);
            } else {
                if (!previous.empty() && !current.empty()) {
                    connect_residue_pairs(previous, current, gap);
                    gap = false;
                }
                if (!current.empty()) {
                    previous = std::move(current);
                    current.clear();
                }
                current.push_back(r);
            }
            lastp = &p;
        }
        if (!previous.empty())
            connect_residue_pairs(previous, current, gap);
        if (auth_chain_id.empty())
            continue;
        auto& input_seq_info = mol->input_seq_info();
        if (input_seq_info.find(auth_chain_id) != input_seq_info.end())
            continue;
        vector<ResName> seqres;
        seqres.reserve(entity_poly_seq.size());
        lastp = nullptr;
        for (auto& p: entity_poly_seq) {
            if (lastp && lastp->seq_id == p.seq_id && p.hetero)
                continue;  // ignore microheterogeneity
            seqres.push_back(p.mon_id);
            lastp = &p;
        }
        mol->set_input_seq_info(auth_chain_id, seqres);
        if (mol->input_seq_source.empty())
            mol->input_seq_source = "mmCIF entity_poly_seq table";
    }
    find_and_add_metal_coordination_bonds(mol);
    if (missing_poly_seq)
        find_missing_structure_bonds(mol);

    // export mapping of label chain ids to entity ids.
    StringVector chain_mapping;
    chain_mapping.reserve(chain_entity_map.size() * 2);
    for (auto i: chain_entity_map) {
        chain_mapping.emplace_back(i.first);
        chain_mapping.emplace_back(i.second);
    }
    generic_tables["chain_entity_map"] = chain_mapping;

    // multiple molecules means there were multiple models,
    // so copy per-model information
    for (auto& im: molecules) {
        auto m = im.second;
        all_molecules.push_back(m);
        m->metadata = generic_tables;
        if (m != mol) {
            copy_nmr_info(mol, m, _logger);
        }
        m->use_best_alt_locs();
    }
    vector<AtomicStructure*> save_molecules;
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
        logger::warning(_logger, "skipping entry category: ", e.what());
        return;
    }
    parse_row(pv);
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
        logger::warning(_logger, "skipping pdbx_database_PDB_obs_spr category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        if (id != "OBSLTE")
            continue;
        if (replace_pdb_id == entry_id)
            logger::warning(_logger, replace_pdb_id, " has been replaced by ",
                         pdb_id);
    }
}

void
ExtractMolecule::parse_generic_category()
{
    const string& category = this->category();
    const StringVector& colnames = this->colnames();
    generic_tables[category] = colnames;
    StringVector& data = parse_whole_category();
    generic_tables[category + " data"].swap(data);
}

void
ExtractMolecule::parse_chem_comp()
{
    ResName name;
    string  type;

    CIFFile::ParseValues pv;
    pv.reserve(4);
    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                name = ResName(start, end - start);
            });
        pv.emplace_back(get_column("type", Required),
            [&] (const char* start, const char* end) {
                type = string(start, end - start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "skipping chem_comp category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        // convert type to lowercase
        for (auto& c: type) {
            if (isupper(c))
                c = tolower(c);
        }
        if (my_templates == nullptr)
            my_templates = new tmpl::Molecule();

        tmpl::Residue* tr = my_templates->find_residue(name);
        if (!tr) {
            tr = my_templates->new_residue(name);
            bool is_peptide = type.find("peptide") != string::npos;
            if (is_peptide)
                tr->description("peptide");
            else {
                bool is_nucleotide = type.compare(0, 3, "dna") == 0
                    || type.compare(0, 3, "rna") == 0;
                if (is_nucleotide)
                    tr->description("nucleotide");
            }
        }
    }
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
        logger::warning(_logger, "skipping chem_comp_bond category: ", e.what());
        return;
    }
    // pretend all atoms are the same element, only need connectivity
    const Element& e = Element::get_element("H");
    while (parse_row(pv)) {
        tmpl::Residue* tr = my_templates->find_residue(rname);
        if (!tr)
            continue;
        tmpl::Atom* a1 = tr->find_atom(aname1);
        if (!a1) {
            a1 = my_templates->new_atom(aname1, e);
            tr->add_atom(a1);
        }
        tmpl::Atom* a2 = tr->find_atom(aname2);
        if (!a2) {
            a2 = my_templates->new_atom(aname2, e);
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
        if (tr->description() == "peptide") {
            tr->chief(tr->find_atom("N"));
            tr->link(tr->find_atom("C"));
        } else if (tr->description() == "nucleotide") {
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
                dict_version = atof(start);
            });
        pv.emplace_back(get_column("pdbx_keywords"),
            [&] (const char* start) {
                has_pdbx = true;
                set_PDBx_keywords(*start == 'Y' || *start == 'y');
            });
        pv.emplace_back(get_column("pdbx_fixed_width_columns"),
            [&] (const char* start, const char* end) {
                has_pdbx = true;
                for (const char *cp = start; cp < end; ++cp) {
                    if (isspace(*cp))
                        continue;
                    start = cp;
                    while (cp < end && !isspace(*cp))
                        ++cp;
                    set_PDBx_fixed_width_columns(string(start, cp - start));
                }
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "skipping audit_conform category: ", e.what());
        return;
    }
    parse_row(pv);
    if (!has_pdbx && dict_name == "mmcif_pdbx.dic" && dict_version > 4) {
        set_PDBx_keywords(true);
        set_PDBx_fixed_width_columns("atom_site");
        set_PDBx_fixed_width_columns("atom_site_anisotrop");
    }
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

    missing_poly_seq = poly_seq.empty();

    try {
        pv.emplace_back(get_column("id"),
            [&] (const char* start) {
                serial_num = readcif::str_to_int(start);
            });

        pv.emplace_back(get_column("label_entity_id"),
            [&] (const char* start, const char* end) {
                entity_id = string(start, end - start);
            });

        pv.emplace_back(get_column("label_asym_id", Required),
            [&] (const char* start, const char* end) {
                chain_id = ChainID(start, end - start);
            });
        pv.emplace_back(get_column("auth_asym_id"),
            [&] (const char* start, const char* end) {
                auth_chain_id = ChainID(start, end - start);
                if (auth_chain_id == "." || auth_chain_id == "?")
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
                if (auth_atom_name == "." || auth_atom_name == "?")
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
                if (auth_residue_name == "." || auth_residue_name == "?")
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
        logger::warning(_logger, "skipping atom_site category: ", e.what());
        return;
    }

    long atom_serial = 0;
    Residue* cur_residue = nullptr;
    AtomicStructure* mol = nullptr;
    int cur_model_num = INT_MAX;
    // residues are uniquely identified by (entity_id, seq_id, comp_id)
    string cur_entity_id;
    int cur_seq_id = INT_MAX;
    int cur_auth_seq_id = INT_MAX;
    ChainID cur_chain_id;
    ResName cur_comp_id;
    for (;;) {
        entity_id.clear();
        if (!parse_row(pv))
            break;
        if (model_num != cur_model_num) {
            if (first_model_num == INT_MAX)
                first_model_num = model_num;
            cur_model_num = model_num;
            mol = molecules[cur_model_num] = new AtomicStructure(_logger);
            cur_residue = nullptr;
        }

        if (cur_residue == nullptr
        || cur_entity_id != entity_id
        || cur_seq_id != position
        || cur_auth_seq_id != auth_position
        || cur_chain_id != chain_id
        || cur_comp_id != residue_name) {
            ResName rname;
            ChainID cid;
            long pos;
            if (!auth_residue_name.empty())
                rname = auth_residue_name;
            else
                rname = residue_name;
            if (!auth_chain_id.empty())
                cid = auth_chain_id;
            else
                cid = chain_id;
            if (!mol->lower_case_chains) {
                for (const char *cp = cid.c_str(); *cp != '\0'; ++cp) {
                    if (islower(*cp)) {
                        mol->lower_case_chains = true;
                        break;
                    }
                }
            }
            if (auth_position != INT_MAX)
                pos = auth_position;
            else
                pos = position;
            cur_residue = mol->new_residue(rname, cid, pos, ins_code);
            cur_residue->set_mmcif_chain_id(chain_id);
            cur_entity_id = entity_id;
            cur_seq_id = position;
            cur_auth_seq_id = auth_position;
            cur_chain_id = chain_id;
            cur_comp_id = residue_name;
            if (missing_poly_seq) {
                if (entity_id.empty())
                    entity_id = cid.c_str();
                // TODO: should only save amino and nucleic acids
                if (residue_name != "HOH")
                    poly_seq[entity_id].emplace(position, residue_name, false);
            }
            chain_entity_map[chain_id] = entity_id;
            if (model_num == first_model_num) {
                all_residues[chain_id]
                    [ResidueKey(entity_id, position, residue_name)] = cur_residue;
            }
        }

        if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
            logger::warning(_logger, "Skipping atom \"", atom_name,
                            "\" near line ", line_number(),
                            ": missing coordinates");
            continue;
        }
        canonicalize_atom_name(&atom_name, &mol->asterisks_translated);

        Atom* a;
        if (alt_id && cur_residue->count_atom(atom_name) == 1) {
            a = cur_residue->find_atom(atom_name);
            a->set_alt_loc(alt_id, true);
        } else {
            const Element& elem = Element::get_element(symbol);
            a = mol->new_atom(atom_name, elem);
            cur_residue->add_atom(a);
            if (alt_id)
                a->set_alt_loc(alt_id, true);
            if (model_num == first_model_num) {
                AtomKey k(chain_id, position, auth_position, ins_code, alt_id,
                        atom_name, residue_name);
                atom_map[k] = a;
            }
        }
        Coord c(x, y, z);
        a->set_coord(c);
        if (serial_num) {
            atom_serial = serial_num;
            a->set_serial_number(atom_serial);
        } else {
            a->set_serial_number(++atom_serial);
        }
        if (b_factor != DBL_MAX)
            a->set_bfactor(b_factor);
        if (occupancy != DBL_MAX)
            a->set_occupancy(occupancy);

    }
}

void
ExtractMolecule::parse_atom_site_anisotrop()
{
    readcif::CIFFile::ParseValues pv;
    pv.reserve(20);

    long serial_num = 0;          // id
    float u11, u12, u13, u22, u23, u33;

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
        logger::warning(_logger, "skipping atom_site_anistrop category: ", e.what());
        return;
    }

    auto mol = all_residues.begin()->second.begin()->second->structure();
    auto& atoms = mol->atoms();
    std::map <long, Atom*> atom_lookup;
    for (auto&& a: atoms) {
        atom_lookup[a->serial_number()] = a;
    }
    while (parse_row(pv)) {
        const auto& ai = atom_lookup.find(serial_num);
        if (ai == atom_lookup.end())
            continue;
        Atom *a = ai->second;
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
    #define INS_CODE "_PDB_ins_code" // pdbx
    #define SYMMETRY "_symmetry"

    // bonds from struct_conn records
    ChainID chain_id1, chain_id2;            // ptrn[12]_label_asym_id
    long position1, position2;              // ptrn[12]_label_seq_id
    long auth_position1 = INT_MAX,
         auth_position2 = INT_MAX;          // ptrn[12]_auth_seq_id
    char ins_code1 = ' ', ins_code2 = ' ';  // pdbx_ptrn[12]_PDB_ins_code
    char alt_id1 = '\0', alt_id2 = '\0';    // pdbx_ptrn[12]_label_alt_id
    AtomName atom_name1, atom_name2;        // ptrn[12]_label_atom_id
    ResName residue_name1, residue_name2;    // ptrn[12]_label_comp_id
    string conn_type;                       // conn_type_id
    string symmetry1, symmetry2;            // ptrn[12]_symmetry
    float distance = 0;                     // pdbx_dist_value

    CIFFile::ParseValues pv;
    pv.reserve(32);
    try {
        pv.emplace_back(get_column("conn_type_id", Required),
            [&] (const char* start, const char* end) {
                conn_type = string(start, end - start);
            });

        pv.emplace_back(get_column(P1 ASYM_ID, Required),
            [&] (const char* start, const char* end) {
                chain_id1 = ChainID(start, end - start);
            });
        pv.emplace_back(get_column("pdbx_" P1 INS_CODE),
            [&] (const char* start, const char* end) {
                if (end == start + 1 && (*start == '.' || *start == '?'))
                    ins_code1 = ' ';
                else {
                    // TODO: check if more than one character
                    ins_code1 = *start;
                }
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
        pv.emplace_back(get_column("pdbx_" P2 INS_CODE),
            [&] (const char* start, const char* end) {
                if (end == start + 1 && (*start == '.' || *start == '?'))
                    ins_code2 = ' ';
                else {
                    // TODO: check if more than one character
                    ins_code2 = *start;
                }
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
        logger::warning(_logger, "skipping struct_conn category: ", e.what());
        return;
    }

    atomstruct::Proxy_PBGroup* metal_pbg = nullptr;
    atomstruct::Proxy_PBGroup* hydro_pbg = nullptr;
    atomstruct::Proxy_PBGroup* missing_pbg = nullptr;
    // connect residues in molecule with all_residues information
    auto mol = all_residues.begin()->second.begin()->second->structure();
    while (parse_row(pv)) {
        if (symmetry1 != symmetry2)
            continue;
        if (atom_name1 == '?' || atom_name2 == '?')
            continue;
        bool normal = false;
        bool metal = false;
        bool hydro = false;
        // TODO: survey PDB mmCIF files and test in descending prevalence
        if (conn_type == "covale" || conn_type == "disulf")
            normal = true;
        else if (conn_type == "hydrog")
            hydro = true;
        else if (conn_type == "metalc")
            metal = true;
        if (!normal && !metal && !hydro)
            continue;   // skip hydrogen, modres, and unknown bonds
        AtomKey k1(chain_id1, position1, auth_position1, ins_code1, alt_id1,
                atom_name1, residue_name1);
        auto ai1 = atom_map.find(k1);
        if (ai1 == atom_map.end())
            continue;
        AtomKey k2(chain_id2, position2, auth_position2, ins_code2, alt_id2,
                atom_name2, residue_name2);
        auto ai2 = atom_map.find(k2);
        if (ai2 == atom_map.end())
            continue;
        Atom* a1 = ai1->second;
        Atom* a2 = ai2->second;
        if (metal) {
            if (metal_pbg == nullptr)
                metal_pbg = mol->pb_mgr().get_group(mol->PBG_METAL_COORDINATION,
                    atomstruct::AS_PBManager::GRP_PER_CS);
            metal_pbg->new_pseudobond(a1, a2);
            continue;
        }
        if (hydro) {
            if (hydro_pbg == nullptr)
                hydro_pbg = mol->pb_mgr().get_group(mol->PBG_HYDROGEN_BONDS,
                    atomstruct::AS_PBManager::GRP_PER_CS);
            hydro_pbg->new_pseudobond(a1, a2);
            continue;
        }
        if (!reasonable_bond_length(a1, a2, distance)) {
            if (missing_pbg == nullptr)
                missing_pbg = mol->pb_mgr().get_group(
                    mol->PBG_MISSING_STRUCTURE,
                    atomstruct::AS_PBManager::GRP_NORMAL);
            missing_pbg->new_pseudobond(a1, a2);
            continue;
        }
        try {
            mol->new_bond(a1, a2);
        } catch (std::invalid_argument& e) {
            // already bonded
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
        logger::warning(_logger, "skipping struct_conf category: ", e.what());
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
            logger::warning(_logger, "Start and end residues of secondary"
                          " structure \"", id,
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

        auto ari = all_residues.find(chain_id1);
        if (ari == all_residues.end()) {
            logger::warning(_logger, "Invalid residue range for secondardy"
                            " structure \"", id, "\": invalid chain \"",
                            chain_id1, "\", near line ", line_number());
            continue;
        }
        const ResidueMap& residue_map = ari->second;
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid residue range for secondary",
                            " structure \"", id, "\": invalid chain \"",
                            chain_id1, "\", near line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly_seq.find(entity_id);
        if (psi == poly_seq.end()) {
            logger::warning(_logger, "Invalid residue range for secondary",
                            " structure \"", id, "\": invalid entity \"",
                            entity_id, "\", near line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for secondary",
                            " structure \"", id, "\": ends before it starts"
                            ", near line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end()) {
            logger::warning(_logger,
                            "Bad residue range for secondary strcture \"", id,
                            "\" near line ", line_number());
            continue;
        }
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
    ChainID chain_id1, chain_id2;            // (beg|end)_label_asym_id
    long position1, position2;              // (beg|end)_label_seq_id
    ResName residue_name1, residue_name2;    // (beg|end)_label_comp_id

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
        logger::warning(_logger, "skipping struct_sheet_range category: ", e.what());
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
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": different chains"
                            ", near line ", line_number());
            continue;
        }

        auto ari = all_residues.find(chain_id1);
        if (ari == all_residues.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": invalid chain \"",
                            chain_id1, "\", near line ", line_number());
            continue;
        }
        const ResidueMap& residue_map = ari->second;
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": invalid chain \"",
                            chain_id1, "\", near line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly_seq.find(entity_id);
        if (psi == poly_seq.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": invalid entity \"",
                            entity_id, "\", near line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": ends before it starts"
                            ", near line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\" near line ", line_number());
            continue;
        }
        int strand_id;
        auto si = strand_ids.find(chain_id1);
        if (si == strand_ids.end()) {
            strand_ids[chain_id1] = 1;
            strand_id = 1;
        } else {
            strand_id = ++(si->second);
        }
        for (auto pi = init_ps; pi != end_ps; ++pi) {
            auto ri = residue_map.find(ResidueKey(entity_id, pi->seq_id,
                                                  pi->mon_id));
            if (ri == residue_map.end())
                continue;
            Residue *r = ri->second;
            r->set_is_strand(true);
            r->set_ss_id(strand_id);
        }
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
        logger::warning(_logger, "skipping entity_poly_seq category: ", e.what());
        return;
    }

    while (parse_row(pv)) {
        poly_seq[entity_id].emplace(seq_id, mon_id, hetero);
    }
}

static PyObject*
structure_pointers(ExtractMolecule &e, const char *filename)
{
    int count = 0;
    for (auto m: e.all_molecules) {
        if (m->atoms().size() > 0) {
            m->set_name(filename);
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
parse_mmCIF_file(const char *filename, PyObject* logger)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_file");
#endif
    ExtractMolecule extract(logger, StringVector());
    extract.parse_file(filename);
    return structure_pointers(extract, filename);
}

PyObject*
parse_mmCIF_file(const char *filename, const StringVector& generic_categories,
                 PyObject* logger)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_file2");
#endif
    ExtractMolecule extract(logger, generic_categories);
    extract.parse_file(filename);
    return structure_pointers(extract, filename);
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file, PyObject* logger)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_buffer");
#endif
    ExtractMolecule extract(logger, StringVector());
    extract.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(extract, "unknown mmCIF file");
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file,
   const StringVector& generic_categories, PyObject* logger)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_buffer2");
#endif
    ExtractMolecule extract(logger, generic_categories);
    extract.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(extract, "unknown mmCIF file");
}

// connect_residue_by_template:
//    Connect bonds in residue according to the given template.  Takes into
//    account alternate atom locations.
void
connect_residue_by_template(Residue* r, const tmpl::Residue* tr)
{
    auto& atoms = r->atoms();

    // Confirm all atoms in residue are in template, if not connect by distance
    for (auto&& a: atoms) {
        tmpl::Atom *ta = tr->find_atom(a->name());
        if (!ta) {
            connect_residue_by_distance(r);
            return;
        }
    }

    // foreach atom in residue
    //    connect up like atom in template
    for (auto&& a: atoms) {
        tmpl::Atom *ta = tr->find_atom(a->name());
        for (auto&& tmpl_nb: ta->neighbors()) {
            Atom *b = r->find_atom(tmpl_nb->name());
            if (b == nullptr)
                continue;
            if (!a->connects_to(b))
                (void) a->structure()->new_bond(a, b);
        }
    }
}


struct ExtractTables: public readcif::CIFFile
{
    struct Done: std::exception {};
    ExtractTables(const StringVector& categories);
    virtual void data_block(const string& name);
    void parse_category();

    PyObject* data;
};

ExtractTables::ExtractTables(const StringVector& categories):
    data(nullptr)
{
    for (auto& c: categories) {
        register_category(c,
            [this] () {
                parse_category();
            });
    }
}

void
ExtractTables::data_block(const string& /*name*/)
{
    // can only handle one data block with categories in it
    if (data)
        throw Done();
}

void
ExtractTables::parse_category()
{
    // this routine leaks memory for the PyStructSequence description
    if (!data)
        data = PyDict_New();
    const string& category = this->category();
    const StringVector& colnames = this->colnames();
    size_t num_colnames = colnames.size();

    PyObject* fields = PyTuple_New(num_colnames);
    if (!fields)
        throw wrappy::PythonError();
    for (size_t i = 0; i < num_colnames; ++i) {
        PyObject* o = wrappy::pyObject(colnames[i]);
        if (!o) {
            Py_DECREF(fields);
            throw wrappy::PythonError();
        }
        PyTuple_SET_ITEM(fields, i, o);
    }

    PyObject* items = PyList_New(0);
    parse_whole_category(
        [&] (const char* start, const char* end) {
            PyObject* o = PyUnicode_DecodeUTF8(start, end - start, "replace");
            if (!o || PyList_Append(items, o) < 0) {
                Py_XDECREF(o);
                Py_DECREF(fields);
                Py_DECREF(items);
                throw wrappy::PythonError();
            }
        });

    PyObject* field_items = PyTuple_New(2);
    if (!field_items) {
        Py_DECREF(fields);
        Py_DECREF(items);
        throw wrappy::PythonError();
    }
    PyTuple_SET_ITEM(field_items, 0, fields);
    PyTuple_SET_ITEM(field_items, 1, items);

    PyObject* o = wrappy::pyObject(category);
    if (!o) {
        Py_DECREF(field_items);
        throw wrappy::PythonError();
    }
    if (PyDict_SetItem(data, o, field_items) < 0) {
        Py_DECREF(field_items);
        throw wrappy::PythonError();
    }
}


PyObject*
extract_mmCIF_tables(const char* filename,
                     const std::vector<std::string> &categories)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("extract_mmCIF tables");
#endif
    ExtractTables extract(categories);
    try {
        extract.parse_file(filename);
    } catch (ExtractTables::Done&) {
        // normal early termination
    }
    if (extract.data == nullptr)
        Py_RETURN_NONE;
    return extract.data;
}

} // namespace mmcif
