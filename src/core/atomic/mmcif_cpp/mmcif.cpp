// vi: set expandtab ts=4 sw=4:
#include "mmcif.h"
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/connect.h>
#include <atomstruct/tmpl/Atom.h>
#include <atomstruct/tmpl/Residue.h>
#include <logger/logger.h>
#include "pythonarray.h"	// Use python_voidp_array()
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>
#include <set>

using std::hash;
using std::map;
using std::multiset;
using std::set;
using std::string;
using std::unordered_map;
using std::vector;

using atomstruct::AtomicStructure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using atomstruct::Element;
using atomstruct::MolResId;
using basegeom::Coord;

namespace mmcif {

using atomstruct::AtomName;
using atomstruct::ChainID;
using atomstruct::ResName;

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

void connect_residue_by_template(Residue* r, const tmpl::Residue* tr);

struct ExtractMolecule: public readcif::CIFFile
{
    static const char* builtin_categories[];
    PyObject* _logger;
    ExtractMolecule(PyObject* logger, const StringVector& generic_categories);
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();
    void parse_audit_conform(bool in_loop);
    void parse_atom_site(bool in_loop);
    void parse_struct_conn(bool in_loop);
    void parse_struct_conf(bool in_loop);
    void parse_struct_sheet_range(bool in_loop);
    void parse_entity_poly_seq(bool in_loop);
    void parse_generic_category(bool in_loop);

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
    _logger(logger), first_model_num(INT_MAX)
{
    register_category("audit_conform",
        [this] (bool in_loop) {
            parse_audit_conform(in_loop);
        });
    register_category("atom_site",
        [this] (bool in_loop) {
            parse_atom_site(in_loop);
        });
    register_category("struct_conn",
        [this] (bool in_loop) {
            parse_struct_conn(in_loop);
        }, { "atom_site" });
    register_category("struct_conf",
        [this] (bool in_loop) {
            parse_struct_conf(in_loop);
        }, { "struct_conn" });
    register_category("struct_sheet_range",
        [this] (bool in_loop) {
            parse_struct_sheet_range(in_loop);
        }, { "struct_conn" });
    register_category("entity_poly_seq",
        [this] (bool in_loop) {
            parse_entity_poly_seq(in_loop);
        });
    for (auto& c: generic_categories) {
        if (std::find(std::begin(builtin_categories), std::end(builtin_categories), c) != std::end(builtin_categories)) {
            logger::warning(_logger, "Can not overriden builtin parsing for "
                            "category: ", c);
            continue;
        }
        register_category(c,
            [this] (bool in_loop) {
                parse_generic_category(in_loop);
            });
    }
}

void
ExtractMolecule::reset_parse()
{
    all_molecules.clear();
    molecules.clear();
    atom_map.clear();
    chain_entity_map.clear();
    all_residues.clear();
}

void
connect_residue_pairs(vector<Residue*> a, vector<Residue*> b, bool gap)
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
            if (gap) {
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
copy_nmr_info(AtomicStructure* from, AtomicStructure* to, PyObject* _logger)
{
    if (from->num_atoms() != to->num_atoms())
        logger::warning(_logger, "Mismatched number of atoms (",
            from->num_atoms(), " vs. ", to->num_atoms(), ")");
    // copy bonds, pseudobonds, secondary structure
    // -- Assumes atoms were added in the exact same order

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

    // Secondary Structure: TODO
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
                        p.seq_id, ") without hetero");
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
            if (lastp && lastp->seq_id == p.seq_id)
                continue;  // ignore microheterogeneity
            seqres.push_back(p.mon_id);
            lastp = &p;
        }
        mol->set_input_seq_info(auth_chain_id, seqres);
        if (mol->input_seq_source.empty())
            mol->input_seq_source = "mmCIF entity_poly_seq table";
    }
    find_and_add_metal_coordination_bonds(mol);
    // multiple molecules means there were multiple models,
    // so copy per-model information
    for (auto& im: molecules) {
        auto m = im.second;
        all_molecules.push_back(m);
        if (m != mol) {
            copy_nmr_info(mol, m, _logger);
        }
    }
    vector<AtomicStructure*> save_molecules;
    save_molecules.swap(all_molecules);
    reset_parse();
    save_molecules.swap(all_molecules);
}

void
ExtractMolecule::data_block(const string& /*name*/)
{
    if (!molecules.empty())
        finished_parse();
}

void
ExtractMolecule::parse_generic_category(bool /*in_loop*/)
{
    const string& category = this->category();
    const StringVector& tags = this->tags();
    generic_tables[category] = tags;

    StringVector data;
    CIFFile::ParseValues pv;
    pv.reserve(tags.size());
    for (auto& t: tags) {
        pv.emplace_back(get_column(t.c_str(), true), true,
            [&data] (const char* start, const char* end) {
                data.push_back(string(start, end - start));
            });
    }
    while (parse_row(pv))
        continue;
    generic_tables[category + " data"] = data;
}

void
ExtractMolecule::parse_audit_conform(bool /*in_loop*/)
{
    // Looking for a way to tell if the mmCIF file was written
    // in the PDBx/mmCIF stylized format.  The following technique
    // is not guaranteed to work, but we'll use it for now.
    string dict_name;
    float dict_version = 0;

    CIFFile::ParseValues pv;
    pv.reserve(2);
    pv.emplace_back(get_column("dict_name", true), true,
        [&dict_name] (const char* start, const char* end) {
            dict_name = string(start, end - start);
        });
    pv.emplace_back(get_column("dict_version"), false,
        [&dict_version] (const char* start, const char*) {
            dict_version = atof(start);
        });
    parse_row(pv);
    if (dict_name == "mmcif_pdbx.dic" && dict_version > 4)
        set_PDB_style(true);
}

void
ExtractMolecule::parse_atom_site(bool /*in_loop*/)
{
    // x, y, z are not required by mmCIF, but are by us

    readcif::CIFFile::ParseValues pv;
    pv.reserve(20);

    string entity_id;             // label_entity_id
    ChainID chain_id;              // label_asym_id
    ChainID auth_chain_id;         // auth_asym_id
    long position;                // label_seq_id
    long auth_position = INT_MAX; // auth_seq_id
    char ins_code = ' ';          // pdbx_PDB_ins_code
    char alt_id = '\0';           // label_alt_id
    AtomName atom_name;           // label_atom_id
    AtomName auth_atom_name;      // auth_atom_id
    ResName residue_name;          // label_comp_id
    ResName auth_residue_name;     // auth_comp_id
    char symbol[3];               // type_symbol
    long serial_num = 0;          // id
    float x, y, z;                // Cartn_[xyz]
    float occupancy = FLT_MAX;    // occupancy
    float b_factor = FLT_MAX;     // B_iso_or_equiv
    int model_num = 0;            // pdbx_PDB_model_num


    pv.emplace_back(get_column("id"), false,
        [&] (const char* start, const char*) {
            serial_num = readcif::str_to_int(start);
        });

    pv.emplace_back(get_column("label_entity_id"), true,
        [&] (const char* start, const char* end) {
            entity_id = string(start, end - start);
        });

    pv.emplace_back(get_column("label_asym_id", true), true,
        [&] (const char* start, const char* end) {
            chain_id = ChainID(start, end - start);
        });
    pv.emplace_back(get_column("auth_asym_id"), true,
        [&] (const char* start, const char* end) {
            auth_chain_id = ChainID(start, end - start);
            if (auth_chain_id == "." || auth_chain_id == "?")
                auth_chain_id.clear();
        });
    pv.emplace_back(get_column("pdbx_PDB_ins_code"), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code = ' ';
            else {
                // TODO: check if more than one character
                ins_code = *start;
            }
        });
    pv.emplace_back(get_column("label_seq_id", true), false,
        [&] (const char* start, const char*) {
            position = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("auth_seq_id"), false,
        [&] (const char* start, const char*) {
            if (*start == '.' || *start == '?')
                auth_position = INT_MAX;
            else
                auth_position = readcif::str_to_int(start);
        });

    pv.emplace_back(get_column("label_alt_id"), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1
            && (*start == '.' || *start == '?' || *start == ' '))
                alt_id = '\0';
            else {
                // TODO: what about more than one character?
                alt_id = *start;
            }
        });
    pv.emplace_back(get_column("type_symbol", true), false,
        [&] (const char* start, const char*) {
            symbol[0] = *start;
            symbol[1] = *(start + 1);
            if (readcif::is_whitespace(symbol[1]))
                symbol[1] = '\0';
            else
                symbol[2] = '\0';
        });
    pv.emplace_back(get_column("label_atom_id", true), true,
        [&] (const char* start, const char* end) {
            atom_name = AtomName(start, end - start);
        });
    pv.emplace_back(get_column("auth_atom_id"), true,
        [&] (const char* start, const char* end) {
            auth_atom_name = AtomName(start, end - start);
            if (auth_atom_name == "." || auth_atom_name == "?")
                auth_atom_name.clear();
        });
    pv.emplace_back(get_column("label_comp_id", true), true,
        [&] (const char* start, const char* end) {
            residue_name = ResName(start, end - start);
        });
    pv.emplace_back(get_column("auth_comp_id"), true,
        [&] (const char* start, const char* end) {
            auth_residue_name = ResName(start, end - start);
            if (auth_residue_name == "." || auth_residue_name == "?")
                auth_residue_name.clear();
        });
    // x, y, z are not required by mmCIF, but are by us
    pv.emplace_back(get_column("Cartn_x", true), false,
        [&] (const char* start, const char*) {
            x = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("Cartn_y", true), false,
        [&] (const char* start, const char*) {
            y = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("Cartn_z", true), false,
        [&] (const char* start, const char*) {
            z = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("occupancy"), false,
        [&] (const char* start, const char*) {
            if (*start == '?')
                occupancy = FLT_MAX;
            else
                occupancy = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("B_iso_or_equiv"), false,
        [&] (const char* start, const char*) {
            if (*start == '?')
                b_factor = FLT_MAX;
            else
                b_factor = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("pdbx_PDB_model_num"), false,
        [&] (const char* start, const char*) {
            model_num = readcif::str_to_int(start);
        });

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
    if (PDB_style())
        set_PDB_fixed_columns(true);
    while (parse_row(pv)) {
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
            chain_entity_map[chain_id] = entity_id;
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
            if (auth_position != INT_MAX)
                pos = auth_position;
            else
                pos = position;
            cur_residue = mol->new_residue(rname, cid, pos, ins_code);
            if (model_num == first_model_num) {
                all_residues[chain_id]
                    [ResidueKey(entity_id, position, residue_name)] = cur_residue;
            }
            cur_entity_id = entity_id;
            cur_seq_id = position;
            cur_auth_seq_id = auth_position;
            cur_chain_id = chain_id;
            cur_comp_id = residue_name;
        }

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
        if (b_factor != FLT_MAX)
            a->set_bfactor(b_factor);
        if (occupancy != FLT_MAX)
            a->set_occupancy(occupancy);

    }
}

void
ExtractMolecule::parse_struct_conn(bool /*in_loop*/)
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
    pv.emplace_back(get_column("conn_type_id", true), true,
        [&] (const char* start, const char* end) {
            conn_type = string(start, end - start);
        });

    pv.emplace_back(get_column(P1 ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id1 = ChainID(start, end - start);
        });
    pv.emplace_back(get_column("pdbx_" P1 INS_CODE), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code1 = ' ';
            else {
                // TODO: check if more than one character
                ins_code1 = *start;
            }
        });
    pv.emplace_back(get_column(P1 SEQ_ID, true), false,
        [&] (const char* start, const char*) {
            position1 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column(P1 AUTH_SEQ_ID), false,
        [&] (const char* start, const char*) {
            if (*start == '.' || *start == '?')
                auth_position1 = INT_MAX;
            else
                auth_position1 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("pdbx_" P1 ALT_ID), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1
            && (*start == '.' || *start == '?' || *start == ' '))
                alt_id1 = '\0';
            else {
                // TODO: what about more than one character?
                alt_id1 = *start;
            }
        });
    pv.emplace_back(get_column(P1 ATOM_ID, true), true,
        [&] (const char* start, const char* end) {
            atom_name1 = AtomName(start, end - start);
        });
    pv.emplace_back(get_column(P1 COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name1 = ResName(start, end - start);
        });
    pv.emplace_back(get_column(P1 SYMMETRY), true,
        [&] (const char* start, const char* end) {
            symmetry1 = string(start, end - start);
        });

    pv.emplace_back(get_column(P2 ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id2 = ChainID(start, end - start);
        });
    pv.emplace_back(get_column("pdbx_" P2 INS_CODE), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code2 = ' ';
            else {
                // TODO: check if more than one character
                ins_code2 = *start;
            }
        });
    pv.emplace_back(get_column(P2 SEQ_ID, true), false,
        [&] (const char* start, const char*) {
            position2 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column(P2 AUTH_SEQ_ID), false,
        [&] (const char* start, const char*) {
            if (*start == '.' || *start == '?')
                auth_position2 = INT_MAX;
            else
                auth_position2 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("pdbx_" P2 ALT_ID), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1
            && (*start == '.' || *start == '?' || *start == ' '))
                alt_id2 = '\0';
            else {
                // TODO: what about more than one character?
                alt_id2 = *start;
            }
        });
    pv.emplace_back(get_column(P2 ATOM_ID, true), true,
        [&] (const char* start, const char* end) {
            atom_name2 = AtomName(start, end - start);
        });
    pv.emplace_back(get_column(P2 COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name2 = ResName(start, end - start);
        });
    pv.emplace_back(get_column(P2 SYMMETRY), true,
        [&] (const char* start, const char* end) {
            symmetry2 = string(start, end - start);
        });
    pv.emplace_back(get_column("pdbx_dist_value"), false,
        [&] (const char* start, const char*) {
            distance = readcif::str_to_float(start);
        });

    atomstruct::Proxy_PBGroup* metal_pbg = nullptr;
    atomstruct::Proxy_PBGroup* hydro_pbg = nullptr;
    atomstruct::Proxy_PBGroup* missing_pbg = nullptr;
    // connect residues in molecule with all_residues information
    auto mol = all_residues.begin()->second.begin()->second->structure();
    while (parse_row(pv)) {
        if (symmetry1 != symmetry2)
            continue;
        bool normal = false;
        bool metal = false;
        bool hydro = false;
        // TODO: survey PDB mmCIF files and test in descending prevalence
        if (conn_type == "covale" || conn_type == "disulf")
            normal = true;
        else if (conn_type == "hydro")
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
        float idealBL = Element::bond_length(a1->element(), a2->element());
        float sqlength;
        if (distance > 0)
            sqlength = distance * distance;
        else {
            sqlength = a1->coord().sqdistance(a2->coord());
        }
        if (sqlength >= 3.0625f * idealBL * idealBL) {
            // 3.0625 == 1.75 squared
            // (allows ASP 223.A OD2 <-> PLP 409.A N1 bond in 1aam
            // and SER 233.A OG <-> NDP 300.A O1X bond in 1a80
            // to not be classified as missing seqments)
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
ExtractMolecule::parse_struct_conf(bool /*in_loop*/)
{
    if (molecules.empty())
        return;

    // these strings are concatenated to make the column headers needed
    #define BEG "beg"
    #define END "end"
    #define ASYM_ID "_label_asym_id"
    #define COMP_ID "_label_comp_id"
    #define SEQ_ID "_label_seq_id"
    #define INS_CODE "_PDB_ins_code" // pdbx
    string conf_type;                       // conf_type_id
    string id;                              // id
    ChainID chain_id1, chain_id2;            // (beg|end)_label_asym_id
    long position1, position2;              // (beg|end)_label_seq_id
    ResName residue_name1, residue_name2;    // (beg|end)_label_comp_id
    char ins_code1 = ' ', ins_code2 = ' ';  // pdbx_(beg|end)_PDB_ins_code

    CIFFile::ParseValues pv;
    pv.reserve(32);
    pv.emplace_back(get_column("id", true), true,
        [&] (const char* start, const char* end) {
            id = string(start, end - start);
        });
    pv.emplace_back(get_column("conf_type_id", true), true,
        [&] (const char* start, const char* end) {
            conf_type = string(start, end - start);
        });

    pv.emplace_back(get_column(BEG ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id1 = ChainID(start, end - start);
        });
    pv.emplace_back(get_column(BEG COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name1 = ResName(start, end - start);
        });
    pv.emplace_back(get_column(BEG SEQ_ID, true), false,
        [&] (const char* start, const char*) {
            position1 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("pdbx_" BEG INS_CODE), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code1 = ' ';
            else {
                // TODO: check if more than one character
                ins_code1 = *start;
            }
        });

    pv.emplace_back(get_column(END ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id2 = ChainID(start, end - start);
        });
    pv.emplace_back(get_column(END COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name2 = ResName(start, end - start);
        });
    pv.emplace_back(get_column(END SEQ_ID, true), false,
        [&] (const char* start, const char*) {
            position2 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("pdbx_" END INS_CODE), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code2 = ' ';
            else {
                // TODO: check if more than one character
                ins_code2 = *start;
            }
        });

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
                          "\" are in different chains on line ", line_number());
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
            logger::warning(_logger, "Invalid sheet range for secondardy"
                            " structure \"", id, "\": invalid chain"
                            ", on line ", line_number());
            continue;
        }
        const ResidueMap& residue_map = ari->second;
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid sheet range for secondary",
                            " structure \"", id, "\": invalid chain"
                            ", on line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly_seq.find(entity_id);
        if (psi == poly_seq.end()) {
            logger::warning(_logger, "Invalid sheet range for secondary",
                            " structure \"", id, "\": invalid chain"
                            ", on line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for secondary",
                            " structure \"", id, "\": ends before it starts"
                            ", on line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end()) {
            logger::warning(_logger,
                            "Bad residue range for secondary strcture \"", id,
                            "\" on line ", line_number());
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
                r->set_is_sheet(true);
                r->set_ss_id(strand_id);
            }
        }
    }
}

void
ExtractMolecule::parse_struct_sheet_range(bool /*in_loop*/)
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
    #define INS_CODE "_PDB_ins_code"        // pdbx
    string sheet_id;                        // sheet_id
    string id;                              // id
    ChainID chain_id1, chain_id2;            // (beg|end)_label_asym_id
    long position1, position2;              // (beg|end)_label_seq_id
    char ins_code1 = ' ', ins_code2 = ' ';  // pdbx_(beg|end)_PDB_ins_code
    ResName residue_name1, residue_name2;    // (beg|end)_label_comp_id

    CIFFile::ParseValues pv;
    pv.reserve(32);
    pv.emplace_back(get_column("sheet_id", true), true,
        [&] (const char* start, const char* end) {
            sheet_id = string(start, end - start);
        });
    pv.emplace_back(get_column("id", true), true,
        [&] (const char* start, const char* end) {
            id = string(start, end - start);
        });

    pv.emplace_back(get_column(BEG ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id1 = ChainID(start, end - start);
        });
    pv.emplace_back(get_column(BEG COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name1 = ResName(start, end - start);
        });
    pv.emplace_back(get_column(BEG SEQ_ID, true), false,
        [&] (const char* start, const char*) {
            position1 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("pdbx_" BEG INS_CODE), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code1 = ' ';
            else {
                // TODO: check if more than one character
                ins_code1 = *start;
            }
        });

    pv.emplace_back(get_column(END ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id2 = ChainID(start, end - start);
        });
    pv.emplace_back(get_column(END COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name2 = ResName(start, end - start);
        });
    pv.emplace_back(get_column(END SEQ_ID, true), false,
        [&] (const char* start, const char*) {
            position2 = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("pdbx_" END INS_CODE), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code2 = ' ';
            else {
                // TODO: check if more than one character
                ins_code2 = *start;
            }
        });

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
                            ", on line ", line_number());
            continue;
        }

        auto ari = all_residues.find(chain_id1);
        if (ari == all_residues.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": invalid chain"
                            ", on line ", line_number());
            continue;
        }
        const ResidueMap& residue_map = ari->second;
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": invalid chain"
                            ", on line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly_seq.find(entity_id);
        if (psi == poly_seq.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": invalid chain"
                            ", on line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\": ends before it starts"
                            ", on line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end()) {
            logger::warning(_logger, "Invalid sheet range for strand \"",
                            sheet_id, ' ', id, "\" on line ", line_number());
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
            r->set_is_sheet(true);
            r->set_ss_id(strand_id);
        }
    }
}

void
ExtractMolecule::parse_entity_poly_seq(bool /*in_loop*/)
{
    // have to save all of entity_poly_seq because the same entity
    // can appear in more than one chain
    string entity_id;
    long seq_id = 0;
    ResName mon_id;
    bool hetero = false;

    CIFFile::ParseValues pv;
    pv.reserve(4);
    pv.emplace_back(get_column("entity_id", true), true,
        [&] (const char* start, const char* end) {
            entity_id = string(start, end - start);
        });
    pv.emplace_back(get_column("num", true), false,
        [&] (const char* start, const char*) {
            seq_id = readcif::str_to_int(start);
        });
    pv.emplace_back(get_column("mon_id", true), true,
        [&] (const char* start, const char* end) {
            mon_id = ResName(start, end - start);
        });
    pv.emplace_back(get_column("hetero"), false,
        [&] (const char* start, const char*) {
            hetero = *start == 'Y' || *start == 'y';
        });

    while (parse_row(pv)) {
        poly_seq[entity_id].emplace(seq_id, mon_id, hetero);
    }
}

static PyObject*
structure_pointers(ExtractMolecule &e, const char *filename)
{
    int count = 0;
    for (auto m: e.all_molecules)
        if (m->atoms().size() > 0) {
	    m->set_name(filename);
	    count += 1;
        m->metadata = e.generic_tables;
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
clock_t start_t, end_t;
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
clock_t start_t, end_t;
#endif
    ExtractMolecule extract(logger, generic_categories);
    extract.parse_file(filename);
    return structure_pointers(extract, filename);
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file, PyObject* logger)
{
#ifdef CLOCK_PROFILING
clock_t start_t, end_t;
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
clock_t start_t, end_t;
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
        if (ta == NULL) {
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


} // namespace mmcif
