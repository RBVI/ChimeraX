// vi: set expandtab ts=4 sw=4:
#include "mmcif.h"
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <blob/StructBlob.h>
#include <atomstruct/connect.h>
#include <atomstruct/tmpl/Atom.h>
#include <atomstruct/tmpl/Residue.h>
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>

using std::hash;
using std::map;
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
    ExtractMolecule();
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();
    void parse_audit_conform(bool in_loop);
    void parse_atom_site(bool in_loop);
    void parse_struct_conn(bool in_loop);
    void parse_entity_poly_seq(bool in_loop);

    vector<AtomicStructure*> all_molecules;
    map<int, AtomicStructure*> molecules;
    struct AtomKey {
        string chain_id;
        long position;
        char ins_code;
        char alt_id;
        string atom_name;
        string residue_name;
        AtomKey(const string& c, long p, char i, char a, const string& n, const string& r):
            chain_id(c), position(p), ins_code(i), alt_id(a), atom_name(n), residue_name(r) {}
        bool operator==(const AtomKey& k) const {
            return position == k.position && ins_code == k.ins_code
                && alt_id == k.alt_id && atom_name == k.atom_name
                && residue_name == k.residue_name && chain_id == k.chain_id;
        }
    };
    struct hash_AtomKey {
        size_t operator()(const AtomKey& k) const {
            return hash<string>()(k.chain_id) ^ hash<long>()(k.position)
                ^ hash<char>()(k.ins_code) ^ hash<char>()(k.alt_id)
                ^ hash<string>()(k.atom_name) ^ hash<string>()(k.residue_name);
        }
    };
    unordered_map<AtomKey, Atom*, hash_AtomKey> atom_map;
    struct ResidueKey {
        string entity_id;
        long seq_id;
        string mon_id;
        ResidueKey(const string& e, long n, const string& m):
            entity_id(e), seq_id(n), mon_id(m) {}
        bool operator==(const ResidueKey& k) const {
            return seq_id == k.seq_id && entity_id == k.entity_id
                && mon_id == k.mon_id;
        }
    };
    struct hash_ResidueKey {
        size_t operator()(const ResidueKey& k) const {
            return hash<string>()(k.entity_id) ^ hash<long>()(k.seq_id)
                ^ hash<string>()(k.mon_id);
        }
    };
    unordered_map<ResidueKey, Residue*, hash_ResidueKey> residue_map;
};

ExtractMolecule::ExtractMolecule()
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
    register_category("entity_poly_seq",
        [this] (bool in_loop) {
            parse_entity_poly_seq(in_loop);
        }, { "atom_site", "struct_conn" });
}

void
ExtractMolecule::reset_parse()
{
    all_molecules.clear();
    molecules.clear();
    atom_map.clear();
    residue_map.clear();
}

void
ExtractMolecule::finished_parse()
{
    if (molecules.empty())
        return;
    // connect residues in first molecule,
    auto mol = molecules.begin()->second;
    for (auto&& r : mol->residues()) {
        auto tr = find_template_residue(r->name());
        if (tr == nullptr) {
            connect_residue_by_distance(r.get());
        } else {
            connect_residue_by_template(r.get(), tr);
        }
    }
    // TODO: next copy connectivity to other molecules
    for (auto& im: molecules) {
        all_molecules.push_back(im.second);
    }
    molecules.clear();
    atom_map.clear();
    residue_map.clear();
}

void
ExtractMolecule::data_block(const string& /*name*/)
{
    if (!molecules.empty())
        finished_parse();
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
    string chain_id;              // label_asym_id
    string auth_chain_id;         // auth_asym_id
    long position;                // label_seq_id
    long auth_position = INT_MAX; // auth_seq_id
    char ins_code = ' ';          // pdbx_PDB_ins_code
    char alt_id = '\0';           // label_alt_id
    string atom_name;             // label_atom_id
    string auth_atom_name;        // auth_atom_id
    string residue_name;          // label_comp_id
    string auth_residue_name;     // auth_comp_id
    char symbol[3];               // type_symbol
    long serial_num = 0;          // id
    float x, y, z;                // Cartn_[xyz]
    float occupancy = FLT_MAX;    // occupancy
    float b_factor = FLT_MAX;     // B_iso_or_equiv
    int model_num = 1;            // pdbx_PDB_model_num


    pv.emplace_back(get_column("id", false), false,
        [&] (const char* start, const char*) {
            serial_num = readcif::str_to_int(start);
        });

    pv.emplace_back(get_column("label_entity_id", false), true,
        [&] (const char* start, const char* end) {
            entity_id = string(start, end - start);
        });

    pv.emplace_back(get_column("label_asym_id", true), true,
        [&] (const char* start, const char* end) {
            chain_id = string(start, end - start);
        });
    pv.emplace_back(get_column("auth_asym_id"), true,
        [&] (const char* start, const char* end) {
            auth_chain_id = string(start, end - start);
            if (auth_chain_id == "." || auth_chain_id == "?")
                auth_chain_id.clear();
        });
    pv.emplace_back(get_column("pdbx_PDB_ins_code", false), true,
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

    pv.emplace_back(get_column("label_alt_id", false), true,
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
            atom_name = string(start, end - start);
        });
    pv.emplace_back(get_column("auth_atom_id"), true,
        [&] (const char* start, const char* end) {
            auth_atom_name = string(start, end - start);
            if (auth_atom_name == "." || auth_atom_name == "?")
                auth_atom_name.clear();
        });
    pv.emplace_back(get_column("label_comp_id", true), true,
        [&] (const char* start, const char* end) {
            residue_name = string(start, end - start);
        });
    pv.emplace_back(get_column("auth_comp_id"), true,
        [&] (const char* start, const char* end) {
            auth_residue_name = string(start, end - start);
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
            if (*start == '.' || *start == '?')
                model_num = 1;
            else
                model_num = readcif::str_to_int(start);
        });

    long atom_serial = 0;
    Residue* cur_residue = nullptr;
    AtomicStructure* mol = nullptr;
    int cur_model_num = INT_MAX;
    // residues are uniquely identified by (entity_id, seq_id, comp_id)
    string cur_entity_id;
    int cur_seq_id = INT_MAX;
    string cur_comp_id;
    if (PDB_style())
        set_PDB_fixed_columns(true);
    while (parse_row(pv)) {
        if (position == 0) {
            // HETATM residues (waters) might be missing a sequence number
            if (cur_residue == nullptr || cur_residue->chain_id() != chain_id)
                position = 1;
            else
                ++position;
        }

        if (model_num != cur_model_num) {
            cur_model_num = model_num;
            mol = molecules[cur_model_num] = new AtomicStructure;
            cur_residue = nullptr;
        }

        if (cur_residue == nullptr
        || cur_entity_id != entity_id
        || cur_seq_id != position
        || cur_comp_id != residue_name) {
            cur_residue = mol->new_residue(residue_name, chain_id,
                                                        position, ins_code);
            residue_map[ResidueKey(entity_id, position, residue_name)]
                = cur_residue;
            cur_entity_id = entity_id;
            cur_seq_id = position;
            cur_comp_id = residue_name;
        }

        Atom* a;
        if (alt_id && cur_residue->count_atom(atom_name) == 1) {
            a = cur_residue->find_atom(atom_name);
            a->set_alt_loc(alt_id, true);
        } else {
            Element elem(symbol);
            a = mol->new_atom(atom_name, elem);
            cur_residue->add_atom(a);
            if (alt_id)
                a->set_alt_loc(alt_id, true);
            if (position != 0 && model_num == 1) {
                AtomKey k(chain_id, position, ins_code, alt_id, atom_name,
                                                            residue_name);
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
    #define ATOM_ID "_label_atom_id"
    #define ALT_ID "_label_alt_id" // pdbx
    #define INS_CODE "_PDB_ins_code" // pdbx

    // bonds from struct_conn records
    string chain_id1, chain_id2;            // ptrn[12]_label_asym_id
    long position1, position2;              // ptrn[12]_label_seq_id
    char ins_code1 = ' ', ins_code2 = ' ';  // pdbx_ptrn[12]_PDB_ins_code
    char alt_id1 = '\0', alt_id2 = '\0';    // pdbx_ptrn[12]_label_alt_id
    string atom_name1, atom_name2;          // ptrn[12]_label_atom_id
    string residue_name1, residue_name2;    // ptrn[12]_label_comp_id
    string conn_type;                       // conn_type_id

    CIFFile::ParseValues pv;
    pv.reserve(32);
    pv.emplace_back(get_column("conn_type_id", true), true,
        [&] (const char* start, const char* end) {
            conn_type = string(start, end - start);
        });

    pv.emplace_back(get_column(P1 ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id1 = string(start, end - start);
        });
    pv.emplace_back(get_column("pdbx_" P1 INS_CODE, false), true,
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
    pv.emplace_back(get_column("pdbx_" P1 ALT_ID, false), true,
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
            atom_name1 = string(start, end - start);
        });
    pv.emplace_back(get_column(P1 COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name1 = string(start, end - start);
        });

    pv.emplace_back(get_column(P2 ASYM_ID, true), true,
        [&] (const char* start, const char* end) {
            chain_id2 = string(start, end - start);
        });
    pv.emplace_back(get_column("pdbx_" P2 INS_CODE, false), true,
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
    pv.emplace_back(get_column("pdbx_" P2 ALT_ID, false), true,
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
            atom_name2 = string(start, end - start);
        });
    pv.emplace_back(get_column(P2 COMP_ID, true), true,
        [&] (const char* start, const char* end) {
            residue_name2 = string(start, end - start);
        });

    auto mol = molecules.begin()->second;
    while (parse_row(pv)) {
        if (conn_type != "covale" && conn_type != "disulf")
            continue;   // skip hydrogen and modres bonds
        AtomKey k1(chain_id1, position1, ins_code1, alt_id1, atom_name1,
                                                        residue_name1);
        auto ai1 = atom_map.find(k1);
        if (ai1 == atom_map.end())
            continue;
        AtomKey k2(chain_id2, position2, ins_code2, alt_id2, atom_name2,
                                                        residue_name2);
        auto ai2 = atom_map.find(k2);
        if (ai2 == atom_map.end())
            continue;
        try {
            mol->new_bond(ai1->second, ai2->second);
        } catch (std::invalid_argument& e) {
            // already bonded
        }
    }
    #undef P1
    #undef P2
    #undef ASYM_ID
    #undef COMP_ID
    #undef SEQ_ID
    #undef ATOM_ID
    #undef ALT_ID
    #undef INS_CODE
}

void
connect_residue_pairs(vector<Residue*> a, vector<Residue*> b, bool /*gap*/)
{
    // TODO: if gap, chain is discontinious
    for (auto&& r0: a) {
        auto tr0 = find_template_residue(r0->name());
        if (tr0 == nullptr)
            continue;
        Atom *a0 = r0->find_atom(tr0->link()->name());
        if (a0 == nullptr)
            continue;
        for (auto&& r1: b) {
            auto tr1 = find_template_residue(r1->name());
            // only connect residues of the same type
            if (tr1 == nullptr || tr1->description() != tr0->description())
                continue;
            Atom *a1 = r1->find_atom(tr1->chief()->name());
            if (a1 == nullptr)
                continue;
            if (!a0->connects_to(a1))
                (void) a0->structure()->new_bond(a0, a1);
        }
    }
}

void
ExtractMolecule::parse_entity_poly_seq(bool /*in_loop*/)
{
    string entity_id;
    long seq_id = 0;
    string mon_id;
    bool hetero = false;
    vector<Residue*> previous, current;

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
            mon_id = string(start, end - start);
        });
    pv.emplace_back(get_column("hetero"), false,
        [&] (const char* start, const char*) {
            hetero = *start == 'Y' || *start == 'y';
        });

    string last_entity_id = "";
    bool last_hetero = false;
    bool gap = false;
    while (parse_row(pv)) {
        if (entity_id != last_entity_id) {
            if (!previous.empty()) {
                connect_residue_pairs(previous, current, gap);
                gap = false;
            }
            previous.clear();
            current.clear();
            last_entity_id = entity_id;
            last_hetero = false;
            gap = false;
        }
        auto ri = residue_map.find(ResidueKey(entity_id, seq_id, mon_id));
        if (ri == residue_map.end()) {
            gap = true;
            continue;
        }
        Residue* r = ri->second;
        if (hetero) {
            if (last_hetero) {
                current.push_back(r);
            } else {
                if (!previous.empty()) {
                    connect_residue_pairs(previous, current, gap);
                    gap = false;
                }
                previous = std::move(current);
                current.clear();
                current.push_back(r);
            }
        } else {
            if (!previous.empty()) {
                connect_residue_pairs(previous, current, gap);
                    gap = false;
                }
            previous = std::move(current);
            current.clear();
            current.push_back(r);
        }

        last_hetero = hetero;
    }
    if (!previous.empty())
        connect_residue_pairs(previous, current, gap);
}

bool
init_structaccess()
{
    // ensure structaccess module objects are initialized
    PyObject* structaccess_mod = PyImport_ImportModule("chimera.core.structaccess");
    return structaccess_mod != nullptr;
}

PyObject*
parse_mmCIF_file(const char *filename)
{
#ifdef CLOCK_PROFILING
clock_t start_t, end_t;
#endif
    ExtractMolecule extract;

    extract.parse_file(filename);

    using blob::StructBlob;
    StructBlob* sb = static_cast<StructBlob*>(blob::newBlob<StructBlob>(&blob::StructBlob_type));
    for (auto m: extract.all_molecules) {
        if (m->atoms().size() == 0)
            continue;
        sb->_items->emplace_back(m);
    }
    return sb;
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file)
{
#ifdef CLOCK_PROFILING
clock_t start_t, end_t;
#endif
    ExtractMolecule extract;

    extract.parse(reinterpret_cast<const char *>(whole_file));

    using blob::StructBlob;
    StructBlob* sb = static_cast<StructBlob*>(blob::newBlob<StructBlob>(&blob::StructBlob_type));
    for (auto m: extract.all_molecules) {
        if (m->atoms().size() == 0)
            continue;
        sb->_items->emplace_back(m);
    }
    return sb;
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
