// vim: set expandtab ts=4 sw=4:
#include "PDBio.h"
#include "atomstruct/AtomicStructure.h"
#include "atomstruct/Residue.h"
#include "atomstruct/Bond.h"
#include "atomstruct/Atom.h"
#include "atomstruct/CoordSet.h"
#include "blob/StructBlob.h"
#include "atomstruct/connect.h"
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>

using std::string;
using std::vector;
using std::unordered_map;
using std::hash;

using atomstruct::AtomicStructure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using atomstruct::Element;
using atomstruct::MolResId;
using basegeom::Coord;
	
#define LOG_PY_ERROR_NULL(arg) \
                if (log_file != Py_None) { \
                    std::stringstream msg; \
                    msg << arg; \
                    if (PyFile_WriteString(msg.str().c_str(), log_file) == -1) { \
                        PyErr_Clear(); \
                        return NULL; \
                    } \
                }
#define LOG_PY_ERROR_VOID(arg) \
                if (log_file != Py_None) { \
                    std::stringstream msg; \
                    msg << arg; \
                    if (PyFile_WriteString(msg.str().c_str(), log_file) == -1) { \
                        PyErr_Clear(); \
                        return; \
                    } \
                }

namespace pdb {

typedef vector<string> StringVector;
typedef vector<unsigned> UIntVector;

// DEBUG code
template <typename t> std::ostream&
operator<<(std::ostream& os, const vector<t>& v)
{
    for (auto i = v.begin(), e = v.end(); i != e; ++i) {
        os << *i;
        if (i != e - 1)
            os << ", ";
    }
    return os;
}

struct ExtractCIF: public readcif::CIFFile 
{
    ExtractCIF();
    virtual void data_block(const string& name);
    void parse(const char *whole_file);
    void parse_audit_conform(bool in_loop);
    void parse_atom_site(bool in_loop);
    void parse_struct_conn(bool in_loop);

    vector<AtomicStructure*> molecules;
    struct AtomKey {
        string chain_id;
        int position;
        char ins_code;
        char alt_id;
        string atom_name;
        string residue_name;
        AtomKey(const string& c, int p, char i, char a, const string& n, const string& r):
            chain_id(c), position(p), ins_code(i), alt_id(a), atom_name(n), residue_name(r) {}
        bool operator==(const AtomKey& k) const {
            return position == k.position && ins_code == k.ins_code
                && alt_id == k.alt_id && atom_name == k.atom_name
                && residue_name == k.residue_name && chain_id == k.chain_id;
        }
    };
    struct hash_AtomKey {
        size_t operator()(const AtomKey& k) const {
            return hash<string>()(k.chain_id) ^ hash<int>()(k.position)
                ^ hash<char>()(k.ins_code) ^ hash<char>()(k.alt_id)
                ^ hash<string>()(k.atom_name) ^ hash<string>()(k.residue_name);
        }
    };
    unordered_map<AtomKey, Atom*, hash_AtomKey> atom_map;
};

ExtractCIF::ExtractCIF()
{
    register_category("audit_conform",
        [this] (bool in) {
            parse_audit_conform(in/*_loop*/);
        });
    register_category("atom_site",
        [this] (bool in) {
            parse_atom_site(in/*_loop*/);
        });
    register_category("struct_conn",
        [this] (bool in) {
            parse_struct_conn(in/*_loop*/);
        }, { "atom_site" });
}

void
ExtractCIF::parse(const char *whole_file)
{
    molecules.clear();
    readcif::CIFFile::parse(whole_file);

    // TODO: bonds from templates

}

void
ExtractCIF::data_block(const string& name)
{
    molecules.push_back(new AtomicStructure);
    atom_map.clear();
}

void
ExtractCIF::parse_audit_conform(bool in_loop)
{
    if (molecules.size() == 0)
        throw ExtractCIF::error("missing data keyword");

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
        set_PDBx_stylized(true);
}

void
ExtractCIF::parse_atom_site(bool in_loop)
{
    if (molecules.size() == 0)
        throw ExtractCIF::error("missing data keyword");

    //const unsigned label_entity_id = get_column(atom_site_names, "label_entity_id");
    // x, y, z are not required by mmCIF, but are by us
    readcif::CIFFile::ParseValues pv;
	pv.reserve(20);

    string chain_id;            // label_asym_id
    int position;               // label_seq_id
    char ins_code = ' ';        // pdbx_PDB_ins_code
    char alt_id = '\0';         // label_alt_id
    string atom_name;           // label_atom_id
    string residue_name;        // label_comp_id
    char symbol[3];             // type_symbol
    int serial = 0;             // id
    float x, y, z;              // Cartn_[xyz]
    float occupancy = FLT_MAX;  // occupancy
    float b_factor = FLT_MAX;   // B_iso_or_equiv

	pv.emplace_back(get_column("id", false), false,
        [&] (const char* start, const char* end) {
            serial = readcif::str_to_int(start);
        });

	pv.emplace_back(get_column("label_asym_id", true), true,
        [&] (const char* start, const char* end) {
            chain_id = string(start, end - start);
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
	pv.emplace_back(get_column("label_comp_id", true), true,
			[&] (const char* start, const char* end) {
                residue_name = string(start, end - start);
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

    auto mol = molecules.back();
    int atom_serial = 0;
    Residue* cur_residue = NULL;
	while (parse_row(pv)) {
        if (position == 0) {
            // HETATM residues (waters) might be missing a sequence number
            if (cur_residue == NULL || cur_residue->chain_id() != chain_id)
                position = 1;
            else
                ++position;
        }

        if (cur_residue == NULL
        || cur_residue->chain_id() != chain_id
        || cur_residue->position() != position
        || cur_residue->insertion_code() != ins_code) {
            cur_residue = mol->new_residue(residue_name, chain_id,
                                                        position, ins_code);
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
            if (position != 0) {
                AtomKey k(chain_id, position, ins_code, alt_id, atom_name,
                                                            residue_name);
                atom_map[k] = a;
            }
        }
        Coord c(x, y, z);
        a->set_coord(c);
        if (serial) {
            atom_serial = serial;
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
ExtractCIF::parse_struct_conn(bool in_loop)
{
    if (molecules.size() == 0)
        throw ExtractCIF::error("missing data keyword");

#   define P1 "ptnr1"
#   define P2 "ptnr2"
#   define ASYM "_label_asym_id"
#   define COMP "_label_comp_id"
#   define SEQ "_label_seq_id"
#   define ATOM "_label_atom_id"
#   define ALT "_label_alt_id" // pdbx
#   define INS "_PDB_ins_code" // pdbx

    // bonds from struct_conn records
    string chain_id1, chain_id2;            // ptrn[12]_label_asym_id
    int position1, position2;               // ptrn[12]_label_seq_id
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

	pv.emplace_back(get_column(P1 ASYM, true), true,
        [&] (const char* start, const char* end) {
            chain_id1 = string(start, end - start);
        });
	pv.emplace_back(get_column("pdbx_" P1 INS, false), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code1 = ' ';
            else {
                // TODO: check if more than one character
                ins_code1 = *start;
            }
        });
	pv.emplace_back(get_column(P1 SEQ, true), false,
        [&] (const char* start, const char*) {
            position1 = readcif::str_to_int(start);
        });
	pv.emplace_back(get_column("pdbx_" P1 ALT, false), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1
            && (*start == '.' || *start == '?' || *start == ' '))
                alt_id1 = '\0';
            else {
                // TODO: what about more than one character?
                alt_id1 = *start;
            }
        });
	pv.emplace_back(get_column(P1 ATOM, true), true,
			[&] (const char* start, const char* end) {
                atom_name1 = string(start, end - start);
			});
	pv.emplace_back(get_column(P1 COMP, true), true,
			[&] (const char* start, const char* end) {
                residue_name1 = string(start, end - start);
			});

	pv.emplace_back(get_column(P2 ASYM, true), true,
        [&] (const char* start, const char* end) {
            chain_id2 = string(start, end - start);
        });
	pv.emplace_back(get_column("pdbx_" P2 INS, false), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1 && (*start == '.' || *start == '?'))
                ins_code2 = ' ';
            else {
                // TODO: check if more than one character
                ins_code2 = *start;
            }
        });
	pv.emplace_back(get_column(P2 SEQ, true), false,
        [&] (const char* start, const char*) {
            position2 = readcif::str_to_int(start);
        });
	pv.emplace_back(get_column("pdbx_" P2 ALT, false), true,
        [&] (const char* start, const char* end) {
            if (end == start + 1
            && (*start == '.' || *start == '?' || *start == ' '))
                alt_id2 = '\0';
            else {
                // TODO: what about more than one character?
                alt_id2 = *start;
            }
        });
	pv.emplace_back(get_column(P2 ATOM, true), true,
			[&] (const char* start, const char* end) {
                atom_name2 = string(start, end - start);
			});
	pv.emplace_back(get_column(P2 COMP, true), true,
			[&] (const char* start, const char* end) {
                residue_name2 = string(start, end - start);
			});

    auto mol = molecules.back();
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
#   undef P1
#   undef P2
#   undef ASYM
#   undef COMP
#   undef SEQ
#   undef ATOM
#   undef ALT
#   undef INS
}

PyObject *
parse_mmCIF(const char *whole_file, PyObject *log_file, bool explode)
{
#ifdef CLOCK_PROFILING
clock_t start_t, end_t;
#endif
    ExtractCIF extract;

    extract.parse(whole_file);

    // ensure structaccess module objects are initialized
    PyObject* structaccess_mod = PyImport_ImportModule("structaccess");
    if (structaccess_mod == NULL)
        return NULL;
    using blob::StructBlob;
    StructBlob* sb = static_cast<StructBlob*>(blob::newBlob<StructBlob>(&blob::StructBlob_type));
    for (auto m: extract.molecules) {
        // TODO: if not atoms, do not add
        sb->_items->emplace_back(m);
    }
    return sb;
}

} // namespace pdb
