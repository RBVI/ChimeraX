// vim: set expandtab ts=4 sw=4:
#include "PDBio.h"
#include "atomstruct/AtomicStructure.h"
#include "atomstruct/Residue.h"
#include "atomstruct/Bond.h"
#include "atomstruct/Atom.h"
#include "atomstruct/CoordSet.h"
#include "blob/StructBlob.h"
#include "connectivity/connect.h"
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>

using std::string;
using std::vector;

using atomstruct::AtomicStructure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using atomstruct::Element;
using connectivity::MolResId;
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
#if 0
    register_category("struct_conn",
        [this] (bool in) {
            parse_atom_site(in/*_loop*/);
        }, { "atom_site" });
#endif
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

    Residue* cur_residue = NULL;
    string chain_id;
    int position;
    char ins_code = ' ';
    char alt_id = '\0';
    char symbol[3];
    string atom_name;
    string residue_name;
    int atom_serial = 0;
    int serial = 0;
    float x, y, z;
    float occupancy = FLT_MAX;
    float b_factor = FLT_MAX;

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
            if (position == 0) {
                // HETATM residues (waters) might be missing a sequence number
                if (cur_residue == NULL || cur_residue->chain_id() != chain_id)
                    position = 1;
                else
                    ++position;
            }
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
	while (parse_row(pv)) {
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

#if 0
    // bonds from struct_conn records
    const unsigned conn_type_id = get_column(struct_conn_names, "conn_type_id");
    const unsigned ptnr1_label_asym_id = get_column(struct_conn_names, "ptnr1_label_asym_id");
    const unsigned ptnr1_label_comp_id = get_column(struct_conn_names, "ptnr1_label_comp_id");
    const unsigned ptnr1_label_seq_id = get_column(struct_conn_names, "ptnr1_label_seq_id");
    const unsigned ptnr1_label_atom_id = get_column(struct_conn_names, "ptnr1_label_atom_id");
    const unsigned pdbx_ptnr1_label_alt_id = get_column(struct_conn_names, "pdbx_ptnr1_label_alt_id");
    const unsigned pdbx_ptnr1_PDB_ins_code = get_column(struct_conn_names, "pdbx_ptnr1_PDB_ins_code");
    const unsigned ptnr2_label_asym_id = get_column(struct_conn_names, "ptnr2_label_asym_id");
    const unsigned ptnr2_label_comp_id = get_column(struct_conn_names, "ptnr2_label_comp_id");
    const unsigned ptnr2_label_seq_id = get_column(struct_conn_names, "ptnr2_label_seq_id");
    const unsigned ptnr2_label_atom_id = get_column(struct_conn_names, "ptnr2_label_atom_id");
    const unsigned pdbx_ptnr2_label_alt_id = get_column(struct_conn_names, "pdbx_ptnr2_label_alt_id");
    const unsigned pdbx_ptnr2_PDB_ins_code = get_column(struct_conn_names, "pdbx_ptnr2_PDB_ins_code");

    StringVector search_columns1, search_columns2;
    if (ptnr1_label_asym_id != -1)
        search_columns1.push_back("label_asym_id");
    if (ptnr1_label_comp_id != -1)
        search_columns1.push_back("label_comp_id");
    if (ptnr1_label_seq_id != -1)
        search_columns1.push_back("label_seq_id");
    if (ptnr1_label_atom_id != -1)
        search_columns1.push_back("label_atom_id");
    if (pdbx_ptnr1_label_alt_id != -1)
        search_columns1.push_back("label_alt_id");
    if (pdbx_ptnr1_PDB_ins_code != -1)
        search_columns1.push_back("pdbx_PDB_ins_code");
    if (ptnr2_label_asym_id != -1)
        search_columns2.push_back("label_asym_id");
    if (ptnr2_label_comp_id != -1)
        search_columns2.push_back("label_comp_id");
    if (ptnr2_label_seq_id != -1)
        search_columns2.push_back("label_seq_id");
    if (ptnr2_label_atom_id != -1)
        search_columns2.push_back("label_atom_id");
    if (pdbx_ptnr2_label_alt_id != -1)
        search_columns2.push_back("label_alt_id");
    if (pdbx_ptnr2_PDB_ins_code != -1)
        search_columns2.push_back("pdbx_PDB_ins_code");

    for (unsigned i = 0, n = struct_conn.GetNumRows(); i < n; ++i) {
        const StringVector& row = struct_conn.GetRow(i);
        const string& type = row[conn_type_id];
        if (type != "covale" && type != "disulf")
            continue;   // skip hydrogen and modres bonds
        StringVector targets;
        if (ptnr1_label_asym_id != -1)
            targets.push_back(row[ptnr1_label_asym_id]);
        if (ptnr1_label_comp_id != -1)
            targets.push_back(row[ptnr1_label_comp_id]);
        if (ptnr1_label_seq_id != -1)
            targets.push_back(row[ptnr1_label_seq_id]);
        if (ptnr1_label_atom_id != -1)
            targets.push_back(row[ptnr1_label_atom_id]);
        if (pdbx_ptnr1_label_alt_id != -1) {
            const string& alt = row[pdbx_ptnr1_label_alt_id]; 
            targets.push_back(alt == "?" ? "." : alt );
        }
        if (pdbx_ptnr1_PDB_ins_code != -1)
            targets.push_back(row[pdbx_ptnr1_PDB_ins_code]);
        UIntVector results1;
        atom_site.Search(results1, targets, search_columns1);
        targets.clear();
        if (ptnr2_label_asym_id != -1)
            targets.push_back(row[ptnr2_label_asym_id]);
        if (ptnr2_label_comp_id != -1)
            targets.push_back(row[ptnr2_label_comp_id]);
        if (ptnr2_label_seq_id != -1)
            targets.push_back(row[ptnr2_label_seq_id]);
        if (ptnr2_label_atom_id != -1)
            targets.push_back(row[ptnr2_label_atom_id]);
        if (pdbx_ptnr2_label_alt_id != -1) {
            const string& alt = row[pdbx_ptnr2_label_alt_id]; 
            targets.push_back(alt == "?" ? "." : alt );
        }
        if (pdbx_ptnr2_PDB_ins_code != -1)
            targets.push_back(row[pdbx_ptnr2_PDB_ins_code]);
        UIntVector results2;
        atom_site.Search(results2, targets, search_columns2);

        for (auto j = results1.begin(); j != results1.end(); ++j) {
            Atom *a1 = mol->atoms()[*j].get();
            for (auto k = results2.begin(); k != results2.end(); ++k) {
                Atom *a2 = mol->atoms()[*k].get();
                try {
                    mol->new_bond(a1, a2);
                } catch (std::invalid_argument&) {
                    // already bonded
                }
            }
        }
    }
#endif
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
        sb->_items->emplace_back(m);
    }
    return sb;
}

} // namespace pdb
