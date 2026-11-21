// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// Declarations shared by the mmcif*.cpp files that make up ExtractMolecule,
// split up so that its parts compile in parallel

#ifndef mmcif_extract_h
#define mmcif_extract_h

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
#include <cstddef>

#undef CLOCK_PROFILING
#ifdef CLOCK_PROFILING
#include <ctime>
#endif

#ifdef Py_LIMITED_API
#define PyTuple_SET_ITEM PyTuple_SetItem
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

// typedef unordered_set<Residue*, hash_ResidueKey> ResidueSet;
typedef set<Residue*> ResidueSet;

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

inline bool
reasonable_bond_length(Atom* a1, Atom* a2, float distance = std::numeric_limits<float>::quiet_NaN())
{
    // use same criteria as pdb_connect::find_missing_structure_bonds
    Real idealBL = Element::bond_length(a1->element(), a2->element());
    Real sqlength;
    if (!std::isnan(distance))
        sqlength = distance * distance;
    else
        sqlength = a1->coord().sqdistance(a2->coord());
    // 3.0625 == 1.75 squared
    // (allows ASP /A:223 OD2 <-> PLP /A:409 N1 bond in 1aam
    // and SER /A:233 OG <-> NDP /A:300 O1X bond in 1a80
    // to not be classified as missing seqments)
    return (sqlength < 3.0625f * idealBL * idealBL);
}

struct ExtractMolecule: public readcif::CIFFile
{
    static const char* builtin_categories[];
    PyObject* _logger;
    ExtractMolecule(PyObject* logger, const StringVector& generic_categories, bool coordsets,
        bool atomic, bool ignore_styling);
    ~ExtractMolecule();
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();
    void _preopt();
    void _connect_inter_residues(Structure *mol, int model_num, ResidueSet &start_residues, ResidueSet &stop_residues);
    void _connect_intra_residues(Structure *mol, int model_num, ResidueSet &start_residues, ResidueSet &stop_residues);
    void _propagate_changes();
    void connect_polymer_pair(Residue* r0, Residue* r1, bool gap, bool nstd_okay, int model_num);
    void connect_residue_by_template(Residue* r, const tmpl::Residue* tr, int model_num, bool *has_metal);
    const tmpl::Residue* find_template_residue(const ResName& name, bool start = false, bool stop = false);
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
    bool ignore_styling;  // ignore any information about PDBx/mmCIF styling
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
    Residue* find_residue(const ChainResidueMap& crm, const ChainID& chain_id, const ResidueKey& rk);
    set<ResName> residue_names;
};

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
ExtractMolecule::find_residue(const ChainResidueMap& crm, const ChainID& chain_id, const ResidueKey& rk)
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

} // namespace mmcif

#endif  // mmcif_extract_h
