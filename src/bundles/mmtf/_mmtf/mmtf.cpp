// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2017 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <Python.h>
#include <map>
#include <vector>
#include <fstream>

#include <mmtf.hpp>
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/Pseudobond.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/connect.h>
#include <logger/logger.h>
#include <arrays/pythonarray.h> // Use python_voidp_array()

#undef REPORT_TIME
#ifdef REPORT_TIME
#include <time.h>
#endif

namespace {

using std::map;
using std::string;
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

typedef vector<AtomicStructure*> Models;

PyObject*
structure_pointers(const Models& models, const char *filename)
{
    int count = 0;
    for (auto m: models) {
        if (m->atoms().size() > 0) {
            m->set_name(filename);
            count += 1;
        }
    }

    void **sa;
    PyObject *s_array = python_voidp_array(count, &sa);
    int i = 0;
    for (auto m: models)
        if (m->atoms().size() > 0)
            sa[i++] = static_cast<void *>(m);

    return s_array;
}

const char HELIX[] = "helix";
const char STRAND[] = "strand";

Models
extract_data(const mmtf::StructureData& data, PyObject* _logger, bool coordset)
{
    // Data structure traversal based on example given at
    // https://github.com/rcsb/mmtf/blob/v1.0/spec.md#traversal

    // detect optional data
    bool has_bond_atom_list = data.bondAtomList.size() > 0;
    bool has_bond_order_list = data.bondOrderList.size() > 0;
    bool has_b_factor_list = data.bFactorList.size() > 0;
    bool has_occupancy_list = data.occupancyList.size() > 0;
    bool has_alt_loc_list = data.altLocList.size() > 0;
    bool has_sec_struct_list = data.secStructList.size() > 0;
    bool has_ins_code_list = data.insCodeList.size() > 0;
    bool has_sequence_index_list = data.sequenceIndexList.size() > 0;
    bool has_chain_name = data.chainNameList.size() > 0;

    int32_t chain_index = -1;
    int32_t group_index = -1;
    int32_t atom_index = -1;

    vector<Atom*> atoms(data.xCoordList.size(), nullptr);
    Models models;
    models.reserve(16);  // reduce memory use

    vector<int> per_chain_entity_index(data.numChains);
    for (int i = 0; i < data.entityList.size(); ++i) {
        for (int j: data.entityList[i].chainIndexList) {
            per_chain_entity_index[j] = i;
        }
    }

    for (int32_t model_index = 0; model_index < data.numModels; ++model_index) {
        auto m = new AtomicStructure(_logger);
        models.push_back(m);
        int32_t model_chain_count = data.chainsPerModel[model_index];
        // traverse chains
        for (auto _model_chain = 0; _model_chain < model_chain_count; ++_model_chain) {
            chain_index += 1;
            string chain_id = data.chainIdList[chain_index];
            string chain_name;
            if (has_chain_name)
                chain_name = data.chainNameList[chain_index];
            else
                chain_name = chain_id;
            int32_t chain_group_count = data.groupsPerChain[chain_index];
            auto& entity = data.entityList[per_chain_entity_index[chain_index]];

#ifdef TODO
            if (entity.type == "polymer") {
                //  want 3-letter residue names, mmtf gives single-letter
                //  (so no "(ABC)"s either)
                vector<ResName> seqres;
                seqres.reserve(entity.sequence.size());
                for (auto c: entity.sequence)
                    seqres.emplace_back(c);
                m->set_input_seq_info(chain_name, seqres);
                m->input_seq_source = "MMTF sequence";
            }
#endif

            // traverse groups
            const char* last_ss = nullptr;
            int ss_id = 0;
            char ins_code = ' ';
            int last_sequence_index = -1;
            Residue* last_residue = nullptr;
            bool gap = false;

            vector<std::pair<Residue*, Residue*>> gaps;
            for (auto _chain_group = 0; _chain_group < chain_group_count; ++_chain_group) {
                group_index += 1;
                auto group_id = data.groupIdList[group_index];  // ordinal
                if (has_ins_code_list) {
                    ins_code = data.insCodeList[group_index];
                    if (ins_code == '\x00')
                        ins_code = ' ';
                }

                group_id = data.groupIdList[group_index];
                auto group_type = data.groupTypeList[group_index];
                auto& group = data.groupList[group_type];

                int8_t sec_struct;
                int sequence_index;
                if (has_sec_struct_list)
                    sec_struct = data.secStructList[group_index];
                if (has_sequence_index_list) {
                    sequence_index = data.sequenceIndexList[group_index];
                    if (entity.type == "polymer")
                        gap = sequence_index != last_sequence_index + 1;
                    else
                        gap = false;
                    last_sequence_index = sequence_index;
                }

                const string& group_name = group.groupName;
                const auto& atom_name_list = group.atomNameList;
                const auto& element_list = group.elementList;
                const auto& bond_atom_list = group.bondAtomList;
                // formal_charge_list = group.formalChargeList;  // TODO
                // bond_order_list = group.bondOrderList;        // TODO

                auto r = m->new_residue(group_name.c_str(), chain_name.c_str(), group_id, ins_code);
                if (gap && last_residue)
                    gaps.emplace_back(last_residue, r);
                last_residue = r;
                if (has_sec_struct_list) {
                    // Code    Name
                    //   0   pi helix
                    //   1   bend
                    //   2   alpha helix
                    //   3   extended
                    //   4   3-10 helix
                    //   5   bridge
                    //   6   turn
                    //   7   coil
                    //  -1   undefined
                    if (sec_struct == 0 || sec_struct == 2 || sec_struct == 4) {
                        if (last_ss != HELIX)
                            ss_id += 1;
                        r->set_is_helix(true);
                        r->set_ss_id(ss_id);
                        last_ss = HELIX;
                    } else if (sec_struct == 3) {
                        if (last_ss != STRAND)
                            ss_id += 1;
                        r->set_is_strand(true);
                        r->set_ss_id(ss_id);
                        last_ss = STRAND;
                    } else
                        last_ss = nullptr;

                }
                // traverse atoms
                auto start_atom = atom_index + 1;
                map<string, Atom*> group_alt_atoms;
                auto group_atom_count = atom_name_list.size();
                for (size_t i = 0; i < group_atom_count; ++i) {
                    atom_index += 1;

                    auto x_coord = data.xCoordList[atom_index];
                    auto y_coord = data.yCoordList[atom_index];
                    auto z_coord = data.zCoordList[atom_index];
                    float b_factor;
                    float occupancy;
                    char alt_loc;
                    if (has_b_factor_list)
                        b_factor = data.bFactorList[atom_index];
                    if (has_occupancy_list)
                        occupancy = data.occupancyList[atom_index];
                    if (has_alt_loc_list)
                        alt_loc = data.altLocList[atom_index];
                    auto& atom_name = atom_name_list[i];
                    auto& atom_element = element_list[i];
                    // formal_charge = formal_charge_list[i];        // TODO

                    Atom* a;
                    if (group_alt_atoms.find(atom_name) != group_alt_atoms.end()) {
                        a = group_alt_atoms[atom_name];
                        a->set_alt_loc(alt_loc, true);
                    } else {
                        const Element& e =Element::get_element(atom_element.c_str());
                        atoms[atom_index] = a = m->new_atom(atom_name.c_str(), e);
                        r->add_atom(a);
                        if (has_alt_loc_list && alt_loc != '\x00') {
                            a->set_alt_loc(alt_loc, true);
                            group_alt_atoms[atom_name] = a;
                        }
                    }
                    Coord c(x_coord, y_coord, z_coord);
                    a->set_coord(c);
                    if (has_b_factor_list)
                        a->set_bfactor(b_factor);
                    if (has_occupancy_list)
                        a->set_occupancy(occupancy);
                }

                // connect bonds in residue
                for (size_t i = 0; i < bond_atom_list.size(); i += 2) {
                    // bond_order = bond_order_list[i / 2];  // TODO
                    auto a0 = atoms[start_atom + bond_atom_list[i]];
                    auto a1 = atoms[start_atom + bond_atom_list[i + 1]];
                    if (a0 == nullptr || a1 == nullptr) {
                        // ignore bonds for alternate atoms
                        // assumes that all 'A' atoms were created first
                        continue;
                    }
                    m->new_bond(a0, a1);
                }
            }

            // create gap bonds
            atomstruct::Proxy_PBGroup* missing_pbg = nullptr;
            if (gaps.size() > 0) {
                if (missing_pbg == nullptr)
                    missing_pbg = m->pb_mgr().get_group(
                        m->PBG_MISSING_STRUCTURE, atomstruct::AS_PBManager::GRP_NORMAL);
                for (auto& residues : gaps) {
                    Residue* r0 = residues.first;
                    Residue* r1 = residues.second;
                    Atom* a0 = nullptr;
                    Atom* a1 = nullptr;
                    find_nearest_pair(r0, r1, &a0, &a1);
                    if (a1 != nullptr)
                        missing_pbg->new_pseudobond(a0, a1);
                }
            }
        }
    }

    if (has_bond_atom_list) {
        // traverse inter-group bonds
        // PBGManager xlinks = nullptr;
        auto& bond_atom_list = data.bondAtomList;
        for (size_t i = 0; i < bond_atom_list.size(); i += 2) {
            // bond_order = data.bondOrderList[i / 2];    // TODO
            auto a0 = atoms[bond_atom_list[i]];
            auto a1 = atoms[bond_atom_list[i + 1]];
            // TODO:
            //    can bonds connect two models?
            //    if so, create pseudobond
            auto m0 = a0->structure();
            auto m1 = a1->structure();
            if (m0 == m1)
                m0->new_bond(a0, a1);
            // else:
            //     if xlinks == nullptr:
            //         xlinks = session.pb_manager.get_group("cross links");
            //     xlinks.new_pseudobond(a0, a1);
        }
    }

    for (auto m : models)
        find_and_add_metal_coordination_bonds(m);

    return models;
}

PyObject*
parse_MMTF_file(PyObject *, PyObject *args, PyObject *keywds)
{
    PyObject* tmp;
    const char *filename;
    PyObject* _logger;
    int coordsets;
    const char *kwlist[] = {"filename", "logger", "coordsets", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&Op"),
                                     (char **) kwlist,
                                     PyUnicode_FSConverter, &tmp,
                                     &_logger, &coordsets))
        return NULL;
    filename = PyBytes_AS_STRING(tmp);

#ifdef REPORT_TIME
    clock_t start_t = clock();
#endif

    mmtf::StructureData data;
    try {
        mmtf::decodeFromFile(data, filename);
    } catch (mmtf::DecodeError& e) {
        Py_DECREF(tmp);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
#ifdef REPORT_TIME
    clock_t end_t = clock();
    std::cerr << "Load MMTF file " << (end_t - start_t) / (float)CLOCKS_PER_SEC << " seconds\n";
    start_t = clock();
#endif
    Models models = extract_data(data, _logger, coordsets);
#ifdef REPORT_TIME
    end_t = clock();
    std::cerr << "Convert to ChimeraX objects " << (end_t - start_t) / (float)CLOCKS_PER_SEC << " seconds\n";
#endif

    Py_DECREF(tmp);
    return structure_pointers(models, filename);
}

static PyMethodDef mmtf_methods[] = {
  {const_cast<char*>("parse_MMTF_file"), (PyCFunction)parse_MMTF_file,
   METH_VARARGS|METH_KEYWORDS,
   "parse_MMTF_file(filename, logger, coordsets)\n"
   "\n"
   "Parse MMTF file into atomic structures\n"
   "Implemented in C++.\n"
  },
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mmtf_def =
{
    PyModuleDef_HEAD_INIT,
    "_mmtf",
    "MMTF file parser",
    -1,
    mmtf_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

} // namespace

PyMODINIT_FUNC
PyInit__mmtf()
{
    return PyModule_Create(&mmtf_def);
}
