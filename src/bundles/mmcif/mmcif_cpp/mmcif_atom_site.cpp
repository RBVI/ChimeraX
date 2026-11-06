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

#include "mmcif_extract.h"

namespace mmcif {

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
    bool is_corecif = false;
    try {
        get_column("fract_x", Required);
        is_corecif = true;
    } catch (std::runtime_error& e) {
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
                if (chain_id.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id = ChainID(" ");
            });
        pv.emplace_back(get_column("auth_asym_id"),
            [&] (const char* start, const char* end) {
                auth_chain_id = ChainID(start, end - start);
                if (auth_chain_id.size() == 1 && (*start == '.' || *start == '?'))
                    auth_chain_id = ChainID(" ");
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
                if (molecules.find(cur_model_num) != molecules.end()) {
                    logger::warning(_logger, "Previously completed PDB model ", cur_model_num,
                                    " found.  Trying to extend.");
                    mol = molecules[cur_model_num];
                    cur_entity_id.clear();
                    cur_seq_id = INT_MAX;
                    cur_auth_seq_id = INT_MAX;
                    cur_chain_id.clear();
                    cur_comp_id.clear();
                } else if (atomic) {
                    mol = molecules[cur_model_num] = new AtomicStructure(_logger);
                    mol->set_res_numbering_valid(RN_CANONICAL, true);
                } else {
                    mol = molecules[cur_model_num] = new Structure(_logger);
                    mol->set_res_numbering_valid(RN_CANONICAL, true);
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
                    mol->set_res_numbering_valid(RN_CANONICAL, true);
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
        }

        bool missing_entity_id = entity_id.empty();
        if (missing_entity_id) {
            // no entity_id, use chain id
            if (chain_id != " ")
                entity_id = chain_id;
            else
                entity_id = chain_id = auth_chain_id;
        }
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
            residue_names.insert(rname);
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
            bool missing_poly_seq = has_poly_seq.find(entity_id) == has_poly_seq.end();
            if (!missing_poly_seq) {
                // protect against bad input
                auto psi = poly.find(entity_id);
                auto& entity_poly_seq = psi->second.seq;
                auto ps_key = PolySeq(position, residue_name, false);
                auto pi = entity_poly_seq.lower_bound(ps_key);
                if (pi == entity_poly_seq.end() || pi->seq_id != position /*|| pi->mon_id != residue_name*/) {
                    logger::warning(_logger, "Ignoring sequence due to unknown position ", position, " on line ", line_number());
                    has_poly_seq.erase(entity_id);
                    missing_poly_seq = true;
                } else {
                    auto pi2 = entity_poly_seq.upper_bound(ps_key);
                    bool found = false;
                    for (auto i = pi; i != pi2; ++i) {
                        if (i->mon_id == residue_name) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        logger::warning(_logger, "Ignoring sequence due to mismatch at position ", position,
                                    ": expected \"", pi->mon_id, "\" got \"", residue_name, "\" on line ", line_number());
                        has_poly_seq.erase(entity_id);
                        missing_poly_seq = true;
                    }
                }
            }
            if (missing_poly_seq && non_poly.find(entity_id) == non_poly.end()) {
                // sequence not given in entity_poly_seq and not in a known non-polymeric entity
                auto tr = find_template_residue(residue_name);
                if (tr && tr->polymer_type() != PolymerType::PT_NONE) {
                    // only save polymer residues
                    if (position == INT_MAX) {
                        if (!missing_seq_id_warning) {
                            logger::warning(_logger, "Unable to infer polymer connectivity due to "
                                            "unspecified label_seq_id for residue \"",
                                            residue_name, "\" on line ", line_number());
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
                                                "' on line ", line_number());
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
                            "\" on line ", line_number(),
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
                if (chain_id1.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id1 = ChainID(" ");
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
                if (chain_id2.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id2 = ChainID(" ");
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

} // namespace mmcif
