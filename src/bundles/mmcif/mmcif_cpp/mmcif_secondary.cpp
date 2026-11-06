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
                if (chain_id1.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id1 = ChainID(" ");
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
                if (chain_id2.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id2 = ChainID(" ");
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

        auto ari = all_residues[first_model_num].find(chain_id1);
        if (ari == all_residues[first_model_num].end()) {
            logger::warning(_logger, "Invalid residue range for struct_conf \"",
                            id, "\": invalid chain \"", chain_id1,
                            "\", on line ", line_number());
            continue;
        }
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid residue range for struct_conf \"",
                            id, "\": invalid chain \"", chain_id1,
                            "\", on line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly.find(entity_id);
        if (psi == poly.end()) {
            logger::warning(_logger, "Invalid residue range for struct_conf \"",
                            id, "\": residue in non-polymer entity \"", entity_id,
                            "\", on line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second.seq;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for struct_conf \"",
                            id, "\": ends before it starts"
                            ", on line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end()) {}
            logger::warning(_logger,
                            "Bad residue range for struct_conf \"", id,
                            "\" on line ", line_number());
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
                if (chain_id1.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id1 = ChainID(" ");
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
                if (chain_id2.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id2 = ChainID(" ");
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
                            ", on line ", line_number());
            continue;
        }

        auto ari = all_residues[first_model_num].find(chain_id1);
        if (ari == all_residues[first_model_num].end()) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": invalid chain \"",
                            chain_id1, "\", on line ", line_number());
            continue;
        }
        auto cemi = chain_entity_map.find(chain_id1);
        if (cemi == chain_entity_map.end()) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": invalid chain \"",
                            chain_id1, "\", on line ", line_number());
            continue;
        }
        string entity_id = cemi->second;
        auto psi = poly.find(entity_id);
        if (psi == poly.end()) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": residue in non-polymer entity \"",
                            entity_id, "\", on line ", line_number());
            continue;
        }
        auto& entity_poly_seq = psi->second.seq;

        auto init_ps_key = PolySeq(position1, residue_name1, false);
        auto end_ps_key = PolySeq(position2, residue_name2, false);
        if (end_ps_key < init_ps_key) {
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\": ends before it starts"
                            ", on line ", line_number());
            continue;
        }
        auto init_ps = entity_poly_seq.lower_bound(init_ps_key);
        auto end_ps = entity_poly_seq.upper_bound(end_ps_key);
        if (init_ps == entity_poly_seq.end()) {
        // TODO: || end_ps == entity_poly_seq.end())
            logger::warning(_logger, "Invalid sheet range for struct_sheet_range \"",
                            sheet_id, ' ', id, "\" on line ", line_number());
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
                if (chain_id1.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id1 = ChainID(" ");
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
                if (chain_id2.size() == 1 && (*start == '.' || *start == '?'))
                    chain_id2 = ChainID(" ");
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
        if (entity_description.find(entity_id) != entity_description.end()) {
            logger::warning(_logger, "Skipping duplicate entity id ", entity_id);
            continue;
        }
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
        if (poly.find(entity_id) != poly.end()) {
            logger::warning(_logger, "Skipping duplicate entity_poly for id ", entity_id);
            continue;
        }
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
                            "' on line ", line_number());
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
                            "' on line ", line_number());
            // fake polymer entity to cut down on secondary warnings
            poly.emplace(entity_id, false);
        }
        has_poly_seq[entity_id] = true;
#if 0
        poly.at(entity_id).seq.emplace(seq_id, mon_id, hetero);
#else
        auto& seq = poly.at(entity_id).seq;
        PolySeq ps(seq_id, mon_id, hetero);
        auto psit = seq.lower_bound(ps);
        if (psit != seq.end() && psit->seq_id == seq_id) {
            auto psit2 = psit;
            for (;;) {
                if (psit2->seq_id == seq_id && psit2->mon_id == mon_id) {
                    // TODO: limit warnings to once per entity?
                    logger::warning(_logger, "Duplicate entity_poly_seq entry on line ", line_number());
                    goto skip;
                }
                ++psit2;
                if (psit2 == seq.end() || psit2->seq_id != seq_id)
                    break;
            }
            psit = psit2;
        }
        seq.insert(psit, ps);
    skip: ;
#endif
    }
}

} // namespace mmcif
