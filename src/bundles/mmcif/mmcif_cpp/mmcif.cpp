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


Atom* closest_atom_by_template(tmpl::Atom* ta, const tmpl::Residue* tr, const Residue* r)
{
    // Find closest (in hops) from existing (heavy) atom to given atom in template
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
    for (size_t i = 0; i != to_visit.size(); ++i) {
        ta = to_visit[i];
        if (visited.find(ta) != visited.end())
            continue;
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

const char* ExtractMolecule::builtin_categories[] = {
    "chimerax_audit_syntax",
    "audit_conform", "audit_syntax", "entity", "entity_poly", "entity_poly_seq",
    "atom_site", "atom_site_anisotrop",
    "atom_site_aniso",  // for small CIF detection
    "struct_conn", "struct_conf", "struct_sheet_range",
#ifdef SHEET_HBONDS
    "struct_sheet_order", "pdbx_struct_sheet_hbond",
#endif
    "chem_comp", "chem_comp_bond",
};
#define MIXED_CASE_BUILTIN_CATEGORIES 0

ExtractMolecule::ExtractMolecule(PyObject* logger, const StringVector& generic_categories, bool coordsets, bool atomic, bool ignore_styling):
    _logger(logger), first_model_num(INT_MAX), my_templates(nullptr),
    found_missing_poly_seq(false), coordsets(coordsets), atomic(atomic),
    ignore_styling(ignore_styling),guess_fixed_width_categories(false),
    verbose(false), hydrogens_missing_in_template(0)
{
    empty_residue_templates.insert("UNL");  // Unknown ligand
    empty_residue_templates.insert("UNX");  // Unknown atom or ion
    if (!ignore_styling)
        register_heuristic_stylized_detection();   // audit_conform and audit_syntax
    register_category("chimerax_audit_syntax",
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
    register_category("atom_site_aniso",
        // So parsing small v1 CIF files does treat this as atom_site table
        [] () {});
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
    // from 12/12/2023 PDB News item, https://www2.rcsb.org/news/656f4404d78e004e766a96c6:
    // "wwPDB has reserved a set of CCD IDs: 01 - 99, DRG, INH, LIG that will never be used in the PDB."
    missing_residue_templates.insert("DRG");
    missing_residue_templates.insert("INH");
    missing_residue_templates.insert("LIG");
    for (char ones = '1'; ones <= '9'; ++ones) {
        char resname[3] = { '0', ones, 0 };
        missing_residue_templates.insert(resname);
    }
    for (char tens = '1'; tens <= '9'; ++tens) {
        for (char ones = '0'; ones <= '9'; ++ones) {
            char resname[3] = { tens, ones, 0 };
            missing_residue_templates.insert(resname);
        }
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
    residue_names.clear();
}

const tmpl::Residue*
ExtractMolecule::find_template_residue(const ResName& name, bool start, bool stop)
{
    if (missing_residue_templates.find(name) != missing_residue_templates.end()) {
        if (my_templates) {
            auto tr = my_templates->find_residue(name);
            if (tr && tr->atoms_map().size() > 0)
                return tr;
        }
        return nullptr;
    }
    auto tr = mmcif::find_template_residue(name, start, stop);
    if (tr == nullptr) {
        // keep track of missing residues to prevent multiple warnings
        missing_residue_templates.insert(name);
        if (my_templates) {
            auto tr = my_templates->find_residue(name);
            if (tr && tr->atoms_map().size() > 0) {
                logger::warning(_logger,
                    "Unable to fetch template for '", name, "': will use incomplete information in mmCIF file");
                return tr;
            }
        }
        logger::warning(_logger,
            "Unable to fetch template for '", name, "': will connect using distance criteria");
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
ExtractMolecule::connect_residue_by_template(Residue* r, const tmpl::Residue* tr, int model_num, bool *has_metal)
{
    auto& atoms = r->atoms();
    if (atoms.size() <= 1)
        return;

    // Confirm all atoms in residue are in template.  If not, connect by distance.
    bool all_connected = true;
    for (auto&& a: atoms) {
        bool connected = false;
        auto bonds = a->bonds();
        for (auto&& b: bonds) {
            if (b->other_atom(a)->residue() == r) {
                connected = true;
                break;
            }
        }
        if (connected)
            continue;
        all_connected = false;

        tmpl::Atom *ta = tr->find_atom(a->name());
        if (ta)
            continue;

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
        if (!*has_metal) {
            for (auto&& atom: r->atoms()) {
                if (atom->element().is_metal()) {
                    *has_metal = true;
                    break;
                }
            }
        }
        pdb_connect::connect_residue_by_distance(r);
        return;
    }
    if (all_connected) {
        //logger::warning(_logger, "Ignored residue template for ", r->str(), ", already connected");
        return;
    }

    // foreach atom in residue
    //    connect up like atom in template
    for (auto&& a: atoms) {
        tmpl::Atom *ta = tr->find_atom(a->name());
        if (ta == nullptr)
            continue;
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
ExtractMolecule::_preopt()
{
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

    // prefetch residue templates, so we know the one-letter code
    // before set_input_seq_info()
    for (auto rname: residue_names) {
        (void) find_template_residue(rname);
    }
}

void
ExtractMolecule::_connect_inter_residues(Structure *mol, int model_num, ResidueSet &start_residues, ResidueSet &stop_residues)
{
    // Connect residues in entity_poly_seq.
    // Because some positions are heterogeneous, delay connecting
    // until next group of residues is found.
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
            //auto count = std::distance(pit.first, pit.second);
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
                    Residue* r = ri2->second;
                    string c_id = r->chain_id();
                    if (c_id == " ")
                        c_id = "' '";
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
    if (found_missing_poly_seq && !no_polymer && model_num == first_model_num)
        logger::warning(_logger, "Missing or incorrect <a='https://mmcif.wwpdb.org/dictionaries/mmcif_std.dic/Categories/entity_poly_seq.html'>sequence information</a>.  Inferred polymer connectivity.", true);
}

void
ExtractMolecule::_connect_intra_residues(Structure *mol, int model_num, ResidueSet &start_residues, ResidueSet &stop_residues)
{
    // connect residues in molecule with all_residues information
    bool has_metal = false;
    for (auto&& r : mol->residues()) {
        bool start = start_residues.find(r) != start_residues.end();
        bool stop = stop_residues.find(r) != stop_residues.end();
        auto tr = find_template_residue(r->name(), start, stop);
        if (tr == nullptr || tr->atoms_map().size() == 0) {
            if (model_num == first_model_num) {
                if (tr != nullptr && tr->atoms_map().size() == 0) {
                    if (empty_residue_templates.find(r->name()) == empty_residue_templates.end()) {
                        empty_residue_templates.insert(r->name());
                        logger::warning(_logger, "Empty ", r->name(),
                                        " residue template");
                    }
                }
                if (!has_metal) {
                    for (auto&& atom: r->atoms()) {
                        if (atom->element().is_metal()) {
                            has_metal = true;
                            break;
                        }
                    }
                }
            }
            pdb_connect::connect_residue_by_distance(r);
        } else {
            has_metal = has_metal || tr->has_metal();
            connect_residue_by_template(r, tr, model_num, &has_metal);
        }
    }
    if (has_metal)
        pdb_connect::find_and_add_metal_coordination_bonds(mol);
}

void
ExtractMolecule::_propagate_changes()
{
    // export mapping of label chain ids to entity ids.
    StringVector chain_mapping;
    chain_mapping.reserve(chain_entity_map.size() * 2);
    for (auto i: chain_entity_map) {
        chain_mapping.emplace_back(i.first);
        chain_mapping.emplace_back(i.second);
    }
    generic_tables["struct_asym"] = { "struct_asym", "id", "entity_id" };
    generic_tables["struct_asym data"] = chain_mapping;

    if (found_missing_poly_seq)
        generic_tables["_missing_poly_seq"] = { "true" };

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
}

void
ExtractMolecule::finished_parse()
{
    if (molecules.empty())
        return;

    _preopt();
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

        ResidueSet start_residues, stop_residues;
        _connect_inter_residues(mol, model_num, start_residues, stop_residues);
        _connect_intra_residues(mol, model_num, start_residues, stop_residues);

        if (found_missing_poly_seq)
            pdb_connect::find_missing_structure_bonds(mol);
    }

    _propagate_changes();
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
    StringVector data = parse_whole_category();
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
            logger::info(_logger, "error in chem_comp_bond on line ",
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
parse_mmCIF_file(const char *filename, PyObject* logger, bool coordsets, bool atomic, bool ignore_styling)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_file");
#endif
    ExtractMolecule extract(logger, StringVector(), coordsets, atomic, ignore_styling);
    extract.parse_file(filename);
    return structure_pointers(extract);
}

PyObject*
parse_mmCIF_file(const char *filename, const StringVector& generic_categories,
                 PyObject* logger, bool coordsets, bool atomic, bool ignore_styling)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_file2");
#endif
    ExtractMolecule extract(logger, generic_categories, coordsets, atomic, ignore_styling);
    extract.parse_file(filename);
    return structure_pointers(extract);
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file, PyObject* logger, bool coordsets, bool atomic, bool ignore_styling)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_buffer");
#endif
    ExtractMolecule extract(logger, StringVector(), coordsets, atomic, ignore_styling);
    extract.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(extract);
}

PyObject*
parse_mmCIF_buffer(const unsigned char *whole_file,
   const StringVector& generic_categories, PyObject* logger, bool coordsets, bool atomic, bool ignore_styling)
{
#ifdef CLOCK_PROFILING
    ClockProfile p("parse_mmCIF_buffer2");
#endif
    ExtractMolecule extract(logger, generic_categories, coordsets, atomic, ignore_styling);
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
        if (std::abs((std::ptrdiff_t) (p1 - p0)) != 1) {
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

#ifdef Py_LIMITED_API
    Py_ssize_t len = PyUnicode_GetLength(str);
#else
    Py_ssize_t len = PyUnicode_GET_LENGTH(str);
#endif
    if (len == 0) {
        Py_DECREF(str);
        return PyUnicode_FromString("''");
    }

    Py_UCS4 ch;
#ifdef Py_LIMITED_API
    Py_UCS4* data = PyUnicode_AsUCS4Copy(str);
# define PyUnicode_READ(kind, data, index) data[index]
#else
    int kind = PyUnicode_KIND(str);
    void* data = PyUnicode_DATA(str);
#endif
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
#ifdef Py_LIMITED_API
    PyMem_Free(data);
#endif
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
