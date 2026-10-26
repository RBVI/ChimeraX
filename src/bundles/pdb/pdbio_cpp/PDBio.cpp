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

#include "PDBio.h"

std::string pdb_segment("pdb_segment");
std::string pdb_charge("formal_charge");
std::string pqr_charge("charge");


// standard_polymeric_res_names contains the names of residues that should use
// PDB ATOM records.
static std::set<ResName> standard_polymeric_res_names = {
    // "N" and "DN", are basically "UNK" for nucleic acids
    "A", "ALA", "ARG", "ASN", "ASP", "ASX", "C", "CYS", "DA", "DC", "DG", "DN", "DT",
    "G", "GLN", "GLU", "GLX", "GLY", "HIS", "I", "ILE", "LEU", "LYS", "MET", "N",
    "PHE", "PRO", "SER", "T", "THR", "TRP", "TYR", "U", "UNK", "VAL"
};

static void
canonicalize_atom_name(AtomName& aname, bool *asterisks_translated)
{
    for (int i = aname.length(); i > 0; ) {
        --i;
        // strip embedded blanks
        if (aname[i] == ' ') {
            aname.replace(i, 1, "");
            continue;
        }
        // use prime instead of asterisk
        if (aname[i] == '*') {
            aname[i] = '\'';
            *asterisks_translated = true;
        }
    }
}

static void
canonicalize_res_name(ResName& rname)
{
    for (int i = rname.length(); i > 0; ) {
        --i;
        if (rname[i] == ' ') {
            rname.replace(i, 1, "");
            continue;
        }
        rname[i] = toupper(rname[i]);
    }
}

void correct_chain_ids(std::vector<Residue*>& chain_residues, unsigned char second_chain_id_let,
    bool *two_let_chains)
{
    for (auto r: chain_residues) {
        auto name = r->name();
        name.pop_back();
        r->set_name(name);
        auto cid = r->chain_id();
        cid.insert(0, 1, second_chain_id_let);
        r->set_chain_id(cid);
    }
    if (second_chain_id_let != '\0')
        *two_let_chains = true;
}

#define MCS_FILL 0
#define MCS_SKIP 1
#define MCS_COMPACT 2

#ifdef CLOCK_PROFILING
#include <ctime>
static clock_t cum_preloop_t, cum_loop_preswitch_t, cum_loop_switch_t, cum_loop_postswitch_t, cum_postloop_t;
#endif
// return nullptr on error
// return input if PDB records implying a structure encountered
// return PyNone otherwise (e.g. only blank lines, MASTER records, etc.)
static void *
read_one_structure(std::pair<const char *, PyObject *> (*read_func)(void *),
    void *input, Structure *as,
    int *line_num, std::unordered_map<int, Atom *> &asn,
    std::vector<Residue *> *start_residues,
    std::vector<Residue *> *end_residues,
    std::vector<PDB> *secondary_structure,
    std::vector<PDB::Conect_> *conect_records,
    std::vector<PDB> *link_ssbond_records,
    std::set<MolResId> *mod_res, bool *reached_end, PyObject *py_logger, bool explode, bool *eof,
    std::set<Residue*>& het_res, bool segid_chains, int missing_coordsets, bool *two_let_chains)
{
    bool        start_connect = true;
    int            in_model = 0;
    Structure::Residues::size_type cur_res_index = 0;
    Residue        *cur_residue = nullptr;
    MolResId    cur_rid;
    PDB            record;
    bool        actual_structure = false;
    bool        in_headers = true;
    bool        is_SCOP = false;
    bool        is_babel = false; // have we seen Babel-style atom names?
    bool        recent_TER = false;
    bool        atom_het_TER = false;
    bool        break_hets = false;
    bool        redo_elements = false;
    unsigned char  let, second_chain_id_let = '\0';
    ChainID  seqres_cur_chain;
    int         seqres_cur_count;
    bool        dup_MODEL_numbers = false;
    std::vector<Residue*> chain_residues;
    bool        second_chain_let_okay = true;
    std::map<std::string, decltype(let)> modres_mappings;
#ifdef CLOCK_PROFILING
clock_t     start_t, end_t;
start_t = clock();
#endif

    *reached_end = false;
    *eof = true;
    *two_let_chains = false;
    PDB::reset_state();
#ifdef CLOCK_PROFILING
end_t = clock();
cum_preloop_t += end_t - start_t;
#endif
    while (true) {
#ifdef CLOCK_PROFILING
start_t = clock();
#endif
        std::pair<const char *, PyObject *> read_vals = (*read_func)(input);
        if (PyErr_Occurred() != nullptr)
            return nullptr;
        const char *char_line = read_vals.first;
        if (char_line[0] == '\0') {
            Py_XDECREF(read_vals.second);
            break;
        }
        *eof = false;
        *line_num += 1;
        // allow for initial Unicode byte-order marker
        std::string line(char_line);
        if (*line_num == 1 && line.size() >= 3 && line[0] == '\xEF'
        && line[1] == '\xBB' && line[2] == '\xBF') {
            line.erase(0, 3);
        }

        // extra set of parens on next line to disambiguate from function decl
        std::istringstream is((line));
        Py_XDECREF(read_vals.second);
        is >> record;

#ifdef CLOCK_PROFILING
end_t = clock();
cum_loop_preswitch_t += end_t - start_t;
start_t = end_t;
#endif
        switch (record.type()) {

        default:    // ignore other record types
            break;

        case PDB::UNKNOWN:
            if (record.unknown.junk[0] & 0200) {
                logger::error(py_logger, "Non-ASCII character on line ",
                    *line_num, " of PDB file");
                PyErr_SetString(PyExc_ValueError, "PDB file contains non-ASCII character"
                    " or control character");
                return nullptr;
            }
            logger::warning(py_logger, "Ignored bad PDB record found on line ",
                *line_num, '\n', is.str());
            break;

        case PDB::HEADER:
            // SCOP doesn't provide MODRES records for HETATMs...
            if (strstr(record.header.classification, "SCOP/ASTRAL") != nullptr) {
                is_SCOP = true;
            }
            break;

        case PDB::MODRES:
            mod_res->insert(MolResId(record.modres.res.chain_id,
                    record.modres.res.seq_num,
                    record.modres.res.i_code));
            let = Sequence::protein3to1(record.modres.std_res);
            { // switch statement can't jump past declaration, so add a scope
            auto lookup = modres_mappings.find(record.modres.res.name);
            if (let != 'X') {
                if (lookup == modres_mappings.end()) {
                    Sequence::assign_rname3to1(record.modres.res.name, let, true);
                    modres_mappings[record.modres.res.name] = let;
                } else {
                    if (lookup->second != 'X' && lookup->second != let) {
                        Sequence::assign_rname3to1(record.modres.res.name, 'X', true);
                        modres_mappings[record.modres.res.name] = 'X';
                    }
                }
            } else {
                let = Sequence::nucleic3to1(record.modres.std_res);
                if (let != 'X') {
                    if (lookup == modres_mappings.end()) {
                        Sequence::assign_rname3to1(record.modres.res.name, let, false);
                        modres_mappings[record.modres.res.name] = let;
                    } else {
                        if (lookup->second != 'X' && lookup->second != let) {
                            Sequence::assign_rname3to1(record.modres.res.name, 'X', false);
                            modres_mappings[record.modres.res.name] = 'X';
                        }
                    }
                }
            }}
            break;

        case PDB::HELIX:
        case PDB::SHEET:
            if (secondary_structure)
                secondary_structure->push_back(record);
        case PDB::TURN:
            break;

        case PDB::MODEL: {
            cur_res_index = 0;
            if (in_model && !as->residues().empty())
                cur_residue = as->residues()[0];
            else {
                cur_residue = nullptr;
                if (in_model)
                    // either the first model was empty or we have
                    // consecutive MODEL records with no intervening
                    // ATOM or ENDMDL records; prevent this MODEL
                    // from being treated as the second MODEL...
                    in_model--;
            }
            in_model++;
            // set coordinate set name to model#
            int csid = record.model.serial;
            if (in_model > 1) {
                if (in_model == 2 && csid == as->active_coord_set()->id())
                    dup_MODEL_numbers = true;
                if (dup_MODEL_numbers)
                    csid = as->active_coord_set()->id() + in_model - 1;
                // make additional CoordSets same size as others
                int cs_size = as->active_coord_set()->coords().size();
                if (!explode && csid > as->active_coord_set()->id() + 1) {
                    if (missing_coordsets == MCS_FILL) {
                        // fill in coord sets for Monte-Carlo trajectories
                        const CoordSet *acs = as->active_coord_set();
                        for (int fill_in_ID = acs->id()+1; fill_in_ID < csid; ++fill_in_ID) {
                            CoordSet *cs = as->new_coord_set(fill_in_ID, cs_size);
                            cs->fill(acs);
                        }
                    } else if (missing_coordsets == MCS_COMPACT) {
                        csid = as->active_coord_set()->id() + 1;
                    }
                    // do nothing for MSC_SKIP
                }
                CoordSet *cs = as->new_coord_set(csid, cs_size);
                as->set_active_coord_set(cs);
                as->is_traj = true;
            } else {
                // first CoordSet starts empty
                CoordSet *cs = as->new_coord_set(csid);
                as->set_active_coord_set(cs);
            }
            break;
        }

        case PDB::ENDMDL:
            if (explode)
                goto finished;
            if (in_model > 1 && as->coord_sets().size() > 1) {
                // fill in coord set for Monte-Carlo
                // trajectories if necessary
                CoordSet *acs = as->active_coord_set();
                const CoordSet *prev_cs = as->find_coord_set(acs->id()-1);
                if (prev_cs != nullptr && acs->coords().size() < prev_cs->coords().size())
                    acs->fill(prev_cs);
            }
            break;

        case PDB::END:
            *reached_end = true;
            goto finished;

        case PDB::TER:
            // this code replicated if chain ID changes w/o TER in ATOM/HETATM case below
            start_connect = true;
            recent_TER = true;
            break_hets = false;
            if ((second_chain_let_okay && chain_residues.size() > 1) || *two_let_chains)
                correct_chain_ids(chain_residues, second_chain_id_let, two_let_chains);
            second_chain_let_okay = true;
            second_chain_id_let = '\0';
            chain_residues.clear();
            break;

        case PDB::HETATM:
        case PDB::ATOM:
        case PDB::ATOMQR: {
            actual_structure = true;

            AtomName aname;
            ResName rname;
            auto cid = segid_chains ? ChainID(record.atom.seg_id[0] == '\0' ? " " : record.atom.seg_id)
                : ChainID({record.atom.res.chain_id});
            if (cur_residue != nullptr && cur_residue->chain_id() != cid) {
                // Chain ID changed w/o TER card; nonetheless treat like the TER is there [#6977]
                start_connect = true;
                recent_TER = true;
                break_hets = false;
                // Correct chain IDs for isolated residues if we've already seen
                // two-character chain IDs [#20849]
                if ((second_chain_let_okay && chain_residues.size() > 1) || *two_let_chains)
                    correct_chain_ids(chain_residues, second_chain_id_let, two_let_chains);
                second_chain_let_okay = true;
                second_chain_id_let = '\0';
                chain_residues.clear();
            }
            if (recent_TER)
                atom_het_TER = record.type() == PDB::HETATM;
                
            if (islower(record.atom.res.i_code))
                record.atom.res.i_code = toupper(record.atom.res.i_code);
            int seq_num = record.atom.res.seq_num;
            char i_code = record.atom.res.i_code;
            if (isdigit(i_code)) {
                // presumably an overflow due to a large number of residues
                seq_num = 10 * seq_num + (i_code - '0');
                i_code = ' ';
            }
            MolResId rid(cid, seq_num, i_code);
            rname = record.atom.res.name;
            if (second_chain_let_okay) {
                if (rname.size() < 4) {
                    second_chain_let_okay = false;
                } else {
                    let = rname[3];
                }
            }
            canonicalize_res_name(rname);
            if (recent_TER && cur_residue != nullptr && cur_residue->chain_id() == rid.chain)
                // HETATMs following a TER in the middle of
                // of chain should not be chained even if
                // they are found in MODRES records (e.g. the
                // CH3s in pdb:310d
                break_hets = true;
            recent_TER = false;
            if (in_model > 1) {
                if (MolResId(cur_residue) != rid
                || cur_residue->name() != rname) {
                    if (explode) {
                        if (cur_res_index + 1 < as->residues().size())
                            cur_residue = as->residues()[++cur_res_index];
                    } else {
                        // Monte-Carlo traj?
                        cur_residue = as->find_residue(cid, seq_num, i_code);
                        if (cur_residue == nullptr) {
                            // if chain ID is space and res is het,
                            // then chain ID probably should be
                            // space, check that...
                            cur_residue = as->find_residue(" ", seq_num, i_code);
                            if (cur_residue != nullptr)
                                rid = MolResId(' ', seq_num, i_code);
                        }
                    }
                }
                if (cur_residue == nullptr || MolResId(cur_residue) != rid 
                || cur_residue->name() != rname) {
                    std::stringstream err_msg;
                    err_msg << "Residue " << rid << " not in first model on line "
                        << *line_num << " of PDB file";
                    PyErr_SetString(PyExc_IOError, err_msg.str().c_str());
                    return nullptr;
                }
            } else if (cur_residue == nullptr || cur_rid != rid
            // modifying HETs can be inline...
            || (cur_residue->name() != rname && (record.type() != PDB::HETATM
            || standard_polymeric_res_names.find(cur_residue->name())
            == standard_polymeric_res_names.end())))
            {
                // on to new residue

                if (cur_residue != nullptr && cur_rid.chain != rid.chain) {
                    start_connect = true;
                } else if (record.type() == PDB::HETATM
                && (break_hets || (!is_SCOP
                && mod_res->find(rid) == mod_res->end()))) {
                    start_connect = true;
                } else if (cur_residue != nullptr && cur_residue->number() > rid.number
                && cur_residue->find_atom("OXT") !=  nullptr) {
                    // connected residue numbers can
                    // legitimately drop due to circular
                    // permutations; only break chain
                    // if previous residue has OXT in it
                    start_connect = true;
                } else if (record.type() == PDB::ATOM && atom_het_TER) {
                    // Prevent the HETATM->ATOM chain break repair code below
                    // from crossing an actual TER card {related to #16987)
                    start_connect = true;
                }

                // Some PDB files don't properly mark their
                // modified residues with MODRES records,
                // producing a spurious chain break between
                // the HETATM residue and preceding ATOM
                // residue.  We can't detect this condition
                // until we come out on the "other side" into
                // the following ATOM residue.  When we do,
                // remove the chain break.
                if (!start_connect && cur_residue != nullptr && record.type() == PDB::ATOM
                && standard_polymeric_res_names.find(cur_residue->name())
                    == standard_polymeric_res_names.end()
                && rid.chain != " " && mod_res->find(cur_rid) == mod_res->end()
                && cur_rid.chain == rid.chain){
                    // if there were several HETATM residues
                    // in a row, there may be multiple breaks
                    while (!end_residues->empty()) {
                        Residue *sr = start_residues->back();
                        if (sr->chain_id() != rid.chain)
                            break;
                        if (standard_polymeric_res_names.find(sr->name())
                        != standard_polymeric_res_names.end())
                            break;
                        Residue *er = end_residues->back();
                        if (er->chain_id() != rid.chain)
                            break;
                        start_residues->pop_back();
                        end_residues->pop_back();
                    }
                }
                if (start_connect) {
                    if (second_chain_let_okay)
                        correct_chain_ids(chain_residues, second_chain_id_let, two_let_chains);
                    second_chain_let_okay = true;
                    second_chain_id_let = '\0';
                    chain_residues.clear();
                }

                if (start_connect && cur_residue != nullptr)
                    end_residues->push_back(cur_residue);
                cur_rid = rid;
                cur_residue = as->new_residue(rname, rid.chain, rid.number, rid.insert);
                if (second_chain_let_okay) {
                    chain_residues.push_back(cur_residue);
                    if (let == ' ')
                        second_chain_let_okay = false;
                    else if (second_chain_id_let == '\0')
                        second_chain_id_let = let;
                    else if (let != second_chain_id_let)
                        second_chain_let_okay = false;
                }
                if (record.type() == PDB::HETATM)
                    het_res.insert(cur_residue);
                cur_res_index = as->residues().size() - 1;
                if (start_connect)
                    start_residues->push_back(cur_residue);
                start_connect = false;
                if (record.type() == PDB::ATOM)
                    atom_het_TER = false;
            }
            aname = record.atom.name;
            canonicalize_atom_name(aname, &as->asterisks_translated);
            Coord c(record.atom.xyz);
            if (in_model > 1) {
                Atom *a = cur_residue->find_atom(aname);
                if (a == nullptr) {
                    logger::error(py_logger, "Atom ", aname, " not in first"
                        " model on line ", *line_num, " of PDB file");
                    goto finished;
                }
                // ensure that the name uniquely identifies the atom;
                // if not, then use an index-based 'find'
                // (Monte Carlo trajectories had better use unique names!)
                if (cur_residue->count_atom(aname) > 1) {
                    // no lookup from coord_index to atom, so search the Residue...
                    unsigned int index = as->active_coord_set()->coords().size();
                    const Residue::Atoms &atoms = cur_residue->atoms();
                    for (Residue::Atoms::const_iterator rai = atoms.begin();
                    rai != atoms.end(); ++rai) {
                        Atom *ma = *rai;
                        if (ma->coord_index() == index) {
                            a = ma;
                            break;
                        }
                    }
                }
                a->set_coord(c);
                break;
            }
            const Element *e;
            if (!is_babel) {
                if (record.atom.element[0] != '\0')
                    e = &Element::get_element(record.atom.element);
                else {
                    if (strlen(record.atom.name) == 4
                    && record.atom.name[0] == 'H')
                        e = &Element::get_element(1);
                    else
                        e = &Element::get_element(record.atom.name);
                    if ((e->number() > 83 || e->number() == 61
                      || e->number() == 43 || e->number() == 0)
                      && record.atom.name[0] != ' ') {
                        // probably one of those funky PDB
                        // non-standard-residue atom names;
                        // try _just_ the second character...
                        char atsym[2];
                        atsym[0] = record.atom.name[1];
                        atsym[1] = '\0';
                        e = &Element::get_element(atsym);
                    }
                }

                if (e->number() == 0 && !(
                  // explicit lone pair
                  (record.atom.name[0] == 'L' &&
                  record.atom.name[1] == 'P')
                  // ambiguous atom or NMR pseudoatom
                  || (record.atom.name[0] == ' ' &&
                  (record.atom.name[1] == 'A'
                  || record.atom.name[1] == 'Q')))

                  // also not just garbage
                  && (isalpha(record.atom.name[0]) ||
                  (record.atom.name[0] == ' ' &&
                  isalpha(record.atom.name[1])))
                  ) {
                      // presumably a Babel "PDB" file
                    is_babel = true;
                }
            }
            if (is_babel) {
                // Babel mis-aligns names and uses
                // mixed-case for two-letter element names.
                // Try that.
                char babel_name[3];
                int name_start = isspace(record.atom.name[0]) ?  1 : 0;
                babel_name[0] = record.atom.name[name_start];
                babel_name[2] = '\0';
                if (record.atom.name[name_start+1] != '\0'
                && islower(record.atom.name[name_start+1]))
                    babel_name[1] = record.atom.name[name_start+1];
                else
                    babel_name[1] = '\0';
                e = &Element::get_element(babel_name);
                
            }
            Atom *a;
            if (record.atom.alt_loc != ' ' && cur_residue->count_atom(aname) == 1) {
                a = cur_residue->find_atom(aname);
                a->set_alt_loc(record.atom.alt_loc, true);
                a->set_coord(c);
                a->set_serial_number(record.atom.serial);
                a->set_bfactor(record.atom.temp_factor);
                a->set_occupancy(record.atom.occupancy);
            } else {
                a = as->new_atom(aname.c_str(), *e);
                if (record.atom.alt_loc)
                    a->set_alt_loc(record.atom.alt_loc, true);
                cur_residue->add_atom(a);
                a->set_coord(c);
                a->set_serial_number(record.atom.serial);
                if (record.type() == PDB::ATOMQR) {
                    a->register_attribute(pqr_charge, record.atomqr.charge);
                    if (record.atomqr.radius > 0.0)
                        a->set_radius(record.atomqr.radius);
                } else {
                    a->set_bfactor(record.atom.temp_factor);
                    a->set_occupancy(record.atom.occupancy);
                    if (record.atom.seg_id[0] != '\0')
                        a->residue()->register_attribute(pdb_segment, record.atom.seg_id);
                    if (record.atom.charge[0] != '\0')
                        a->register_attribute(pdb_charge, atoi(record.atom.charge));
                }
            }
            if (e->number() == 0 && aname != "LP" && aname != "lp")
                redo_elements = true;
            if (in_model == 0 && asn.find(record.atom.serial) != asn.end())
                logger::warning(py_logger, "Duplicate atom serial number"
                    " found: ", record.atom.serial);
            asn[record.atom.serial] = a;
            break;
        }

        case PDB::ANISOU: {
            int serial = record.anisou.serial;
            std::unordered_map<int, Atom *>::const_iterator si = asn.find(serial);
            if (si == asn.end()) {
                logger::warning(py_logger, "Unknown atom serial number (",
                    serial, ") in ANISOU record");
                break;
            }
            int *u = record.anisou.u;
            float u11 = *u++ / 10000.0;
            float u22 = *u++ / 10000.0;
            float u33 = *u++ / 10000.0;
            float u12 = *u++ / 10000.0;
            float u13 = *u++ / 10000.0;
            float u23 = *u++ / 10000.0;
            Atom *a = (*si).second;
            a->set_alt_loc(record.anisou.alt_loc);
            a->set_aniso_u(u11, u12, u13, u22, u23, u33);
            break;
        }
        case PDB::CONECT:
            conect_records->push_back(record.conect);
            break;

        case PDB::LINK:
            link_ssbond_records->push_back(record);
            break;

        case PDB::LINKR:
            link_ssbond_records->push_back(record);
            break;

        case PDB::SSBOND: {
            // process SSBOND records as CONECT because Phenix uses them that way
            link_ssbond_records->push_back(record);
            break;
        }

        case PDB::SEQRES: {
            auto cid_ptr = record.seqres.chain_id;
            if (*cid_ptr == ' ')
                cid_ptr++;
            auto chain_id = ChainID(cid_ptr);
            if (chain_id != seqres_cur_chain) {
                seqres_cur_chain = chain_id;
                seqres_cur_count = 0;
            }
            int rem = record.seqres.num_res - seqres_cur_count;
            int num_to_read = rem < 13 ? rem : 13;
            seqres_cur_count += num_to_read;
            for (int i = 0; i < num_to_read; ++i) {
                auto rn = record.seqres.res_name[i];
                // remove leading/trailing spaces
                while (*rn == ' ') ++rn;
                auto brn = rn;
                while (*brn != '\0') {
                    if (*brn == ' ' && (*brn == ' ' || *brn == '\0')) {
                        *brn = '\0';
                        break;
                    }
                    ++brn;
                }
                ResName res_name(rn);
                // some non-standard PDB files may not have a full 13 residues per line...
                if (res_name.empty()) {
                    seqres_cur_count -= 1;
                    continue;
                }
                as->extend_input_seq_info(chain_id, res_name);
            }
            if (as->input_seq_source.empty())
                as->input_seq_source = "PDB SEQRES record";
            break;
        }

        case PDB::OBSLTE: {
            for (int i = 0; i < 8; ++i) {
                auto r_id_code = record.obslte.r_id_code[i];
                if (r_id_code[0] != '\0')
                    logger::warning(py_logger, "Entry ", record.obslte.id_code,
                        " superseded by entry ", r_id_code);
            }
            break;
        }
        }
#ifdef CLOCK_PROFILING
end_t = clock();
cum_loop_switch_t += end_t - start_t;
start_t = end_t;
#endif

        // separate switch for recording headers, since some
        // of the records handled above are headers, and don't
        // want to duplicate code in multiple places
        if (in_headers) {
            switch (record.type()) {

            case PDB::MODEL:
            case PDB::ATOM:
            case PDB::HETATM:
            case PDB::ATOMQR:
            case PDB::SIGATM:
            case PDB::ANISOU:
            case PDB::SIGUIJ:
            case PDB::TER:
            case PDB::ENDMDL:
            case PDB::CONECT:
            case PDB::MASTER:
            case PDB::END:
                in_headers = 0;
                break;

            default:
                std::string key(line, 0, 6);
                // remove trailing spaces from key
                for (int i = key.length()-1; i >= 0 && key[i] == ' '; i--)
                    key.erase(i, 1);
                
                decltype(as->metadata)::mapped_type &h = as->metadata[key];
                decltype(as->metadata)::mapped_type::value_type hdr = line;
                hdr.pop_back(); // drop trailing newline
                if (hdr.back() == '\r') // drop trailing carriage return if present
                    hdr.pop_back();
                h.push_back(hdr);
                break;

            }
        }
#ifdef CLOCK_PROFILING
end_t = clock();
cum_loop_postswitch_t += end_t - start_t;
#endif
    }
    *reached_end = true;

finished:
#ifdef CLOCK_PROFILING
start_t = clock();
#endif
    // make the last residue an end residue
    if (cur_residue != nullptr) {
        end_residues->push_back(cur_residue);
    }
    as->pdb_version = record.pdb_input_version();
    if (second_chain_let_okay)
        correct_chain_ids(chain_residues, second_chain_id_let, two_let_chains);

    if (redo_elements) {
        char test_name[3];
        test_name[2] = '\0';
        for (auto& a: as->atoms()) {
            if (a->name().empty())
                continue;
            test_name[0] = a->name()[0];
            test_name[1] = '\0';
            const Element& e1 = Element::get_element(test_name);
            if (e1.number() != 0) {
                a->_switch_initial_element(e1);
                continue;
            }
            if (a->name().size() < 2)
                continue;
            test_name[1] = a->name()[1];
            const Element& e2 = Element::get_element(test_name);
            if (e2.number() != 0)
                a->_switch_initial_element(e2);
        }
    }
#ifdef CLOCK_PROFILING
cum_postloop_t += clock() - start_t;
#endif
    if (actual_structure)
        return input;
    Py_INCREF(Py_None);
    return Py_None;
}

inline static void
add_bond(Atom *a1, Atom *a2)
{
    if (!a1->connects_to(a2))
        (void) a1->structure()->new_bond(a1, a2);
}

// add_bond:
//    Add a bond to structure given two atom serial numbers.
//    (atom_serial_nums argument should be const, but operator[] isn't const)
static void
add_bond(std::unordered_map<int, Atom *> &atom_serial_nums, int from, int to, PyObject *py_logger)
{
    if (to <= 0 || from <= 0)
        return;
    if (to == from) {
        logger::warning(py_logger, "CONECT record from atom to itself: ", from);
        return;
    }
    // serial "from" check happens before this routine is called
    if (atom_serial_nums.find(to) == atom_serial_nums.end()) {
        logger::warning(py_logger, "CONECT record to nonexistent atom: (",
            from, ", ", to, ")");
        return;
    }
    Atom *a1 = atom_serial_nums[from];
    Atom *a2 = atom_serial_nums[to];
    if (a1 == a2) {
        logger::warning(py_logger, "CONECT record from alternate atom to itself: ", from);
        return;
    }
    add_bond(a1, a2);
}

// assign_secondary_structure:
//    Assign secondary structure state to residues using PDB
//    HELIX and SHEET records
static void
assign_secondary_structure(Structure *as, const std::vector<PDB> &ss, PyObject *py_logger, bool two_let_chains)
{
    std::vector<std::pair<Structure::Residues::const_iterator,
        Structure::Residues::const_iterator> > strand_ranges;
    int ss_id;
    for (std::vector<PDB>::const_iterator i = ss.begin(); i != ss.end(); ++i) {
        const PDB &r = *i;
        const PDB::Residue *init, *end;
        switch (r.type()) {
          case PDB::HELIX:
            init = &r.helix.init;
            end = &r.helix.end;
            ss_id = r.helix.ser_num;
            break;
          case PDB::SHEET:
            init = &r.sheet.init;
            end = &r.sheet.end;
            break;
          default:
            // Should not happen
            continue;
        }
        auto chain_id = ChainID({init->chain_id});
        ResName name = init->name;
        if (two_let_chains && name.size() == 4) {
            chain_id.insert(chain_id.begin(), name[3]);
            name.pop_back();
        }
        Residue *init_res = as->find_residue(chain_id, init->seq_num,
            init->i_code, name);
        if (init_res == nullptr) {
            logger::warning(py_logger, "Start residue of secondary structure"
                " not found: ", r.c_str());
            continue;
        }
        chain_id = ChainID({end->chain_id});
        name = end->name;
        if (two_let_chains && name.size() == 4) {
            chain_id.insert(chain_id.begin(), name[3]);
            name.pop_back();
        }
        Residue *end_res = as->find_residue(chain_id, end->seq_num,
            end->i_code, name);
        if (end_res == nullptr) {
            logger::warning(py_logger, "End residue of secondary structure"
                " not found: ", r.c_str());
            continue;
        }
        Structure::Residues::const_iterator first = as->residues().end();
        Structure::Residues::const_iterator last = as->residues().end();
        for (Structure::Residues::const_iterator
        ri = as->residues().begin(); ri != as->residues().end(); ++ri) {
            Residue *r = *ri;
            if (r == init_res)
                first = ri;
            if (r == end_res) {
                last = ri;
                break;
            }
        }
        if (first == as->residues().end()
        || last == as->residues().end()) {
            logger::warning(py_logger, "Bad residue range for secondary"
                " structure: ", r.c_str());
            continue;
        }
        if (r.type() == PDB::SHEET)
            strand_ranges.push_back(std::pair<Structure::Residues::const_iterator,
                Structure::Residues::const_iterator>(first, last));
        else  {
            for (Structure::Residues::const_iterator ri = first;
            ri != as->residues().end(); ++ri) {
                (*ri)->set_is_helix(true);
                (*ri)->set_ss_id(ss_id);
                if (ri == last)
                    break;
            }
        }
    }
    std::sort(strand_ranges.begin(), strand_ranges.end());
    int id = 0;
    char last_chain = '\0';
    for (std::vector<std::pair<Structure::Residues::const_iterator, Structure::Residues::const_iterator> >::iterator sri = strand_ranges.begin(); sri != strand_ranges.end(); ++sri) {
        char chain_id = (*sri->first)->chain_id()[0];
        if (chain_id != last_chain) {
            id = 0;
            last_chain = chain_id;
        }
        ++id;
        for (Structure::Residues::const_iterator ri = sri->first;
        ri != as->residues().end(); ++ri) {
            Residue *r = *ri;
            r->set_ss_id(id);
            r->set_is_strand(true);
            if (ri == sri->second)
                break;
        }
    }
}

static void
prune_short_bonds(Structure *as)
{
    std::vector<Bond *> short_bonds;

    const Structure::Bonds &bonds = as->bonds();
    for (Structure::Bonds::const_iterator bi = bonds.begin(); bi != bonds.end(); ++bi) {
        Bond *b = *bi;
        Coord c1 = b->atoms()[0]->coord();
        Coord c2 = b->atoms()[1]->coord();
        if (c1.sqdistance(c2) < 0.001)
            short_bonds.push_back(b);
    }

    for (std::vector<Bond *>::iterator sbi = short_bonds.begin();
            sbi != short_bonds.end(); ++sbi) {
        as->delete_bond(*sbi);
    }
}

static bool
extract_linkup_record_info(PDB& link_ssbond, int* sym1, int* sym2,
    PDB::Residue* pdb_res1, PDB::Residue* pdb_res2, PDB::Atom* pdb_atom1, PDB::Atom* pdb_atom2)
{
    if (link_ssbond.type() == PDB::SSBOND) {
        *sym1 = link_ssbond.ssbond.sym[0];
        *sym2 = link_ssbond.ssbond.sym[1];
        *pdb_res1 = link_ssbond.ssbond.res[0];
        *pdb_res2 = link_ssbond.ssbond.res[1];
        strcpy(*pdb_atom1, "SG");
        strcpy(*pdb_atom2, "SG");
    } else if (link_ssbond.type() == PDB::LINK) {
        *sym1 = link_ssbond.link.sym[0];
        *sym2 = link_ssbond.link.sym[1];
        *pdb_res1 = link_ssbond.link.res[0];
        *pdb_res2 = link_ssbond.link.res[1];
        strcpy(*pdb_atom1, link_ssbond.link.name[0]);
        strcpy(*pdb_atom2, link_ssbond.link.name[1]);
    } else if (link_ssbond.type() == PDB::LINKR) {
        // non-standard Refmac "LINKR" record; blank atom names indicate gap rather than link
        char* name_ptr = link_ssbond.linkr.name[0];
        bool non_space = false;
        while (*name_ptr != '\0')
            if (!isspace(*name_ptr++)) {
                non_space = true;
                break;
            }
        if (!non_space)
            return false;
        *pdb_res1 = link_ssbond.link.res[0];
        *pdb_res2 = link_ssbond.link.res[1];
        strcpy(*pdb_atom1, link_ssbond.link.name[0]);
        strcpy(*pdb_atom2, link_ssbond.link.name[1]);
    } else {
        std::stringstream err_msg;
        err_msg << "Trying to extact linkup info from non-LINK/SSBOND record (record is: '"
            << link_ssbond.c_str() << "')";
        throw std::logic_error(err_msg.str().c_str());
    }
    return true;
}

static Residue*
pdb_res_to_chimera_res(Structure* as, PDB::Residue& pdb_res)
{
    ResName rname = pdb_res.name;
    auto orig_rname = rname;
    ChainID cid({pdb_res.chain_id});
    canonicalize_res_name(rname);
    auto res = as->find_residue(cid, pdb_res.seq_num, pdb_res.i_code, rname);
    if (res != nullptr || orig_rname.size() < 4)
        return res;
    // try two-letter chain ID
    cid.insert(cid.begin(), orig_rname[3]);
    orig_rname.pop_back();
    canonicalize_res_name(orig_rname);
    return as->find_residue(cid, pdb_res.seq_num, pdb_res.i_code, orig_rname);
}

static Atom*
pdb_atom_to_chimera_atom(Structure* as, Residue* res, PDB::Atom& pdb_aname)
{
    AtomName aname = pdb_aname;
    canonicalize_atom_name(aname, &as->asterisks_translated);
    return res->find_atom(aname);
}

static void
link_up(PDB& link_ssbond, Structure *as, PyObject *py_logger)
{
    int sym1, sym2;
    PDB::Residue pdb_res1, pdb_res2;
    PDB::Atom pdb_atom1, pdb_atom2;
    if (!extract_linkup_record_info(link_ssbond, &sym1, &sym2, &pdb_res1, &pdb_res2, &pdb_atom1, &pdb_atom2))
        return; // "gap" Refmac non-standard LINKR record
    if (sym1 != sym2) {
        // don't use LINKs/SSBONDs to symmetry copies;
        // skip if symmetry operators differ (or blank vs. non-blank)
        // (FYI, 1555 is identity transform)
        return;
    }
    Residue *res1 = pdb_res_to_chimera_res(as, pdb_res1);
    if (!res1) {
        logger::warning(py_logger, "Cannot find LINK/SSBOND residue ", pdb_res1.name,
            " (", pdb_res1.seq_num, pdb_res1.i_code, ")");
        return;
    }
    Residue *res2 = pdb_res_to_chimera_res(as, pdb_res2);
    if (!res2) {
        logger::warning(py_logger, "Cannot find LINK/SSBOND residue ", pdb_res2.name,
            " (", pdb_res2.seq_num, pdb_res2.i_code, ")");
        return;
    }
    Atom* a1 = pdb_atom_to_chimera_atom(as, res1, pdb_atom1);
    if (a1 == nullptr) {
        logger::warning(py_logger, "Cannot find LINK/SSBOND atom ", pdb_atom1,
            " in residue ", res1->str());
        return;
    }
    Atom* a2 = pdb_atom_to_chimera_atom(as, res2, pdb_atom2);
    if (a2 == nullptr) {
        logger::warning(py_logger, "Cannot find LINK/SSBOND atom ", pdb_atom2,
            " in residue ", res2->str());
        return;
    }
    if (a1 == a2) {
        logger::warning(py_logger, "LINK or SSBOND record from atom to itself: ", pdb_atom1,
            " in residue ", res1->str());
        return;
    }
    if (!a1->connects_to(a2)) {
        as->new_bond(a1, a2);
    }
}

static std::pair<const char *, PyObject *>
read_no_fileno(void *py_file)
{
    const char *line;
    PyObject *py_line = PyFile_GetLine((PyObject *)py_file, 0);
    if (PyErr_Occurred() != nullptr)
        return std::pair<const char*, PyObject *>(line, Py_None);
    if (PyBytes_Check(py_line)) {
        line = PyBytes_AS_STRING(py_line);
    } else {
        line = PyUnicode_AsUTF8(py_line);
    }
    return std::pair<const char*, PyObject *>(line, py_line);
}

static char read_fileno_buffer[1024];
static std::pair<const char *, PyObject *>
read_fileno(void *f)
{
    if (fgets(read_fileno_buffer, 1024, (FILE *)f) == nullptr)
        read_fileno_buffer[0] = '\0';
    return std::pair<const char *, PyObject *>(read_fileno_buffer, nullptr);
}

static PyObject *
read_pdb(PyObject *pdb_file, PyObject *py_logger, bool explode, bool atomic, bool segid_chains,
    int missing_coordsets)
{
    std::vector<Structure *> file_structs;
    bool reached_end, two_letter_chains;
    std::unordered_map<Structure *, std::vector<Residue *> > start_res_map, end_res_map;
    std::unordered_map<Structure *, std::vector<PDB> > ss_map;
    typedef std::vector<PDB::Conect_> Conects;
    typedef std::unordered_map<Structure *, Conects> ConectMap;
    ConectMap conect_map;
    typedef std::vector<PDB> Links;
    typedef std::unordered_map<Structure *, Links> LinkMap;
    LinkMap link_map;
    std::unordered_map<Structure *, std::set<MolResId> > mod_res_map;
    // Atom Serial Numbers -> Atom*
    typedef std::unordered_map<int, Atom *> Asns;
    std::unordered_map<Structure *, Asns > asn_map;
    bool per_model_conects = false;
    int line_num = 0;
    bool eof;
    std::pair<const char *, PyObject *> (*read_func)(void *);
    void *input;
    std::vector<Structure *> *structs = new std::vector<Structure *>();
#ifdef CLOCK_PROFILING
clock_t start_t, end_t;
#endif
    auto notifications_off = atomstruct::DestructionNotificationsOff();
    PyObject *http_mod = PyImport_ImportModule("http.client");
    if (http_mod == nullptr)
        return nullptr;
    PyObject *http_conn = PyObject_GetAttrString(http_mod, "HTTPResponse");
    if (http_conn == nullptr) {
        Py_DECREF(http_mod);
        PyErr_SetString(PyExc_AttributeError,
            "HTTPResponse class not found in http.client module");
        return nullptr;
    }
    // nowadays, the result of gzip.open() and open() are normally indistinguishable,
    // and the gzip has a fileno of the original compressed file, which is unreadable,
    // so look for the 'from_compressed_source' attribute, which the chimerax.io module sets
    auto fcs = PyObject_GetAttrString(pdb_file, "from_compressed_source");
    bool is_inst = (fcs != nullptr && PyBool_Check(fcs) && fcs == Py_True) || 
        PyObject_IsInstance(pdb_file, http_conn) == 1;
    int fd;
    if (is_inst)
        // due to buffering issues, cannot handle a socket like it 
        // was a file, and compression streams return open _compressed_ file fd!
        fd = -1;
    else
        fd = PyObject_AsFileDescriptor(pdb_file);
    if (fd == -1) {
        read_func = read_no_fileno;
        input = pdb_file;
        PyErr_Clear();
        PyObject *io_mod = PyImport_ImportModule("io");
        if (io_mod == nullptr)
            return nullptr;
        PyObject *io_base = PyObject_GetAttrString(io_mod, "IOBase");
        if (io_base == nullptr) {
            Py_DECREF(io_mod);
            PyErr_SetString(PyExc_AttributeError, "IOBase class not found in io module");
            return nullptr;
        }
        int is_inst = PyObject_IsInstance(pdb_file, io_base);
        if (is_inst == 0)
            PyErr_SetString(PyExc_TypeError, "PDB file is not an instance of IOBase class");
        if (is_inst <= 0) {
            Py_DECREF(io_mod);
            Py_DECREF(io_base);
            return nullptr;
        }
    } else {
        read_func = read_fileno;
        input = fdopen(fd, "r");
    }
    while (true) {
#ifdef CLOCK_PROFILING
start_t = clock();
#endif
        Structure *as;
        if (atomic)
            as = new AtomicStructure(py_logger);
        else
            as = new Structure(py_logger);
        std::set<Residue*> het_res;
        void *ret = read_one_structure(read_func, input, as, &line_num, asn_map[as], &start_res_map[as],
            &end_res_map[as], &ss_map[as], &conect_map[as], &link_map[as], &mod_res_map[as], &reached_end,
            py_logger, explode, &eof, het_res, segid_chains, missing_coordsets, &two_letter_chains);
        if (ret == nullptr) {
            for (std::vector<Structure *>::iterator si = structs->begin();
            si != structs->end(); ++si) {
                delete *si;
            }
            delete as;
            return nullptr;
        }
#ifdef CLOCK_PROFILING
end_t = clock();
std::cerr << "read pdb: " << ((float)(end_t - start_t))/CLOCKS_PER_SEC << "\n";
start_t = end_t;
#endif
        if (ret == Py_None) {
            if (!file_structs.empty()) {
                // NMR ensembles can have trailing CONECT
                // records; integrate them before deleting 
                // the null structure
                if (per_model_conects)
                    conect_map[file_structs.back()] = conect_map[as];
                else {
                    Conects &conects = conect_map[as];
                    for (Conects::iterator ci = conects.begin();
                            ci != conects.end(); ++ci) {
                        PDB::Conect_ &conect = *ci;
                        int serial = conect.serial[0];
                        bool matched = false;
                        for (ConectMap::iterator cmi = conect_map.begin();
                        cmi != conect_map.end(); ++cmi) {
                            Structure *cm = (*cmi).first;
                            Conects &cm_conects = (*cmi).second;
                            Asns &asns = asn_map[cm];
                            if (asns.find(serial) != asns.end()) {
                                cm_conects.push_back(conect);
                                matched = true;
                            }
                        }
                        if (!matched) {
                            logger::warning(py_logger, "CONECT record for"
                                " nonexistent atom: ", serial);
                        }
                    }
                }
            }
            delete as;
            as = nullptr;
        } else {
            // give all members of an ensemble the same metadata
            if (explode && ! structs->empty()) {
                if (as->metadata.empty())
                    as->metadata = (*structs)[0]->metadata;
                if (ss_map[as].empty())
                    ss_map[as] = ss_map[(*structs)[0]];
            }
            if (per_model_conects || (!file_structs.empty() && !conect_map[as].empty())) {
                per_model_conects = true;
                conect_map[file_structs.back()] = conect_map[as];
                conect_map[as].clear();
            }
            structs->push_back(as);
            file_structs.push_back(as);
        }
#ifdef CLOCK_PROFILING
end_t = clock();
std::cerr << "assign CONECTs: " << ((float)(end_t - start_t))/CLOCKS_PER_SEC << "\n";
start_t = end_t;
#endif

        if (!reached_end)
            continue;

        per_model_conects = false;
        for (std::vector<Structure *>::iterator fsi = file_structs.begin();
        fsi != file_structs.end(); ++fsi) {
            Structure *fs = *fsi;
            Conects &conects = conect_map[fs];
            Asns &asns = asn_map[fs];
            std::set<Atom *> conect_atoms;
            for (Conects::iterator ci = conects.begin(); ci != conects.end(); ++ci) {
                PDB::Conect_ &conect = *ci;
                int from_serial = conect.serial[0];
                if (asns.find(from_serial) == asns.end()) {
                    logger::warning(py_logger, "CONECT record for nonexistent"
                        " atom: ", conect.serial[0]);
                    break;
                }
                bool has_covalent = false;
                for (int i = 1; i < 5; i += 1) {
                    add_bond(asns, from_serial, conect.serial[i], py_logger);
                }
                // purely cross-residue bonds are not
                // considered to completely specify an
                // atom's connectivity unless it is
                // the only atom in the residue
                Atom *fa = asns[from_serial];
                for (auto ta: fa->neighbors()) {
                    if (ta->residue() == fa->residue()) {
                        has_covalent = true;
                        break;
                    }
                }
                if (has_covalent || fa->residue()->atoms().size() == 1) {
                    conect_atoms.insert(fa);
                }
            }

            assign_secondary_structure(fs, ss_map[fs], py_logger, two_letter_chains);

            Links &links = link_map[fs];
            for (Links::iterator li = links.begin(); li != links.end(); ++li)
                link_up(*li, fs, py_logger);
            connect_structure(fs, &start_res_map[fs], &end_res_map[fs], &conect_atoms, &mod_res_map[fs], standard_polymeric_res_names, het_res);
            prune_short_bonds(fs);
            fs->use_best_alt_locs();
        }
#ifdef CLOCK_PROFILING
end_t = clock();
std::cerr << "find bonds: " << ((float)(end_t - start_t))/CLOCKS_PER_SEC << "\n";
start_t = end_t;
#endif

        if (eof)
            break;
        file_structs.clear();
        asn_map.clear();
        ss_map.clear();
        conect_map.clear();
        start_res_map.clear();
        end_res_map.clear();
        mod_res_map.clear();
    }
#ifdef CLOCK_PROFILING
std::cerr << "tot: " << ((float)clock() - start_t)/CLOCKS_PER_SEC << "\n";
std::cerr << "read_one breakdown:  pre-loop " << cum_preloop_t/(float)CLOCKS_PER_SEC << "  loop, pre-switch " << cum_loop_preswitch_t/(float)CLOCKS_PER_SEC << "  loop, switch " << cum_loop_switch_t/(float)CLOCKS_PER_SEC << "  loop, post-switch " << cum_loop_postswitch_t/(float)CLOCKS_PER_SEC << "  post-loop " << cum_postloop_t/(float)CLOCKS_PER_SEC << "\n";
#endif

    void **sa;
    PyObject *s_array = python_voidp_array(structs->size(), &sa);
    int i = 0;
    for (auto si = structs->begin(); si != structs->end(); ++si)
      sa[i++] = static_cast<void *>(*si);

    delete structs;
    return s_array;
}

static const char*
docstr_read_pdb_file = 
"read_pdb_file(f, log=None, explode=True, atomic=True, segid_chains=False)\n"
"\n"
"'f' is a file-like object open for reading containing the PDB info\n"
"'log' is a file-like object open for writing warnings/errors and other"
" information\n"
"'explode' controls whether NMR ensembles will be handled as separate models"
" (default True) or as one model with multiple coordinate sets (False)\n"
"'atomic' controls whether models are treated as atomic models with"
" standard chemical properties (default True) or as graphical models"
" (False)\n"
"'segid_chains' controls whether the segment ID is used in lieu of the chain ID\n"
"\n"
"Returns a numpy array of C++ pointers to AtomicStructure objects"
" (if 'atomic' is True, otherwise Structure objects)";

extern "C" PyObject *
read_pdb_file(PyObject *, PyObject *args)
{
    PyObject *pdb_file;
    PyObject *py_logger;
    int explode, atomic, segid_chains, missing_coordsets;
    if (!PyArg_ParseTuple(args, "OOpppi", &pdb_file, &py_logger, &explode, &atomic, &segid_chains,
            &missing_coordsets))
        return nullptr;
    return read_pdb(pdb_file, py_logger, explode, atomic, segid_chains, missing_coordsets);
}

static const char*
docstr_write_pdb_file = 
"write_pdb_file(structures, out, selected_only=False,"
" displayed_only=False, xforms=None, all_coordsets=True, pqr=False,"
" polymeric_res_names=None)\n"
"\n"
"'structures' is a sequence of C++ structure pointers\n"
"'out' is the output destination, a path name or a StringIO buffer\n"
"'selected_only' controls if only selected atoms will be written"
" (default False)\n"
"'displayed_only' controls if only displayed atoms will be written"
" (default False)\n"
"'xforms' is a sequence of 3x4 numpy arrays to transform the atom"
" coordinates of the corresponding structure."
" If None then untransformed coordinates will be used for all structures."
" Similarly, any None in the sequence will cause untransformed coordinates"
" to be used for that structure. (default None)\n"
"'all_coordsets' controls if all coordsets of a trajectory will be written"
" (as multiple MODELS) or just the current coordset"
" (default True = all coordsets)\n" \
"'pqr' controls whether to write PQR-style ATOM records (default False)\n"
"'h36' controls the handling of serial numbers of more than"
" 5 digits (maximum supported by PDB standard).  If False, then the sixth"
" column of ATOM records will be stolen for an additional digit (AMBER style), so up to"
" 999,999 atoms.  If True, then hybrid-36 encoding will be used (see"
" https://cci.lbl.gov/hybrid_36), so up to 87,440,031 atoms."
" 'polymeric_res_names' is a sequence of residue names that"
" should be output using ATOM records rather than HETATM records."
"\n";

static struct PyMethodDef pdbio_functions[] =
{
    { "read_pdb_file", (PyCFunction)read_pdb_file, METH_VARARGS, 
        docstr_read_pdb_file },
    { "write_pdb_file", (PyCFunction)write_pdb_file, METH_VARARGS,
        docstr_write_pdb_file },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef pdbio_def =
{
    PyModuleDef_HEAD_INIT,
    "pdbio",
    "Input/output for PDB files",
    -1,
    pdbio_functions,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit__pdbio()
{
    auto mod = PyModule_Create(&pdbio_def);
    auto res_names = PyFrozenSet_New(nullptr);
    for (auto res_name: standard_polymeric_res_names)
        PySet_Add(res_names, PyUnicode_FromString(res_name.c_str()));
    PyModule_AddObject(mod, "standard_polymeric_res_names", res_names);
    return mod;
}
