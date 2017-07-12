// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <algorithm>  // for std::sort, std::find
#include <cctype>
#include <cmath> // abs
#include <fstream>
#include <set>
#include <sstream>
#include <stdio.h>  // fgets
#include <unordered_map>

#include "Python.h"

#include <arrays/pythonarray.h>	// Use python_voidp_array(), array_from_python()
#include <atomstruct/Atom.h>
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Bond.h>
#include <atomstruct/connect.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/PythonInstance.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Sequence.h>
#include <atomstruct/destruct.h>
#include <logger/logger.h>
#include <pdb/PDB.h>
#include <atomstruct/tmpl/residues.h>

namespace pdb {

using atomstruct::Atom;
using atomstruct::AtomicStructure;
using atomstruct::AtomName;
using atomstruct::Bond;
using atomstruct::ChainID;
using atomstruct::CoordSet;
using element::Element;
using atomstruct::MolResId;
using atomstruct::Real;
using atomstruct::Residue;
using atomstruct::ResName;
using atomstruct::Sequence;
using atomstruct::Structure;
using atomstruct::Coord;

std::string pdb_segment("pdb_segment");
std::string pdb_charge("formal_charge");
std::string pqr_charge("charge");

const std::vector<std::string> record_order = {
    "HEADER", "OBSLTE", "TITLE", "CAVEAT", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR",
    "REVDAT", "SPRSDE", "JRNL", "REMARK", "DBREF", "SEQADV", "SEQRES", "MODRES", "FTNOTE",
    "HET", "HETNAM", "HETSYN", "FORMUL", "HELIX", "SHEET", "TURN", "SSBOND", "LINK", "HYDBND",
    "SLTBRG", "CISPEP", "SITE", "CRYST1", "ORIGX1", "ORIGX2", "ORIGX3", "SCALE1", "SCALE2",
    "SCALE3", "MTRIX1", "MTRIX2", "MTRIX3", "TVECT", "MODEL", "ATOM", "SIGATM", "ANISOU",
    "SIGUIJ", "TER", "HETATM", "ENDMDL", "CONECT", "MASTER", "END",
};

static void
canonicalize_atom_name(AtomName& aname, bool *asterisks_translated)
{
    for (int i = aname.length(); i > 0; ) {
        --i;
        // strip embedded blanks
        if (aname[i] == ' ') {
            int j = i;
            do {
                aname[j] = aname[j+1];
                ++j;
            } while (aname[j-1] != '\0');
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
            auto j = i;
            do {
                rname[j] = rname[j+1];
            } while (rname[j++] != '\0');
            continue;
        }
        rname[i] = toupper(rname[i]);
    }
}

#ifdef CLOCK_PROFILING
#include <ctime>
static clock_t cum_preloop_t, cum_loop_preswitch_t, cum_loop_switch_t, cum_loop_postswitch_t, cum_postloop_t;
#endif
// return nullptr on error
// return input if PDB records implying a structure encountered
// return PyNone otherwise (e.g. only blank lines, MASTER records, etc.)
static void *
read_one_structure(std::pair<char *, PyObject *> (*read_func)(void *),
    void *input, AtomicStructure *as,
    int *line_num, std::unordered_map<int, Atom *> &asn,
    std::vector<Residue *> *start_residues,
    std::vector<Residue *> *end_residues,
    std::vector<PDB> *secondary_structure,
    std::vector<PDB::Conect_> *conect_records,
    std::vector<PDB::Link_> *link_records,
    std::set<MolResId> *mod_res, bool *reached_end,
    PyObject *py_logger, bool explode, bool *eof)
{
    bool        start_connect = true;
    int            in_model = 0;
    AtomicStructure::Residues::size_type cur_res_index = 0;
    Residue        *cur_residue = nullptr;
    MolResId    cur_rid;
    PDB            record;
    bool        actual_structure = false;
    bool        in_headers = true;
    bool        is_SCOP = false;
    bool        is_babel = false; // have we seen Babel-style atom names?
    bool        recent_TER = false;
    bool        break_hets = false;
    bool        redo_elements = false;
    unsigned char  let;
    ChainID  seqres_cur_chain;
    int         seqres_cur_count;
    bool        dup_MODEL_numbers = false;
#ifdef CLOCK_PROFILING
clock_t     start_t, end_t;
start_t = clock();
#endif

    *reached_end = false;
    *eof = true;
    PDB::reset_state();
#ifdef CLOCK_PROFILING
end_t = clock();
cum_preloop_t += end_t - start_t;
#endif
    while (true) {
#ifdef CLOCK_PROFILING
start_t = clock();
#endif
        std::pair<char *, PyObject *> read_vals = (*read_func)(input);
        char *line = read_vals.first;
        if (line[0] == '\0') {
            Py_XDECREF(read_vals.second);
            break;
        }
        *eof = false;

        // extra set of parens on next line to disambiguate from function decl
        std::istringstream is((std::string((char *)line)));
        Py_XDECREF(read_vals.second);
        is >> record;
        *line_num += 1;

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
            if (let != 'X') {
                Sequence::assign_rname3to1(record.modres.res.name, let, true);
            } else {
                let = Sequence::nucleic3to1(record.modres.std_res);
                if (let != 'X') {
                    Sequence::assign_rname3to1(record.modres.res.name, let,
                        false);
                }
            }
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
                    // fill in coord sets for Monte-Carlo
                    // trajectories
                    const CoordSet *acs = as->active_coord_set();
                    for (int fill_in_ID = acs->id()+1; fill_in_ID < csid; ++fill_in_ID) {
                        CoordSet *cs = as->new_coord_set(fill_in_ID, cs_size);
                        cs->fill(acs);
                    }
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
            start_connect = true;
            recent_TER = true;
            break_hets = false;
            break;

        case PDB::HETATM:
        case PDB::ATOM:
        case PDB::ATOMQR: {
            actual_structure = true;

            AtomName aname;
            ResName rname;
            auto cid = ChainID({record.atom.res.chain_id});
            if (islower(record.atom.res.chain_id))
                as->lower_case_chains = true;
            if (islower(record.atom.res.i_code))
                record.atom.res.i_code = toupper(record.atom.res.i_code);
            int seq_num = record.atom.res.seq_num;
            char i_code = record.atom.res.i_code;
            if (isdigit(i_code)) {
                // presumably an overflow due to a large
                // number of residues
                seq_num = 10 * seq_num + (i_code - '0');
                i_code = ' ';
            }
            MolResId rid(cid, seq_num, i_code);
            rname = record.atom.res.name;
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
                    logger::error(py_logger, "Residue ", rid, " not in first"
                        " model on line ", *line_num, " of PDB file");
                    goto finished;
                }
            } else if (cur_residue == nullptr || cur_rid != rid
            // modifying HETs can be inline...
            || (cur_residue->name() != rname && (record.type() != PDB::HETATM
                || cur_residue->is_het())))
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
                }

                // Some PDB files don't properly mark their
                // modified residues with MODRES records,
                // producing a spurious chain break between
                // the HETATM residue and preceding ATOM
                // residue.  We can't detect this condition
                // until we come out on the "other side" into
                // the following ATOM residue.  When we do,
                // remove the chain break.
                if (!start_connect && cur_residue != nullptr
                && record.type() == PDB::ATOM && cur_residue->is_het()
                && rid.chain != " " && mod_res->find(cur_rid) == mod_res->end()
                && cur_rid.chain == rid.chain){
                    // if there were several HETATM residues
                    // in a row, there may be multiple breaks
                    while (!end_residues->empty()) {
                        Residue *sr = start_residues->back();
                        if (sr->chain_id() != rid.chain)
                            break;
                        if (!sr->is_het())
                            break;
                        Residue *er = end_residues->back();
                        if (er->chain_id() != rid.chain)
                            break;
                        start_residues->pop_back();
                        end_residues->pop_back();
                    }
                }

                if (start_connect && cur_residue != nullptr)
                    end_residues->push_back(cur_residue);
                cur_rid = rid;
                cur_residue = as->new_residue(rname, rid.chain, rid.number, rid.insert);
                if (record.type() == PDB::HETATM)
                    cur_residue->set_is_het(true);
                cur_res_index = as->residues().size() - 1;
                if (start_connect)
                    start_residues->push_back(cur_residue);
                start_connect = false;
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
                a = as->new_atom(aname, *e);
                if (record.atom.alt_loc)
                    a->set_alt_loc(record.atom.alt_loc, true);
                cur_residue->add_atom(a);
                a->set_coord(c);
                a->set_serial_number(record.atom.serial);
                if (record.type() == PDB::ATOMQR) {
                    a->register_field(pqr_charge, record.atomqr.charge);
                    if (record.atomqr.radius > 0.0)
                        a->set_radius(record.atomqr.radius);
                } else {
                    a->set_bfactor(record.atom.temp_factor);
                    a->set_occupancy(record.atom.occupancy);
                    if (record.atom.seg_id[0] != '\0')
                        a->register_field(pdb_segment, record.atom.seg_id);
                    if (record.atom.charge[0] != '\0')
                        a->register_field(pdb_charge, atoi(record.atom.charge));
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
            (*si).second->set_aniso_u(u11, u12, u13, u22, u23, u33);
            break;
        }
        case PDB::CONECT:
            conect_records->push_back(record.conect);
            break;

        case PDB::LINK:
            link_records->push_back(record.link);
            break;

        case PDB::SSBOND: {
            // process SSBOND records as CONECT because midas
            // used to use them that way
            auto chain_id = ChainID({record.ssbond.res[0].chain_id});
            Residue *ssres = as->find_residue(chain_id,
                record.ssbond.res[0].seq_num, record.ssbond.res[0].i_code);
            if (ssres == nullptr)
                break;
            if (ssres->name() != record.ssbond.res[0].name) {
                logger::warning(py_logger, "First res name in SSBOND record (",
                    record.ssbond.res[0].name, ") does not match actual"
                    " residue (", ssres->name(), "); skipping.");
                break;
            }
            Atom *ap0 = ssres->find_atom("SG");
            if (ap0 == nullptr) {
                logger::warning(py_logger, "Atom SG not found in ", ssres);
                break;
            }

            chain_id = ChainID({record.ssbond.res[1].chain_id});
            ssres = as->find_residue(chain_id,
                record.ssbond.res[1].seq_num, record.ssbond.res[1].i_code);
            if (ssres == nullptr)
                break;
            if (ssres->name() != record.ssbond.res[1].name) {
                logger::warning(py_logger, "Second res name in SSBOND record (",
                    record.ssbond.res[1].name, ") does not match actual"
                    " residue (", ssres->name(), "); skipping.");
                break;
            }
            Atom *ap1 = ssres->find_atom("SG");
            if (ap1 == nullptr) {
                logger::warning(py_logger, "Atom SG not found in ", ssres);
                break;
            }
            if (!ap0->connects_to(ap1))
                (void) ap0->structure()->new_bond(ap0, ap1);
            break;
        }

        case PDB::SEQRES: {
            auto chain_id = ChainID({record.seqres.chain_id});
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
                        " superceded by entry ", r_id_code);
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
                std::string key((const char *)line, 6);
                // remove trailing spaces from key
                for (int i = key.length()-1; i >= 0 && key[i] == ' '; i--)
                    key.erase(i, 1);
                
                decltype(as->metadata)::mapped_type &h = as->metadata[key];
                decltype(as->metadata)::mapped_type::value_type hdr = line;
                hdr.pop_back(); // drop trailing newline
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
assign_secondary_structure(AtomicStructure *as, const std::vector<PDB> &ss, PyObject *py_logger)
{
    std::vector<std::pair<AtomicStructure::Residues::const_iterator,
        AtomicStructure::Residues::const_iterator> > strand_ranges;
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
        Residue *init_res = as->find_residue(chain_id, init->seq_num,
            init->i_code, name);
        if (init_res == nullptr) {
            logger::warning(py_logger, "Start residue of secondary structure"
                " not found: ", r.c_str());
            continue;
        }
        chain_id = ChainID({end->chain_id});
        name = end->name;
        Residue *end_res = as->find_residue(chain_id, end->seq_num,
            end->i_code, name);
        if (end_res == nullptr) {
            logger::warning(py_logger, "End residue of secondary structure"
                " not found: ", r.c_str());
            continue;
        }
        AtomicStructure::Residues::const_iterator first = as->residues().end();
        AtomicStructure::Residues::const_iterator last = as->residues().end();
        for (AtomicStructure::Residues::const_iterator
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
            strand_ranges.push_back(std::pair<AtomicStructure::Residues::const_iterator,
                AtomicStructure::Residues::const_iterator>(first, last));
        else  {
            for (AtomicStructure::Residues::const_iterator ri = first;
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
    for (std::vector<std::pair<AtomicStructure::Residues::const_iterator, AtomicStructure::Residues::const_iterator> >::iterator sri = strand_ranges.begin(); sri != strand_ranges.end(); ++sri) {
        char chain_id = (*sri->first)->chain_id()[0];
        if (chain_id != last_chain) {
            id = 0;
            last_chain = chain_id;
        }
        ++id;
        for (AtomicStructure::Residues::const_iterator ri = sri->first;
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
prune_short_bonds(AtomicStructure *as)
{
    std::vector<Bond *> short_bonds;

    const AtomicStructure::Bonds &bonds = as->bonds();
    for (AtomicStructure::Bonds::const_iterator bi = bonds.begin(); bi != bonds.end(); ++bi) {
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

static void
link_up(PDB::Link_ &link, AtomicStructure *as, std::set<Atom *> *conect_atoms,
                        PyObject *py_logger)
{
    if (link.sym[0] != link.sym[1]) {
        // don't use LINKs to symmetry copies;
        // skip if symmetry operators differ (or blank vs. non-blank)
        // (FYI, 1555 is identity transform)
        return;
    }
    AtomName aname;
    ResName rname;
    PDB::Residue res = link.res[0];
    ChainID cid({res.chain_id});
    rname = res.name;
    canonicalize_res_name(rname);
    Residue *res1 = as->find_residue(cid, res.seq_num, res.i_code, rname);
    if (!res1) {
        logger::warning(py_logger, "Cannot find LINK residue ", res.name,
            " (", res.seq_num, res.i_code, ")");
        return;
    }
    res = link.res[1];
    cid = ChainID({res.chain_id});
    rname = res.name;
    canonicalize_res_name(rname);
    Residue *res2 = as->find_residue(cid, res.seq_num, res.i_code, rname);
    if (!res2) {
        logger::warning(py_logger, "Cannot find LINK residue ", res.name,
            " (", res.seq_num, res.i_code, ")");
        return;
    }
    aname = link.name[0];
    canonicalize_atom_name(aname, &as->asterisks_translated);
    Atom *a1 = res1->find_atom(aname);
    if (a1 == nullptr) {
        logger::warning(py_logger, "Cannot find LINK atom ", aname,
            " in residue ", res1->str());
        return;
    }
    aname = link.name[1];
    canonicalize_atom_name(aname, &as->asterisks_translated);
    Atom *a2 = res2->find_atom(aname);
    if (a2 == nullptr) {
        logger::warning(py_logger, "Cannot find LINK atom ", aname,
            " in residue ", res2->str());
        return;
    }
    if (!a1->connects_to(a2)) {
        as->new_bond(a1, a2);
        conect_atoms->insert(a1);
        conect_atoms->insert(a2);
    }
}

static std::pair<char *, PyObject *>
read_no_fileno(void *py_file)
{
    char *line;
    PyObject *py_line = PyFile_GetLine((PyObject *)py_file, 0);
    if (PyBytes_Check(py_line)) {
        line = PyBytes_AS_STRING(py_line);
    } else {
        line = PyUnicode_AsUTF8(py_line);
    }
    return std::pair<char*, PyObject *>(line, py_line);
}

static char read_fileno_buffer[1024];
static std::pair<char *, PyObject *>
read_fileno(void *f)
{
    if (fgets(read_fileno_buffer, 1024, (FILE *)f) == nullptr)
        read_fileno_buffer[0] = '\0';
    return std::pair<char *, PyObject *>(read_fileno_buffer, nullptr);
}

static PyObject *
read_pdb(PyObject *pdb_file, PyObject *py_logger, bool explode)
{
    std::vector<AtomicStructure *> file_structs;
    bool reached_end;
    std::unordered_map<AtomicStructure *, std::vector<Residue *> > start_res_map, end_res_map;
    std::unordered_map<AtomicStructure *, std::vector<PDB> > ss_map;
    typedef std::vector<PDB::Conect_> Conects;
    typedef std::unordered_map<AtomicStructure *, Conects> ConectMap;
    ConectMap conect_map;
    typedef std::vector<PDB::Link_> Links;
    typedef std::unordered_map<AtomicStructure *, Links> LinkMap;
    LinkMap link_map;
    std::unordered_map<AtomicStructure *, std::set<MolResId> > mod_res_map;
    // Atom Serial Numbers -> Atom*
    typedef std::unordered_map<int, Atom *> Asns;
    std::unordered_map<AtomicStructure *, Asns > asn_map;
    bool per_model_conects = false;
    int line_num = 0;
    bool eof;
    std::pair<char *, PyObject *> (*read_func)(void *);
    void *input;
    std::vector<AtomicStructure *> *structs = new std::vector<AtomicStructure *>();
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
    PyObject *compression_mod = PyImport_ImportModule("_compression");
    if (compression_mod == nullptr)
        return nullptr;
    PyObject *compression_stream = PyObject_GetAttrString(compression_mod, "BaseStream");
    if (compression_stream == nullptr) {
        Py_DECREF(compression_mod);
        PyErr_SetString(PyExc_AttributeError,
            "BaseStream class not found in _compression module");
        return nullptr;
    }
    bool is_inst = PyObject_IsInstance(pdb_file, http_conn) == 1 || 
        PyObject_IsInstance(pdb_file, compression_stream) == 1;
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
        AtomicStructure *as = new AtomicStructure(py_logger);
        void *ret = read_one_structure(read_func, input, as, &line_num, asn_map[as],
          &start_res_map[as], &end_res_map[as], &ss_map[as], &conect_map[as],
          &link_map[as], &mod_res_map[as], &reached_end, py_logger, explode, &eof);
        if (ret == nullptr) {
            for (std::vector<AtomicStructure *>::iterator si = structs->begin();
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
                            AtomicStructure *cm = (*cmi).first;
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
        for (std::vector<AtomicStructure *>::iterator fsi = file_structs.begin();
        fsi != file_structs.end(); ++fsi) {
            AtomicStructure *fs = *fsi;
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

            assign_secondary_structure(fs, ss_map[fs], py_logger);

            Links &links = link_map[fs];
            for (Links::iterator li = links.begin(); li != links.end(); ++li)
                link_up(*li, fs, &conect_atoms, py_logger);
            connect_structure(fs, &start_res_map[fs], &end_res_map[fs], &conect_atoms, &mod_res_map[fs]);
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

// for sorting atoms by coord index (input order)...
namespace {

struct less_Atom: public std::binary_function<Atom*, Atom*, bool> {
    bool operator()(const Atom* a1, const Atom* a2) {
        return a1->coord_index() < a2->coord_index();
    }
};

}

static std::string
primes_to_asterisks(const char* orig_name)
{
    std::string new_name = orig_name;
    std::string::size_type pos;
    while ((pos = new_name.find("'")) != std::string::npos)
        new_name.replace(pos, 1, "*");
    return new_name;
}

static void
write_coord_set(std::ostream& os, const Structure* s, const CoordSet* cs,
    std::map<const Atom*, int>& rev_asn, bool selected_only, bool displayed_only, double* xform,
    bool pqr, std::set<const Atom*>& written, std::map<const Residue*, int>& polymer_map)
{
    Residue* prev_res = nullptr;
    bool prev_standard = false;
    PDB p, p_ter;
    bool need_ter = false;
    bool some_output = false;
    int serial = 0;
    for (auto r: s->residues()) {
        bool standard = Sequence::rname3to1(r->name()) != 'X';
        if (prev_res != nullptr && (prev_standard || standard) && some_output) {
            // if the preceding residue isn't in the same polymer, output TER
            int prev_pnum = polymer_map[prev_res];
            int pnum = polymer_map[r];
            if (prev_pnum != pnum)
                need_ter = true;
        }
        if (need_ter) {
            p_ter.set_type(PDB::TER);
            p_ter.ter.serial = ++serial;
            strcpy(p_ter.ter.res.name, prev_res->name().c_str());
            p_ter.ter.res.chain_id = prev_res->chain_id()[0];
            int seq_num = prev_res->number();
            char i_code = prev_res->insertion_code();
            if (seq_num > 9999) {
                // usurp the insertion code...
                i_code = '0' + (seq_num % 10);
                seq_num = seq_num / 10;
            }
            p_ter.ter.res.seq_num = seq_num;
            p_ter.ter.res.i_code = i_code;
        }

        // Shared attributes between Atom and Atomqr need to be set via proper pointer
        int *rec_serial;
        char (*rec_name)[5];
        char *rec_alt_loc;
        PDB::Residue *res;
        Real (*xyz)[3];
        if (pqr) {
            p.set_type(PDB::ATOMQR);
            rec_serial = &p.atomqr.serial;
            rec_name = &p.atomqr.name;
            rec_alt_loc = &p.atomqr.alt_loc;
            res = &p.atomqr.res;
            xyz = &p.atomqr.xyz;
        } else {
            if (standard && !r->is_het()) {
                p.set_type(PDB::ATOM);
            } else {
                p.set_type(PDB::HETATM);
            }
            rec_serial = &p.atom.serial;
            rec_name = &p.atom.name;
            rec_alt_loc = &p.atom.alt_loc;
            res = &p.atom.res;
            xyz = &p.atom.xyz;
        }

        // PDB spec no longer specifies atom ordering;
        // sort by coord_index to try to preserve input ordering...
        auto ordering = r->atoms();
        std::sort(ordering.begin(), ordering.end(), less_Atom());

        for (auto a: ordering) {
            if (selected_only && !a->selected())
                continue;
            if (displayed_only && !a->display())
                continue;
            std::string aname = s->asterisks_translated ?
                primes_to_asterisks(a->name()) : a->name().c_str();
            if (strlen(a->element().name()) > 1) {
                strcpy(*rec_name, aname.c_str());
            } else {
                bool element_compares;
                if (strncmp(a->element().name(), a->name().c_str(), 1) == 0) {
                    element_compares = true;
                } else if (a->element().number() == 1) {
                    char h = a->name().c_str()[0];
                    element_compares = (h == 'D' || h == 'T');
                } else {
                    element_compares = false;
                }
                if (element_compares && aname.size() < 4) {
                    strcpy(*rec_name, " ");
                    strcat(*rec_name, aname.c_str());
                } else {
                    strcpy(*rec_name, aname.c_str());
                }
            }
            strcpy(res->name, r->name().c_str());
            res->chain_id = r->chain_id()[0];
            auto seq_num = r->number();
            auto i_code = r->insertion_code();
            if (seq_num > 9999) {
                // usurp the insertion code...
                i_code = '0' + (seq_num % 10);
                seq_num = seq_num / 10;
            }
            res->seq_num = seq_num;
            res->i_code = i_code;
            if (pqr) {
                try {
throw atomstruct::PyAttrError("avoid calling through Python");
                    p.atomqr.charge = a->get_py_float_attr(pqr_charge);
                } catch (atomstruct::PyAttrError&) {
                    p.atomqr.charge = 0.0;
                }
                p.atomqr.radius = a->radius();
            } else {
                try {
throw atomstruct::PyAttrError("avoid calling through Python");
                    auto charge = a->get_py_int_attr(pdb_charge);
                    if (charge > 0.0)
                        p.atom.charge[0] = '+';
                    else if (charge < 0.0)
                        p.atom.charge[0] = '-';
                    else
                        p.atom.charge[0] = ' ';
                    p.atom.charge[1] = '0' + std::abs(charge);
                    p.atom.charge[2] = '\0';
                } catch (atomstruct::PyAttrError&) {
                    p.atom.charge[0] = ' ';
                    p.atom.charge[1] = ' ';
                    p.atom.charge[2] = '\0';
                }
                try {
throw atomstruct::PyAttrError("avoid calling through Python");
                    strcpy(p.atom.seg_id, a->get_py_string_attr(pdb_charge));
                } catch (atomstruct::PyAttrError&) { }
                const char* ename = a->element().name();
                if (a->element().number() == 1) {
                    if (a->name().c_str()[0] == 'D')
                        ename = "D";
                    else if (a->name().c_str()[0] == 'T') {
                        ename = "T";
                    }
                }
                strcpy(p.atom.element, ename);
            }
            if (need_ter) {
                os << p_ter << "\n";
                need_ter = false;
                some_output = false;
            }
            // loop through alt locs
            auto alt_locs = a->alt_locs();
            if (alt_locs.empty())
                alt_locs.insert(' ');
            for (auto alt_loc: alt_locs) {
                *rec_alt_loc = alt_loc;
                *rec_serial = ++serial;
                rev_asn[a] = *rec_serial;
                const Coord* crd;
                float bfactor, occupancy;
                if (alt_loc == ' ') {
                    // no alt locs
                    crd = &a->coord(cs);
                    bfactor = cs->get_bfactor(a);
                    occupancy = cs->get_occupancy(a);
                } else {
                    crd = &a->coord(alt_loc);
                    bfactor = a->bfactor();
                    occupancy = a->occupancy();
                }
                if (!pqr) {
                    p.atom.temp_factor = bfactor;
                    p.atom.occupancy = occupancy;
                }
                Coord final_crd;
                if (xform != nullptr) {
                    double mat[3][3], offset[3];
                    double *off = offset;
                    auto xf_vals = xform;
                    for (int row = 0; row < 3; ++row) {
                        mat[row][0] = *xf_vals++;
                        mat[row][1] = *xf_vals++;
                        mat[row][2] = *xf_vals++;
                        *off++ = *xf_vals++;
                    }
                    final_crd[0] = mat[0][0] * (*crd)[0] + mat[0][1] * (*crd)[1]
                        + mat[0][2] * (*crd)[2] + offset[0];
                    final_crd[1] = mat[1][0] * (*crd)[0] + mat[1][1] * (*crd)[1]
                        + mat[1][2] * (*crd)[2] + offset[1];
                    final_crd[2] = mat[2][0] * (*crd)[0] + mat[2][1] * (*crd)[1]
                        + mat[2][2] * (*crd)[2] + offset[2];
                } else
                    final_crd = *crd;
                (*xyz)[0] = final_crd[0];
                (*xyz)[1] = final_crd[1];
                (*xyz)[2] = final_crd[2];
                os << p << "\n";
                some_output = true;
            }
            written.insert(a);
        }
        prev_res = r;
        prev_standard = standard;
    }
}

static bool
chief_or_link(const Atom* a)
{
    auto r = a->residue();
    auto tr = tmpl::find_template_residue(r->name(), false, false);
    if (tr == nullptr)
        return false;
    return a->name() == tr->chief()->name() || a->name() == tr->link()->name();
}

static void
write_conect(std::ostream& os, const Structure* s, std::map<const Atom*, int>& rev_asn,
    const std::set<const Atom*>& written)
{
    PDB p;
    // to handle circular/cross-linked structures, make a map from residue to residue index...
    std::map<const Residue*, long> res_order;
    long i = 0;
    for (auto r: s->residues())
        res_order[r] = i++;

    // collate the metal coordination bonds for easy access
    std::map<const Atom*, std::vector<const Atom*>> coordinations;
    auto& mgr = s->pb_mgr();
    auto grp = mgr.get_group(Structure::PBG_METAL_COORDINATION);
    if (grp != nullptr) {
        for (auto pb: grp->pseudobonds()) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            coordinations[a1].push_back(a2);
            coordinations[a2].push_back(a1);
        }
    }

    for (auto r: s->residues()) {
        bool standard = atomstruct::standard_residue(r->name());
        // verify that the "standard" residue in fact has standard connectivity...
        if (standard) {
            auto index = res_order[r];
            bool start = index == 0 || !r->connects_to(s->residues()[index-1]);
            bool end = static_cast<Structure::Residues::size_type>(index) == s->residues().size()-1
                || !r->connects_to(s->residues()[index+1]);
            auto tr = tmpl::find_template_residue(r->name(), start, end);
            if (tr) {
                // Gather the intra-residue heavy-atom-only bonds...
                std::set<std::pair<AtomName,AtomName>> res_bonds;
                for (auto a: r->atoms()) {
                    if (a->element().number() == 1)
                        continue;
                    for (auto b: a->bonds()) {
                        auto oa = b->other_atom(a);
                        if (oa->element().number() == 1)
                            continue;
                        if (oa->residue() != r)
                            continue;
                        if (a->name() < oa->name())
                            res_bonds.insert(std::make_pair(a->name(), oa->name()));
                        else
                            res_bonds.insert(std::make_pair(oa->name(), a->name()));
                    }
                }

                // Gather the template heavy-atom-only bonds...
                std::set<std::pair<AtomName,AtomName>> tr_bonds;
                for (auto nm_a: tr->atoms_map()) {
                    auto a = nm_a.second;
                    if (a->element().number() == 1)
                        continue;
                    for (auto b: a->bonds()) {
                        auto oa = b->other_atom(a);
                        if (oa->element().number() == 1)
                            continue;
                        if (a->name() < oa->name())
                            tr_bonds.insert(std::make_pair(a->name(), oa->name()));
                        else
                            tr_bonds.insert(std::make_pair(oa->name(), a->name()));
                    }
                }
                standard = tr_bonds == res_bonds;
            }
        }
        for (auto a: r->atoms()) {
            if (written.find(a) == written.end())
                continue;
            bool skip_conect = standard && coordinations.find(a) == coordinations.end();
            int count = 0;
            p.set_type(PDB::CONECT);
            p.conect.serial[0] = rev_asn[a];
            for (auto b: a->bonds()) {
                auto oa = b->other_atom(a);
                if (written.find(oa) == written.end())
                    continue;
                auto oar = oa->residue();
                if (skip_conect && oar != r) {
                    if (!atomstruct::standard_residue(oar->name())
                    || !chief_or_link(a)
                    || oar->chain_id() != r->chain_id()
                    || std::abs(res_order[r] - res_order[oar]) > 1)
                        skip_conect = false;
                }
                if (count == 4) {
                    if (!skip_conect)
                        os << p << "\n";
                    count = 0;
                    p.set_type(PDB::CONECT);
                    p.conect.serial[0] = rev_asn[a];
                }
                int index = 1 + count++;
                p.conect.serial[index] = rev_asn[oa];
            }
            for (auto oa: coordinations[a]) {
                if (written.find(oa) == written.end())
                    continue;
                if (count == 4) {
                    os << p << "\n";
                    count = 0;
                    p.set_type(PDB::CONECT);
                    p.conect.serial[0] = rev_asn[a];
                }
                int index = 1 + count++;
                p.conect.serial[index] = rev_asn[oa];
            }
            if (!skip_conect && count != 0) {
                os << p << "\n";
            }
        }
    }
}

static void
write_pdb(std::vector<const Structure*> structures, std::ostream& os, bool selected_only,
    bool displayed_only, std::vector<double*>& xforms, bool all_frames, bool pqr)
{
    PDB p;
    // non-selected/displayed atoms may not be written out, so we need to track what
    // was written so we know which CONECT records to output
    std::set<const Atom*> written;
    int out_model_num = 0;
    for (std::vector<const Structure*>::size_type i = 0; i < structures.size(); ++i) {
        auto s = structures[i];
        auto xform = xforms[i];
        bool multi_model = (s->coord_sets().size() > 1) && all_frames;
        // Output headers only before first MODEL
        if (s == structures[0]) {
            // write out known headers first
            auto& headers = s->metadata;
            for (auto& record_type: record_order) {
                if (record_type == "MODEL")
                    // end of headers
                    break;
                auto hdr_i = headers.find(record_type);
                if (hdr_i == headers.end())
                    continue;
                for (auto hdr: hdr_i->second) {
                    os << hdr << '\n';
                }
            }

            // write out unknown headers
            decltype(record_order) known_headers(record_order.begin(),
                std::find(record_order.begin(), record_order.end(), "MODEL"));
            for (auto& type_records: headers) {
                if (!std::isupper(type_records.first[0]) || type_records.first.size() > 6)
                    // not a PDB header
                    continue;
                if (std::find(known_headers.begin(), known_headers.end(), type_records.first)
                != known_headers.end())
                    continue;
                for (auto& record: type_records.second)
                    os << record << '\n';
            }
        }

        std::map<const Atom*, int> rev_asn;
        std::map<const Residue*, int> polymer_map;
        int polymer_num = 1;
        for (auto poly_residues: s->polymers()) {
            for (auto r: poly_residues)
                polymer_map[r] = polymer_num;
            polymer_num++;
        }
        for (auto cs: s->coord_sets()) {
            if (!multi_model && cs != s->active_coord_set())
                continue;
            bool use_MODEL = multi_model || structures.size() > 1;
            if (use_MODEL) {
                p.set_type(PDB::MODEL);
                if (multi_model)
                    p.model.serial = cs->id();
                else
                    p.model.serial = ++out_model_num;
                os << p << "\n";
            }
            write_coord_set(os, s, cs, rev_asn, selected_only, displayed_only, xform, pqr, written,
                polymer_map);
            if (use_MODEL) {
                p.set_type(PDB::ENDMDL);
                os << p << "\n";
            }
        }
        write_conect(os, s, rev_asn, written);
        p.set_type(PDB::END);
        os << p << "\n";
    }
}

static const char*
docstr_read_pdb_file = 
"read_pdb_file(f, log=None, explode=True)\n" \
"\n" \
"f\n" \
"  A file-like object open for reading containing the PDB info\n" \
"log\n" \
"  A file-like object open for writing that warnings/errors and other\n" \
"  information will be written to\n" \
"explode\n" \
"  Controls whether NMR ensembles will be handled as separate models (True)\n" \
"  or as one model with multiple coordinate sets (False)\n" \
"\n" \
"Returns a numpy array of C++ pointers to AtomicStructure objects.";

extern "C" PyObject *
read_pdb_file(PyObject *, PyObject *args, PyObject *keywords)
{
    PyObject *pdb_file;
    PyObject *py_logger = Py_None;
    int explode = 1;
    static const char *kw_list[] = {"file", "log", "explode", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|$Op",
            (char **) kw_list, &pdb_file, &py_logger, &explode))
        return nullptr;
    return read_pdb(pdb_file, py_logger, explode);
}

static const char*
docstr_write_pdb_file = 
"write_pdb_file(structures, file_name, selected_only=False, displayed_only=False, xforms=None\n" \
"    all_frames=True, pqr=False)\n" \
"\n" \
"structures\n" \
"  A sequence of C++ structure pointers\n" \
"file_name\n" \
"  The output file path\n" \
"selected_only\n" \
"  If True, only selected atoms will be written\n" \
"displayed_only\n" \
"  If True, only displayed atoms will be written\n" \
"xforms\n" \
"  A sequence of 3x4 numpy arrays to transform the atom coordinates of the corresponding\n" \
"  structure.  If None then untransformed coordinates will be used for all structures.\n" \
"  Similarly, any None in the sequence will cause untransformed coordinates to be used\n" \
"  for that structure.\n" \
"all_frames\n" \
"  If True, all frames of a trajectory will be written (as multiple MODELS).\n" \
"  Otherwise, just the current frame will be written.\n" \
"pqr\n" \
"  If True, write PQR-style ATOM records\n" \
"\n";

extern "C" PyObject*
write_pdb_file(PyObject *, PyObject *args, PyObject *keywords)
{
    PyObject *py_structures;
    PyObject *py_path;
    int selected_only = (int)false;
    int displayed_only = (int)false;
    PyObject* py_xforms = Py_None;
    int all_frames = (int)true;
    int pqr = (int)false;
    static const char *kw_list[] = {
        "structures", "file_name", "selected_only", "displayed_only", "xforms", "all_frames",
        "pqr", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OO&|$ppOpp",
            (char **) kw_list, &py_structures, PyUnicode_FSConverter, &py_path, &selected_only,
            &displayed_only, &py_xforms, &all_frames, &pqr))
        return nullptr;

    if (!PySequence_Check(py_structures)) {
        PyErr_SetString(PyExc_TypeError, "First arg is not a sequence (of structure pointers)");
        return nullptr;
    }
    auto num_structs = PySequence_Size(py_structures);
    if (num_structs == 0) {
        PyErr_SetString(PyExc_ValueError, "First arg (sequence of structure pointers) is empty");
        return nullptr;
    }
    std::vector<const Structure*> structures;
    for (decltype(num_structs) i = 0; i < num_structs; ++i) {
        PyObject* py_ptr = PySequence_GetItem(py_structures, i);
        if (!PyLong_Check(py_ptr)) {
            std::stringstream err_msg;
            err_msg << "Item at index " << i << " of first arg is not an int (structure pointer)";
            PyErr_SetString(PyExc_TypeError, err_msg.str().c_str());
            return nullptr;
        }
        structures.push_back(static_cast<const Structure*>(PyLong_AsVoidPtr(py_ptr)));
    }

    const char* path = PyBytes_AS_STRING(py_path);
    auto os = std::ofstream(path);
    if (!os.good()) {
        std::stringstream err_msg;
        err_msg << "Unable to open file '" << path << "' for writing";
        PyErr_SetString(PyExc_IOError, err_msg.str().c_str());
        Py_XDECREF(py_path);
        return nullptr;
    }

    std::vector<double*> xforms;
    auto array = Numeric_Array();
    if (py_xforms == Py_None) {
        for (int i = 0; i < num_structs; ++i)
            xforms.push_back(nullptr);
    } else {
        if (PySequence_Check(py_xforms) < 0) {
            PyErr_SetString(PyExc_TypeError, "xforms arg is not a sequence");
            Py_XDECREF(py_path);
            return nullptr;
        }
        if (PySequence_Size(py_xforms) != num_structs) {
            PyErr_SetString(PyExc_TypeError,
                "xforms arg sequence is not the same length as the number of structures");
            Py_XDECREF(py_path);
            return nullptr;
        }
        for (int i = 0; i < num_structs; ++i) {
            PyObject* py_xform = PySequence_GetItem(py_xforms, i);
            if (py_xform == Py_None) {
                xforms.push_back(nullptr);
                continue;
            }
            if (!array_from_python(py_xform, 2, Numeric_Array::Double, &array, false)) {
                Py_XDECREF(py_path);
                return nullptr;
            }
            auto dims = array.sizes();
            if (dims[0] != 3 || dims[1] != 4) {
                std::stringstream err_msg;
                err_msg << "Transform #" << i+1 << " is not 3x4, is " << dims[0] << "x" << dims[1];
                PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
                Py_XDECREF(py_path);
                return nullptr;
            }
            xforms.push_back(static_cast<double*>(array.values()));
        }
    }
    write_pdb(structures, os, (bool)selected_only, (bool)displayed_only, xforms, (bool)all_frames,
        (bool)pqr);

    if (os.bad()) {
        PyErr_SetString(PyExc_ValueError, "Problem writing output PDB file");
        Py_XDECREF(py_path);
        return nullptr;
    }
    Py_XDECREF(py_path);
    Py_RETURN_NONE;
}

static struct PyMethodDef pdbio_functions[] =
{
    { "read_pdb_file", (PyCFunction)read_pdb_file, METH_VARARGS|METH_KEYWORDS, 
        docstr_read_pdb_file },
    { "write_pdb_file", (PyCFunction)write_pdb_file, METH_VARARGS|METH_KEYWORDS,
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

PyMODINIT_FUNC PyInit_pdbio()
{
    return PyModule_Create(&pdbio_def);
}

}  // namespace pdb
