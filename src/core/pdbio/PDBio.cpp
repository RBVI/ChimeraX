// vim: set expandtab ts=4 sw=4:
#include <algorithm>  // for std::sort
#include <set>
#include <sstream>
#include <stdio.h>  // fgets
#include <unordered_map>

#include "PDBio.h"
#include "pdb/PDB.h"
#include "atomstruct/AtomicStructure.h"
#include "atomstruct/Residue.h"
#include "atomstruct/Bond.h"
#include "atomstruct/Atom.h"
#include "atomstruct/connect.h"
#include "atomstruct/CoordSet.h"
#include "atomstruct/Sequence.h"
#include "blob/StructBlob.h"
#include "cpp_logger/logger.h"

namespace pdb {

using atomstruct::AtomicStructure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using atomstruct::Element;
using atomstruct::MolResId;
using atomstruct::Sequence;
using basegeom::Coord;
	
std::string pdb_segment("pdb_segment");
std::string pdb_charge("formal_charge");
std::string pqr_charge("charge");
std::string pqr_radius("radius");

static void
canonicalize_atom_name(std::string *aname, bool *asterisks_translated)
{
    for (int i = aname->length(); i > 0; ) {
        --i;
        // strip embedded blanks
        if ((*aname)[i] == ' ') {
            aname->replace(i, 1, "");
            continue;
        }
        // use prime instead of asterisk
        if ((*aname)[i] == '*') {
            (*aname)[i] = '\'';
            *asterisks_translated = true;
        }
    }
}

static void
canonicalize_res_name(std::string *rname)
{
    for (int i = rname->length(); i > 0; ) {
        --i;
        if ((*rname)[i] == ' ') {
            rname->replace(i, 1, "");
            continue;
        }
        (*rname)[i] = toupper((*rname)[i]);
    }
}

#ifdef CLOCK_PROFILING
#include <ctime>
static clock_t cum_preloop_t, cum_loop_preswitch_t, cum_loop_switch_t, cum_loop_postswitch_t, cum_postloop_t;
#endif
// return NULL on error
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
    Residue        *cur_residue = NULL;
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
                return NULL;
            }
            logger::warning(py_logger, "Ignored bad PDB record found on line ",
                *line_num, '\n', is.str());
            break;

        case PDB::HEADER:
            // SCOP doesn't provide MODRES records for HETATMs...
            if (strstr(record.header.classification, "SCOP/ASTRAL") != NULL) {
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
                cur_residue = as->residues()[0].get();
            else {
                cur_residue = NULL;
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
                if (prev_cs != NULL && acs->coords().size() < prev_cs->coords().size())
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

            std::string aname, rname;
            char cid = record.atom.res.chain_id;
            if (islower(cid))
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
            canonicalize_res_name(&rname);
            if (recent_TER && cur_residue != NULL && cur_residue->chain_id() == rid.chain)
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
                            cur_residue = as->residues()[++cur_res_index].get();
                    } else {
                        // Monte-Carlo traj?
                        std::string string_cid;
                        string_cid += cid;
                        cur_residue = as->find_residue(string_cid, seq_num, i_code);
                        if (cur_residue == NULL) {
                            // if chain ID is space and res is het,
                            // then chain ID probably should be
                            // space, check that...
                            string_cid = " ";
                            cur_residue = as->find_residue(string_cid, seq_num, i_code);
                            if (cur_residue != NULL)
                                rid = MolResId(' ', seq_num, i_code);
                        }
                    }
                }
                if (cur_residue == NULL || MolResId(cur_residue) != rid 
                || cur_residue->name() != rname) {
                    logger::error(py_logger, "Residue ", rid, " not in first"
                        " model on line ", *line_num, " of PDB file");
                    goto finished;
                }
            } else if (cur_residue == NULL || cur_rid != rid
            // modifying HETs can be inline...
            || (cur_residue->name() != rname && (record.type() != PDB::HETATM
                || cur_residue->is_het())))
            {
                // on to new residue

                if (cur_residue != NULL && cur_rid.chain != rid.chain) {
                    start_connect = true;
                } else if (record.type() == PDB::HETATM
                && (break_hets || (!is_SCOP
                && mod_res->find(rid) == mod_res->end()))) {
                    start_connect = true;
                } else if (cur_residue != NULL && cur_residue->position() > rid.pos
                && cur_residue->find_atom("OXT") !=  NULL) {
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
                if (!start_connect && cur_residue != NULL
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

                if (start_connect && cur_residue != NULL)
                    end_residues->push_back(cur_residue);
                cur_rid = rid;
                cur_residue = as->new_residue(rname, rid.chain, rid.pos, rid.insert);
                if (record.type() == PDB::HETATM)
                    cur_residue->set_is_het(true);
                cur_res_index = as->residues().size() - 1;
                if (start_connect)
                    start_residues->push_back(cur_residue);
                start_connect = false;
            }
            aname = record.atom.name;
            canonicalize_atom_name(&aname, &as->asterisks_translated);
            Coord c(record.atom.xyz);
            if (in_model > 1) {
                Atom *a = cur_residue->find_atom(aname);
                if (a == NULL) {
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
            Element *e;
            if (!is_babel) {
                if (record.atom.element[0] != '\0')
                    e = new Element(record.atom.element);
                else {
                    if (strlen(record.atom.name) == 4
                    && record.atom.name[0] == 'H')
                        e = new Element(1);
                    else
                        e = new Element(record.atom.name);
                }

                if ((e->number() > 83 || e->number() == 61
                  || e->number() == 43 || e->number() == 0)
                  && record.atom.name[0] != ' ') {
                    // probably one of those funky PDB
                    // non-standard-residue atom names;
                    // try _just_ the second character...
                    delete e;
                    char atsym[2];
                    atsym[0] = record.atom.name[1];
                    atsym[1] = '\0';
                    e = new Element(atsym);
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
                    delete e;
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
                e = new Element(babel_name);
                
            }
            Atom *a;
            if (record.atom.alt_loc && cur_residue->count_atom(aname) == 1) {
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
                        a->register_field(pqr_radius, record.atomqr.radius);
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
            delete e;
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
                logger::error(py_logger, "Unknown atom serial number (",
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
            std::string string_chain_id;
            string_chain_id += record.ssbond.res[0].chain_id;
            Residue *ssres = as->find_residue(string_chain_id,
                record.ssbond.res[0].seq_num, record.ssbond.res[0].i_code);
            if (ssres == NULL)
                break;
            if (ssres->name() != record.ssbond.res[0].name) {
                logger::warning(py_logger, "First res name in SSBOND record (",
                    record.ssbond.res[0].name, ") does not match actual"
                    " residue (", ssres->name(), "); skipping.");
                break;
            }
            Atom *ap0 = ssres->find_atom("SG");
            if (ap0 == NULL) {
                logger::warning(py_logger, "Atom SG not found in ", ssres);
                break;
            }

            string_chain_id = "";
            string_chain_id += record.ssbond.res[1].chain_id;
            ssres = as->find_residue(string_chain_id,
                record.ssbond.res[1].seq_num, record.ssbond.res[1].i_code);
            if (ssres == NULL)
                break;
            if (ssres->name() != record.ssbond.res[1].name) {
                logger::warning(py_logger, "Second res name in SSBOND record (",
                    record.ssbond.res[1].name, ") does not match actual"
                    " residue (", ssres->name(), "); skipping.");
                break;
            }
            Atom *ap1 = ssres->find_atom("SG");
            if (ap1 == NULL) {
                logger::warning(py_logger, "Atom SG not found in ", ssres);
                break;
            }
            if (!ap0->connects_to(ap1))
                (void) ap0->structure()->new_bond(ap0, ap1);
            break;
        }

        case PDB::SEQRES: {
            std::string chain_id(1, record.seqres.chain_id);
            for (int i = 0; i < record.seqres.num_res; ++i) {
                std::string res_name(record.seqres.res_name[i]);
                as->extend_input_seq_info(chain_id, res_name);
            }
            if (as->input_seq_source.empty())
                as->input_seq_source = "PDB SEQRES record";
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
                
                std::vector<std::string> &h = as->pdb_headers[key];
                h.push_back(std::string((const char *)line));
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
    if (cur_residue != NULL) {
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
            Element e1(test_name);
            if (e1.number() != 0) {
                a->_switch_initial_element(e1);
                continue;
            }
            if (a->name().size() < 2)
                continue;
            test_name[1] = a->name()[1];
            Element e2(test_name);
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
    add_bond(atom_serial_nums[from], atom_serial_nums[to]);
}

// assign_secondary_structure:
//    Assign secondary structure state to residues using PDB
//    HELIX and SHEET records
void
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
        std::string string_chain_id = "";
        string_chain_id += init->chain_id;
        std::string string_name = init->name;
        Residue *init_res = as->find_residue(string_chain_id, init->seq_num,
            init->i_code, string_name);
        if (init_res == NULL) {
            logger::error(py_logger, "Start residue of secondary structure"
                " not found: ", r.c_str());
            continue;
        }
        string_chain_id = "";
        string_chain_id += end->chain_id;
        string_name = end->name;
        Residue *end_res = as->find_residue(string_chain_id, end->seq_num,
            end->i_code, string_name);
        if (end_res == NULL) {
            logger::error(py_logger, "End residue of secondary structure"
                " not found: ", r.c_str());
            continue;
        }
        AtomicStructure::Residues::const_iterator first = as->residues().end();
        AtomicStructure::Residues::const_iterator last = as->residues().end();
        for (AtomicStructure::Residues::const_iterator
        ri = as->residues().begin(); ri != as->residues().end(); ++ri) {
            Residue *r = (*ri).get();
            if (r == init_res)
                first = ri;
            if (r == end_res) {
                last = ri;
                break;
            }
        }
        if (first == as->residues().end()
        || last == as->residues().end()) {
            logger::error(py_logger, "Bad residue range for secondary"
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
            Residue *r = (*ri).get();
            r->set_ss_id(id);
            r->set_is_sheet(true);
            if (ri == sri->second)
                break;
        }
    }
}

// bonded_dist:
//    Are given atoms close enough to bond?  If so, return bond distance,
// otherwise return zero.
float
bonded_dist(Atom *a, Atom *b)
{
    float bond_len = Element::bond_length(a->element(), b->element());
    if (bond_len == 0.0)
        return 0.0;
    float max_bond_len_sq = bond_len + 0.4;
    max_bond_len_sq *= max_bond_len_sq;
    float dist_sq = a->coord().sqdistance(b->coord());
    if (dist_sq > max_bond_len_sq)
        return 0.0;
    return dist_sq;
}

// connect_atom_by_distance:
//    Connect an atom to a residue by distance criteria.  Don't connect a
// hydrogen or lone pair more than once, nor connect to one that's already
// bonded.
static void
connect_atom_by_distance(Atom *a, const Residue::Atoms &atoms,
    Residue::Atoms::const_iterator &a_it, std::set<Atom *> *conect_atoms)
{
    float short_dist = 0.0;
    Atom *close_atom = NULL;

    bool H_or_LP = a->element() <= Element::H;
    if (H_or_LP && !a->bonds().empty())
        return;
    Residue::Atoms::const_iterator end = atoms.end();
    for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != end; ++ai)
    {
        Atom *oa = *ai;
        if (a == oa || a->connects_to(oa)
        || (oa->element() <= Element::H && (H_or_LP || !oa->bonds().empty())))
            continue;
        if (ai < a_it && conect_atoms && conect_atoms->find(oa) == conect_atoms->end())
            // already checked
            continue;
        float dist = bonded_dist(a, oa);
        if (dist == 0.0)
            continue;
        if (H_or_LP) {
            if (short_dist != 0.0 && dist > short_dist)
                continue;
            short_dist = dist;
            close_atom = oa;
        } else
            (void) a->structure()->new_bond(a, oa);
    }
    if (H_or_LP && short_dist != 0) {
        (void) a->structure()->new_bond(a, close_atom);
    }
}

void prune_short_bonds(AtomicStructure *as)
{
    std::vector<Bond *> short_bonds;

    const AtomicStructure::Bonds &bonds = as->bonds();
    for (AtomicStructure::Bonds::const_iterator bi = bonds.begin(); bi != bonds.end(); ++bi) {
        Bond *b = (*bi).get();
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
    std::string aname, rname;
    PDB::Residue res = link.res[0];
    std::string cid;
    cid += res.chain_id;
    rname = res.name;
    canonicalize_res_name(&rname);
    Residue *res1 = as->find_residue(cid, res.seq_num, res.i_code, rname);
    if (!res1) {
        logger::warning(py_logger, "Cannot find LINK residue ", res.name,
            " (", res.seq_num, res.i_code, ")");
        return;
    }
    res = link.res[1];
    cid = "";
    cid += res.chain_id;
    rname = res.name;
    canonicalize_res_name(&rname);
    Residue *res2 = as->find_residue(cid, res.seq_num, res.i_code, rname);
    if (!res2) {
        logger::warning(py_logger, "Cannot find LINK residue ", res.name,
            " (", res.seq_num, res.i_code, ")");
        return;
    }
    aname = link.name[0];
    canonicalize_atom_name(&aname, &as->asterisks_translated);
    Atom *a1 = res1->find_atom(aname);
    if (a1 == NULL) {
        logger::error(py_logger, "Cannot find LINK atom ", aname,
            " in residue ", res1->str());
        return;
    }
    aname = link.name[1];
    canonicalize_atom_name(&aname, &as->asterisks_translated);
    Atom *a2 = res2->find_atom(aname);
    if (a2 == NULL) {
        logger::error(py_logger, "Cannot find LINK atom ", aname,
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
    if (fgets(read_fileno_buffer, 1024, (FILE *)f) == NULL)
        read_fileno_buffer[0] = '\0';
    return std::pair<char *, PyObject *>(read_fileno_buffer, NULL);
}

PyObject *
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
    std::string as_name("unknown PDB file");
#ifdef CLOCK_PROFILING
clock_t start_t, end_t;
#endif
    PyObject *http_mod = PyImport_ImportModule("http.client");
    if (http_mod == NULL)
        return NULL;
    PyObject *http_conn = PyObject_GetAttrString(http_mod, "HTTPResponse");
    if (http_conn == NULL) {
        Py_DECREF(http_mod);
        PyErr_SetString(PyExc_AttributeError,
            "HTTPResponse class not found in http.client module");
        return NULL;
    }
    int is_inst = PyObject_IsInstance(pdb_file, http_conn);
    int fd;
    if (is_inst)
        // due to buffering issues, cannot handle a socket like it 
        // was a file
        fd = -1;
    else
        fd = PyObject_AsFileDescriptor(pdb_file);
    if (fd == -1) {
        read_func = read_no_fileno;
        input = pdb_file;
        PyErr_Clear();
        PyObject *io_mod = PyImport_ImportModule("io");
        if (io_mod == NULL)
            return NULL;
        PyObject *io_base = PyObject_GetAttrString(io_mod, "IOBase");
        if (io_base == NULL) {
            Py_DECREF(io_mod);
            PyErr_SetString(PyExc_AttributeError, "IOBase class not found in io module");
            return NULL;
        }
        int is_inst = PyObject_IsInstance(pdb_file, io_base);
        if (is_inst == 0)
            PyErr_SetString(PyExc_TypeError, "PDB file is not an instance of IOBase class");
        if (is_inst <= 0) {
            Py_DECREF(io_mod);
            Py_DECREF(io_base);
            return NULL;
        }
    } else {
        read_func = read_fileno;
        input = fdopen(fd, "r");
        // try to get file name
        PyObject* name_attr = PyObject_GetAttrString(pdb_file, "name");
        if (name_attr != nullptr) {
            as_name = PyUnicode_AsUTF8(name_attr);
        }
    }
    while (true) {
#ifdef CLOCK_PROFILING
start_t = clock();
#endif
        AtomicStructure *as = new AtomicStructure(py_logger);
        as->set_name(as_name);
        void *ret = read_one_structure(read_func, input, as, &line_num, asn_map[as],
          &start_res_map[as], &end_res_map[as], &ss_map[as], &conect_map[as],
          &link_map[as], &mod_res_map[as], &reached_end, py_logger, explode, &eof);
        if (ret == NULL) {
            for (std::vector<AtomicStructure *>::iterator si = structs->begin();
            si != structs->end(); ++si) {
                delete *si;
            }
            delete as;
            return NULL;
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
            as = NULL;
        } else {
            // give all members of an ensemble the same pdb_headers
            if (explode && ! structs->empty()) {
                if (as->pdb_headers.empty())
                    as->pdb_headers = (*structs)[0]->pdb_headers;
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
    // ensure structaccess module objects are initialized
    PyObject* structaccess_mod = PyImport_ImportModule("structaccess");
    if (structaccess_mod == NULL) {
        delete structs;
        return NULL;
    }
    using blob::StructBlob;
    StructBlob* sb = static_cast<StructBlob*>(blob::newBlob<StructBlob>(&blob::StructBlob_type));
    for (auto si = structs->begin(); si != structs->end(); ++si) {
        sb->_items->emplace_back(*si);
    }
    delete structs;
    return sb;
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
"Returns a structaccess.StructBlob.";

extern "C" PyObject *
read_pdb_file(PyObject *, PyObject *args, PyObject *keywords)
{
    PyObject *pdb_file, *mols;
    PyObject *py_logger = Py_None;
    bool explode = true;
    static const char *kw_list[] = {"file", "log", "explode", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|$Op", (char **) kw_list,
        &pdb_file, &py_logger, &explode))
            return NULL;
    mols = read_pdb(pdb_file, py_logger, explode);
    return mols;
}

static struct PyMethodDef pdbio_functions[] =
{
    { "read_pdb_file", (PyCFunction)read_pdb_file, METH_VARARGS|METH_KEYWORDS,
        docstr_read_pdb_file },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef pdbio_def =
{
    PyModuleDef_HEAD_INIT,
    "pdbio",
    "Input/output for PDB files",
    -1,
    pdbio_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_pdbio()
{
    return PyModule_Create(&pdbio_def);
}

}  // namespace pdb
