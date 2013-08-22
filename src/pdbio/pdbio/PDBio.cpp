#include "PDBio.h"
#include "pdb/PDB.h"
#include "molecule/Molecule.h"
#include "molecule/Residue.h"
#include "molecule/Bond.h"
#include "molecule/Atom.h"
#include "molecule/CoordSet.h"
#include "molecule/templates/TmplResidue.h"
#include "molecule/templates/TmplAtom.h"
#include "molecule/templates/residues.h"
#include "capsule/capsule.h"
#include <set>
#include <sstream>
#include <algorithm>  // for std::sort
#include <stdio.h>  // fgets

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

std::string pdb_segment("pdb_segment");
std::string pdb_charge("formal_charge");
std::string pqr_charge("charge");
std::string pqr_radius("radius");

// standard_residues contains the names of residues that should have PDB ATOM records.
static std::set<std::string, std::less<std::string> >	standard_residues;

static void
init_standard_residues()
{
	standard_residues.insert("A");
	standard_residues.insert("ALA");
	standard_residues.insert("ARG");
	standard_residues.insert("ASN");
	standard_residues.insert("ASP");
	standard_residues.insert("ASX");
	standard_residues.insert("C");
	standard_residues.insert("CYS");
	standard_residues.insert("DA");
	standard_residues.insert("DC");
	standard_residues.insert("DG");
	standard_residues.insert("DT");
	standard_residues.insert("G");
	standard_residues.insert("GLN");
	standard_residues.insert("GLU");
	standard_residues.insert("GLX");
	standard_residues.insert("GLY");
	standard_residues.insert("HIS");
	standard_residues.insert("I");
	standard_residues.insert("ILE");
	standard_residues.insert("LEU");
	standard_residues.insert("LYS");
	standard_residues.insert("MET");
	standard_residues.insert("PHE");
	standard_residues.insert("PRO");
	standard_residues.insert("SER");
	standard_residues.insert("T");
	standard_residues.insert("THR");
	standard_residues.insert("TRP");
	standard_residues.insert("TYR");
	standard_residues.insert("U");
	standard_residues.insert("VAL");
}

//TODO: these 3 funcs need to be wrapped also
bool
standard_residue(const std::string &name)
{
	if (standard_residues.empty())
		init_standard_residues();
	return standard_residues.find(name) != standard_residues.end();
}

void
add_standard_residue(const std::string &name)
{
	if (standard_residues.empty())
		init_standard_residues();
	standard_residues.insert(name);
}

void
remove_standard_residue(const std::string &name)
{
	if (standard_residues.empty())
		init_standard_residues();
	standard_residues.erase(name);
}

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

class MolResId {
	// convenience class for testing chain/position/insert-code equality
public:
	int	pos;
	std::string chain;
	char insert;
	MolResId() {};
	MolResId(char c, int p, char ic) {
		chain = c;
		pos = p;
		insert = ic;
	};
	MolResId(const Residue *r) {
		chain = r->chain_id();
		pos = r->position();
		insert = r->insertion_code();
	};
	bool operator==(const MolResId &rid) const {
		return rid.pos == pos && rid.chain == chain && rid.insert == insert;
	}
	bool operator!=(const MolResId &rid) const {
		return rid.pos != pos || rid.chain != chain || rid.insert != insert;
	}
	bool operator<(const MolResId &rid) const {
		return chain < rid.chain || 
			(chain == rid.chain && (pos < rid.pos || 
			(pos == rid.pos && insert < rid.insert)));
	}
};

std::ostream & operator<<(std::ostream &os, const MolResId &rid) {
	os << rid.pos;
	if (rid.insert != ' ')
		os << rid.insert;
	if (rid.chain != " ")
		os << "." << rid.chain;
	return os;
}

#ifdef CLOCK_PROFILING
#include <ctime>
static clock_t cum_preloop_t, cum_loop_preswitch_t, cum_loop_switch_t, cum_loop_postswitch_t, cum_postloop_t;
#endif
// return NULL on error
// return input if PDB records implying a molecule encountered
// return PyNone otherwise (e.g. only blank lines, MASTER records, etc.)
static void *
read_one_molecule(std::pair<char *, PyObject *> (*read_func)(void *),
	void *input, Molecule *m,
	int *line_num, std::map<int, Atom *> &asn,
	std::vector<Residue *> *start_residues,
	std::vector<Residue *> *end_residues,
	std::vector<PDB> *secondary_structure,
	std::vector<PDB::Conect_> *conect_records,
	std::vector<PDB::Link_> *link_records,
	std::set<MolResId> *mod_res, bool *reached_end,
	PyObject *log_file, bool explode, bool *eof)
{
	bool		start_connect = true;
	int			in_model = 0;
	Molecule::Residues::size_type cur_res_index = 0;
	Residue		*cur_residue = NULL;
	MolResId	cur_rid;
	PDB			record;
	bool		actual_molecule = false;
	bool		in_headers = true;
	bool		is_SCOP = false;
	bool		is_babel = false; // have we seen Babel-style atom names?
	bool		recent_TER = false;
	bool		break_hets = false;
#ifdef CLOCK_PROFILING
clock_t	 start_t, end_t;
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

		default:	// ignore other record types
			break;

		case PDB::UNKNOWN:
			if (record.unknown.junk[0] & 0200) {
				LOG_PY_ERROR_NULL("Non-ASCII character on line " << *line_num
					<< " of PDB file\n");
				return NULL;
			}
			LOG_PY_ERROR_NULL("warning:  Ignored bad PDB record found on line "
					<< *line_num << '\n' << is.str() << "\n");
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
			break;

		case PDB::HELIX:
		case PDB::SHEET:
			if (secondary_structure)
				secondary_structure->push_back(record);
		case PDB::TURN:
			break;

	  	case PDB::MODEL: {
			cur_res_index = 0;
			if (in_model && !m->residues().empty())
				cur_residue = m->residues()[0];
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
				int cs_size = m->active_coord_set()->coords().size();
				if (!explode && csid > m->active_coord_set()->id() + 1) {
					// fill in coord sets for Monte-Carlo
					// trajectories
					const CoordSet *acs = m->active_coord_set();
					for (int fill_in_ID = acs->id()+1; fill_in_ID < csid; ++fill_in_ID) {
						CoordSet *cs = m->new_coord_set(fill_in_ID, cs_size);
						cs->fill(acs);
					}
				}
				CoordSet *cs = m->new_coord_set(csid, cs_size);
				m->set_active_coord_set(cs);
			} else {
				// first CoordSet starts empty
				CoordSet *cs = m->new_coord_set(csid);
				m->set_active_coord_set(cs);
			}
			break;
		}

		case PDB::ENDMDL:
			if (explode)
				goto finished;
			if (in_model > 1 && m->coord_sets().size() > 1) {
				// fill in coord set for Monte-Carlo
				// trajectories if necessary
				CoordSet *acs = m->active_coord_set();
				const CoordSet *prev_cs = m->find_coord_set(acs->id()-1);
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
			actual_molecule = true;

			std::string aname, rname;
			char cid = record.atom.res.chain_id;
			if (islower(cid))
				m->lower_case_chains = true;
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
						if (cur_res_index + 1 < m->residues().size())
							cur_residue = m->residues()[++cur_res_index];
					} else {
						// Monte-Carlo traj?
						std::string string_cid;
						string_cid += cid;
						cur_residue = m->find_residue(string_cid, seq_num, i_code);
						if (cur_residue == NULL) {
							// if chain ID is space and res is het,
							// then chain ID probably should be
							// space, check that...
							string_cid = " ";
							cur_residue = m->find_residue(string_cid, seq_num, i_code);
							if (cur_residue != NULL)
								rid = MolResId(' ', seq_num, i_code);
						}
					}
				}
				if (cur_residue == NULL || MolResId(cur_residue) != rid 
				|| cur_residue->name() != rname) {
					LOG_PY_ERROR_NULL("Residue " << rid << " not in first model"
						<< " on line " << *line_num << " of PDB file\n");
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
				cur_residue = m->new_residue(rname, rid.chain, rid.pos, rid.insert);
				if (record.type() == PDB::HETATM)
					cur_residue->set_is_het(true);
				cur_res_index = m->residues().size() - 1;
				if (start_connect)
					start_residues->push_back(cur_residue);
				start_connect = false;
			}
			aname = record.atom.name;
			canonicalize_atom_name(&aname, &m->asterisks_translated);
			Coord c(record.atom.xyz);
			if (in_model > 1) {
				Atom *a = cur_residue->find_atom(aname);
				if (a == NULL) {
					LOG_PY_ERROR_NULL("Atom " << aname << " not in first model on line "
						<< *line_num << " of PDB file\n");
					goto finished;
				}
				// ensure that the name uniquely identifies the atom;
				// if not, then use an index-based 'find'
				// (Monte Carlo trajectories had better use unique names!)
				if (cur_residue->count_atom(aname) > 1) {
					// no lookup from coord_index to atom, so search the Residue...
					unsigned int index = m->active_coord_set()->coords().size();
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
				a = m->new_atom(aname, *e);
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
			delete e;
			if (in_model == 0 && asn.find(record.atom.serial) != asn.end())
				LOG_PY_ERROR_NULL("warning:  duplicate atom serial number found: "
					<< record.atom.serial << '\n');
			asn[record.atom.serial] = a;
			break;
		}

		case PDB::ANISOU: {
			int serial = record.anisou.serial;
			std::map<int, Atom *>::const_iterator si = asn.find(serial);
			if (si == asn.end()) {
				LOG_PY_ERROR_NULL("Unknown atom serial number (" << serial
					<< ") in ANISOU record\n");
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
			Residue *ssres = m->find_residue(string_chain_id,
				record.ssbond.res[0].seq_num, record.ssbond.res[0].i_code);
			if (ssres == NULL)
				break;
			if (ssres->name() != record.ssbond.res[0].name) {
				LOG_PY_ERROR_NULL("warning: first res name in SSBOND record ("
					<< record.ssbond.res[0].name << ") does not match actual residue ("
					<< ssres->name() << "); skipping.\n");
				break;
			}
			Atom *ap0 = ssres->find_atom("SG");
			if (ap0 == NULL) {
				LOG_PY_ERROR_NULL("warning: Atom SG not found in " << ssres << "\n");
				break;
			}

			string_chain_id = "";
			string_chain_id += record.ssbond.res[1].chain_id;
			ssres = m->find_residue(string_chain_id,
				record.ssbond.res[1].seq_num, record.ssbond.res[1].i_code);
			if (ssres == NULL)
				break;
			if (ssres->name() != record.ssbond.res[1].name) {
				LOG_PY_ERROR_NULL("warning: second res name in SSBOND record ("
					<< record.ssbond.res[1].name << ") does not match actual residue ("
					<< ssres->name() << "); skipping.\n");
				break;
			}
			Atom *ap1 = ssres->find_atom("SG");
			if (ap1 == NULL) {
				LOG_PY_ERROR_NULL("warning: Atom SG not found in " << ssres << "\n");
				break;
			}
			if (!ap0->connects_to(ap1))
				(void) ap0->molecule()->new_bond(ap0, ap1);
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
				
				std::vector<std::string> &h = m->pdb_headers[key];
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
	m->pdb_version = record.pdb_input_version();

#ifdef CLOCK_PROFILING
cum_postloop_t += clock() - start_t;
#endif
	if (actual_molecule)
		return input;
	return Py_None;
}

inline static void
add_bond(Atom *a1, Atom *a2)
{
	if (!a1->connects_to(a2))
		(void) a1->molecule()->new_bond(a1, a2);
}

// add_bond:
//	Add a bond to molecule given two atom serial numbers.
//	(atom_serial_nums argument should be const, but operator[] isn't const)
static void
add_bond(std::map<int, Atom *> &atom_serial_nums, int from, int to, PyObject *log_file)
{
	if (to <= 0 || from <= 0)
		return;
	if (to == from) {
		LOG_PY_ERROR_VOID("warning: CONECT record from atom to itself: " << from << "\n");
		return;
	}
	// serial "from" check happens before this routine is called
	if (atom_serial_nums.find(to) == atom_serial_nums.end()) {
		LOG_PY_ERROR_VOID("warning:  CONECT record to nonexistent atom: ("
				<< from << ", " << to << ")\n");
		return;
	}
	add_bond(atom_serial_nums[from], atom_serial_nums[to]);
}

// assign_secondary_structure:
//	Assign secondary structure state to residues using PDB
//	HELIX and SHEET records
void
assign_secondary_structure(Molecule *m, const std::vector<PDB> &ss, PyObject *log_file)
{
	std::vector<std::pair<Molecule::Residues::const_iterator,
		Molecule::Residues::const_iterator> > strand_ranges;
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
		Residue *init_res = m->find_residue(string_chain_id, init->seq_num,
			init->i_code, string_name);
		if (init_res == NULL) {
			LOG_PY_ERROR_VOID("Start residue of secondary structure not found: "
				<< r.c_str() << '\n');
			continue;
		}
		string_chain_id = "";
		string_chain_id += end->chain_id;
		string_name = end->name;
		Residue *end_res = m->find_residue(string_chain_id, end->seq_num,
			end->i_code, string_name);
		if (end_res == NULL) {
			LOG_PY_ERROR_VOID("End residue of secondary structure not found: "
				<< r.c_str() << '\n');
			continue;
		}
		Molecule::Residues::const_iterator first = m->residues().end();
		Molecule::Residues::const_iterator last = m->residues().end();
		for (Molecule::Residues::const_iterator
		ri = m->residues().begin(); ri != m->residues().end(); ++ri) {
			Residue *r = *ri;
			if (r == init_res)
				first = ri;
			if (r == end_res) {
				last = ri;
				break;
			}
		}
		if (first == m->residues().end()
		|| last == m->residues().end()) {
			LOG_PY_ERROR_VOID("Bad residue range for secondary structure: "
				<< r.c_str() << '\n');
			continue;
		}
		if (r.type() == PDB::SHEET)
			strand_ranges.push_back(std::pair<Molecule::Residues::const_iterator,
				Molecule::Residues::const_iterator>(first, last));
		else  {
			for (Molecule::Residues::const_iterator ri = first;
			ri != m->residues().end(); ++ri) {
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
	for (std::vector<std::pair<Molecule::Residues::const_iterator, Molecule::Residues::const_iterator> >::iterator sri = strand_ranges.begin(); sri != strand_ranges.end(); ++sri) {
		char chain_id = (*sri->first)->chain_id()[0];
		if (chain_id != last_chain) {
			id = 0;
			last_chain = chain_id;
		}
		++id;
		for (Molecule::Residues::const_iterator ri = sri->first;
		ri != m->residues().end(); ++ri) {
			Residue *r = *ri;
			r->set_ss_id(id);
			r->set_is_sheet(true);
			if (ri == sri->second)
				break;
		}
	}
}

// bonded_dist:
//	Are given atoms close enough to bond?  If so, return bond distance,
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
//	Connect an atom to a residue by distance criteria.  Don't connect a
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
			(void) a->molecule()->new_bond(a, oa);
	}
	if (H_or_LP && short_dist != 0) {
		(void) a->molecule()->new_bond(a, close_atom);
	}
}

// connect_residue_by_distance:
//	Connect atoms in residue by distance.  This is an n-squared algorithm.
//	Takes into account alternate atom locations.  'conect_atoms' are
//	atoms whose connectivity is already known.
void
connect_residue_by_distance(Residue *r, std::set<Atom *> *conect_atoms)
{
	// connect up atoms in residue by distance
	const Residue::Atoms &atoms = r->atoms();
	for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != atoms.end(); ++ai) {
		Atom *a = *ai;
		if (conect_atoms && conect_atoms->find(a) != conect_atoms->end()) {
			// connectivity specified in a CONECT record, skip
			continue;
		}
		connect_atom_by_distance(a, atoms, ai, conect_atoms);
	}
}

// connect_residue_by_template:
//	Connect bonds in residue according to the given template.  Takes into
//	acount alternate atom locations.
static void
connect_residue_by_template(Residue *r, const TmplResidue *tr,
						std::set<Atom *> *conect_atoms)
{
	// foreach atom in residue
	//	connect up like atom in template
	bool some_connectivity_unknown = false;
	std::set<Atom *> known_connectivity;
	const Residue::Atoms &atoms = r->atoms();
	for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != atoms.end(); ++ai) {
		Atom *a = *ai;
		if (conect_atoms->find(a) != conect_atoms->end()) {
			// connectivity specified in a CONECT record, skip
			known_connectivity.insert(a);
			continue;
		}
		TmplAtom *ta = tr->find_atom(a->name());
		if (ta == NULL) {
			some_connectivity_unknown = true;
			continue;
	 	}
		// non-template atoms will be able to connect to known atoms;
		// avoid rechecking known atoms though...
		known_connectivity.insert(a);

		for (TmplAtom::BondsMap::const_iterator bi = ta->bonds_map().begin();
		bi != ta->bonds_map().end(); ++bi) {
			Atom *b = r->find_atom(bi->first->name());
			if (b == NULL)
				continue;
			if (!a->connects_to(b))
				(void) a->molecule()->new_bond(a, b);
		}
	}
	// For each atom that wasn't connected (i.e. not in template),
	// connect it by distance
	if (!some_connectivity_unknown)
		return;
	connect_residue_by_distance(r, &known_connectivity);
}

// find_closest:
//	Find closest heavy atom to given heavy atom with residue that has
//	the same alternate location identifier (or none) and optionally return
static Atom *
find_closest(Atom *a, Residue *r, float *ret_dist_sq)
{
	if (a == NULL)
		return NULL;
	if (a->element().number() == 1)
		return NULL;
	const Residue::Atoms &r_atoms = r->atoms();
	Residue::Atoms::const_iterator ai = r_atoms.begin();
	if (ai == r_atoms.end())
		return NULL;
	Atom *closest = NULL;
	float dist_sq = 0.0;
	const Coord &c = a->coord();
	for (; ai != r_atoms.end(); ++ai) {
		Atom *oa = *ai;
		if (oa->element().number() == 1)
			continue;
		if ((a->residue() == r && a->name() == oa->name()))
			continue;
		const Coord &c1 = oa->coord();
		float new_dist_sq = c.sqdistance(c1);
		if (closest != NULL && new_dist_sq >= dist_sq)
			continue;
		dist_sq = new_dist_sq;
		closest = oa;
	}
	if (ret_dist_sq)
		*ret_dist_sq = dist_sq;
	return closest;
}

// add_bond_nearest_pair:
//	Add a bond between two residues.
static void
add_bond_nearest_pair(Residue *from, Residue *to, bool any_length=true)
{
	Atom	*fsave = NULL, *tsave = NULL;
	float	dist_sq = 0.0;

	const Residue::Atoms &atoms = from->atoms();
	for (Residue::Atoms::const_iterator ai = atoms.begin(); ai != atoms.end(); ++ai) {
		float	new_dist_sq;

		Atom *a = *ai;
		Atom *b = find_closest(a, to, &new_dist_sq);
		if (b == NULL)
			continue;
		if (fsave == NULL || new_dist_sq < dist_sq) {
			fsave = a;
			tsave = b;
			dist_sq = new_dist_sq;
		}
	}
	if (fsave != NULL) {
		if (!any_length && bonded_dist(fsave, tsave) == 0.0)
			return;
		add_bond(fsave, tsave);
	}
}

static bool
hookup(Atom *a, Residue *res, bool definitely_connect=true)
{
	bool made_connection = false;
	Atom *b = find_closest(a, res, NULL);
	if (!definitely_connect && b->coord().sqdistance(a->coord()) > 9.0)
		return false;
	if (b != NULL) {
		add_bond(a, b);
		made_connection = true;
	}
	return made_connection;
}

// connect_molecule:
//	Connect atoms in molecule by template if one is found, or by distance.
//	Adjacent residues are connected if appropriate.
void
connect_molecule(Molecule *m, std::vector<Residue *> *start_residues,
	std::vector<Residue *> *end_residues, std::set<Atom *> *conect_atoms,
	std::set<MolResId> *mod_res)
{
	// walk the residues, connecting residues as appropriate and
	// connect the atoms within the residue
	Residue *link_res = NULL, *prev_res = NULL, *first_res = NULL;
	Atom *link_atom;
	std::string link_atom_name("");
	for (Molecule::Residues::const_iterator ri = m->residues().begin();
	ri != m->residues().end(); ++ri) {
		Residue *r = *ri;

		if (!first_res)
			first_res = r;
		const TmplResidue *tr;
		if (mod_res->find(MolResId(r)) != mod_res->end())
			// residue in MODRES record;
			// don't try to use template connectivity
			tr = NULL;
		else
			tr = find_template_residue(r->name(),
				std::find(start_residues->begin(),
				start_residues->end(), r) != start_residues->end(),
				std::find(end_residues->begin(),
				end_residues->end(), r) != end_residues->end());
		if (tr != NULL)
			connect_residue_by_template(r, tr, conect_atoms);
		else
			connect_residue_by_distance(r, conect_atoms);

		// connect up previous residue
		if (link_res != NULL) {
			if (tr == NULL || tr->chief() == NULL) {
				add_bond_nearest_pair(link_res, r);
			} else {
				bool made_connection = false;
				// don't definitely connect a leading HET residue
				bool definitely_connect = (link_res != first_res
					|| link_atom_name != "");
				Atom *chief = r->find_atom(tr->chief()->name());
				if (chief != NULL) {
					if (link_atom != NULL) {
						add_bond(link_atom, chief);
						made_connection = true;
					} else {
						made_connection = hookup(chief, link_res, definitely_connect);
					}
				}
				if (!made_connection && definitely_connect)
					add_bond_nearest_pair(link_res, r);
			}
		} else if (r->atoms().size() > 1 && prev_res != NULL
				&& prev_res->chain_id() == r->chain_id()
				&& r->is_het() && conect_atoms->find(
				(*r->atoms().begin())) == conect_atoms->end()) {
			// multi-atom HET residues with no CONECTs (i.e. _not_
			// a standard PDB entry) _may_ connect to previous residue...
			add_bond_nearest_pair(prev_res, r, false);
		}

		prev_res = r;
		if (std::find(end_residues->begin(), end_residues->end(), r)
		!= end_residues->end()) {
			link_res = NULL;
		} else {
			link_res = r;
			if (tr == NULL || tr->link() == NULL) {
				link_atom_name = "";
				link_atom = NULL;
			} else {
				link_atom_name = tr->link()->name();
				link_atom = r->find_atom(link_atom_name);
			}
		}
	}

	// if no CONECT/MODRES records and there are non-standard residues not
	// in HETATM records (i.e. this is clearly a non-standard PDB
	// like those output by CCP4's refmac), then examine the inter-
	// residue bonds and break the non-physical ones (> 1.5 normal length)
	// involving at least one non-standard residue
	bool break_long = false;
	if (conect_atoms->empty() && mod_res->empty()) {
		for (Molecule::Residues::const_iterator ri=m->residues().begin()
		; ri != m->residues().end(); ++ri) {
			Residue *r = *ri;
			if (standard_residue(r->name()) || r->name() == "UNK")
				continue;
			if (!r->is_het()) {
				break_long = true;
				break;
			}
		}
	}
	if (break_long) {
		std::vector<Bond *> break_these;
		for (Molecule::Bonds::const_iterator bi = m->bonds().begin();
		bi != m->bonds().end(); ++bi) {
			Bond *b = *bi;
			const Bond::Atoms & atoms = b->atoms();
			Residue *r1 = atoms[0]->residue();
			Residue *r2 = atoms[1]->residue();
			if (r1 == r2)
				continue;
			if (standard_residue(r1->name()) && standard_residue(r2->name()))
				continue;
			// break if non-physical
			float criteria = 1.5 * Element::bond_length(atoms[0]->element(),
				atoms[1]->element());
			if (criteria * criteria < b->sqlength())
				break_these.push_back(b);
		}
		for (std::vector<Bond *>::iterator bi = break_these.begin();
		bi != break_these.end(); ++bi) {
			Bond *b = *bi;
			m->delete_bond(b);
		}
	}
}

void prune_short_bonds(Molecule *m)
{
	std::vector<Bond *> short_bonds;

	const Molecule::Bonds &bonds = m->bonds();
	for (Molecule::Bonds::const_iterator bi = bonds.begin(); bi != bonds.end(); ++bi) {
		Bond *b = *bi;
		Coord c1 = b->atoms()[0]->coord();
		Coord c2 = b->atoms()[1]->coord();
		if (c1.sqdistance(c2) < 0.001)
			short_bonds.push_back(b);
	}

	for (std::vector<Bond *>::iterator sbi = short_bonds.begin();
			sbi != short_bonds.end(); ++sbi) {
		m->delete_bond(*sbi);
	}
}

static void
link_up(PDB::Link_ &link, Molecule *m, std::set<Atom *> *conect_atoms,
						PyObject *log_file)
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
	Residue *res1 = m->find_residue(cid, res.seq_num, res.i_code, rname);
	if (!res1) {
		LOG_PY_ERROR_VOID("warning: cannot find LINK residue " << res.name << " ("
			<< res.seq_num << res.i_code << ")\n");
		return;
	}
	res = link.res[1];
	cid = "";
	cid += res.chain_id;
	rname = res.name;
	canonicalize_res_name(&rname);
	Residue *res2 = m->find_residue(cid, res.seq_num, res.i_code, rname);
	if (!res2) {
		LOG_PY_ERROR_VOID("warning: cannot find LINK residue " << res.name << " ("
			<< res.seq_num << res.i_code << ")\n");
		return;
	}
	aname = link.name[0];
	canonicalize_atom_name(&aname, &m->asterisks_translated);
	Atom *a1 = res1->find_atom(aname);
	if (a1 == NULL) {
		LOG_PY_ERROR_VOID("error: cannot find LINK atom " << aname << " in residue " << res1->str() << "\n");
		return;
	}
	aname = link.name[1];
	canonicalize_atom_name(&aname, &m->asterisks_translated);
	Atom *a2 = res1->find_atom(aname);
	if (a2 == NULL) {
		LOG_PY_ERROR_VOID("error: cannot find LINK atom " << aname << " in residue " << res2->str() << "\n");
		return;
	}
	if (!a1->connects_to(a2)) {
		m->new_bond(a1, a2);
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
read_pdb(PyObject *pdb_file, PyObject *log_file, bool explode)
{
	std::vector<Molecule *> file_mols;
	bool reached_end;
	std::map<Molecule *, std::vector<Residue *> > start_res_map, end_res_map;
	std::map<Molecule *, std::vector<PDB> > ss_map;
	typedef std::vector<PDB::Conect_> Conects;
	typedef std::map<Molecule *, Conects> ConectMap;
	ConectMap conect_map;
	typedef std::vector<PDB::Link_> Links;
	typedef std::map<Molecule *, Links> LinkMap;
	LinkMap link_map;
	std::map<Molecule *, std::set<MolResId> > mod_res_map;
	// Atom Serial Numbers -> Atom*
	typedef std::map<int, Atom *, std::less<int> > Asns;
	std::map<Molecule *, Asns > asn_map;
	bool per_model_conects = false;
	int line_num = 0;
	bool eof;
	std::pair<char *, PyObject *> (*read_func)(void *);
	void *input;
	std::vector<Molecule *> *mols = new std::vector<Molecule *>();
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
	} else {
		read_func = read_fileno;
		input = fdopen(fd, "r");
	}
	while (true) {
#ifdef CLOCK_PROFILING
start_t = clock();
#endif
		Molecule *m = new Molecule;
		void *ret = read_one_molecule(read_func, input, m, &line_num, asn_map[m],
		  &start_res_map[m], &end_res_map[m], &ss_map[m], &conect_map[m],
		  &link_map[m], &mod_res_map[m], &reached_end, log_file, explode, &eof);
		if (ret == NULL) {
			for (std::vector<Molecule *>::iterator mi = mols->begin();
			mi != mols->end(); ++mi) {
				delete *mi;
			}
			delete m;
			return NULL;
		}
#ifdef CLOCK_PROFILING
end_t = clock();
std::cerr << "read pdb: " << ((float)(end_t - start_t))/CLOCKS_PER_SEC << "\n";
start_t = end_t;
#endif
		if (ret == Py_None) {
			if (!file_mols.empty()) {
				// NMR ensembles can have trailing CONECT
				// records; integrate them before deleting 
				// the null molecule
				if (per_model_conects)
					conect_map[file_mols.back()] = conect_map[m];
				else {
					Conects &conects = conect_map[m];
					for (Conects::iterator ci = conects.begin();
							ci != conects.end(); ++ci) {
						PDB::Conect_ &conect = *ci;
						int serial = conect.serial[0];
						bool matched = false;
						for (ConectMap::iterator cmi = conect_map.begin();
						cmi != conect_map.end(); ++cmi) {
							Molecule *cm = (*cmi).first;
							Conects &cm_conects = (*cmi).second;
							Asns &asns = asn_map[cm];
							if (asns.find(serial) != asns.end()) {
								cm_conects.push_back(conect);
								matched = true;
							}
						}
						if (!matched) {
							LOG_PY_ERROR_NULL("warning: CONECT record for nonexistent atom: "
								<< serial << '\n');
						}
					}
				}
			}
			delete m;
			m = NULL;
		} else {
			// give all members of an ensemble the same pdb_headers
			if (explode && ! mols->empty()) {
				if (m->pdb_headers.empty())
					m->pdb_headers = (*mols)[0]->pdb_headers;
				if (ss_map[m].empty())
					ss_map[m] = ss_map[(*mols)[0]];
			}
			if (per_model_conects || (!file_mols.empty() && !conect_map[m].empty())) {
				per_model_conects = true;
				conect_map[file_mols.back()] = conect_map[m];
				conect_map[m].clear();
			}
			mols->push_back(m);
			file_mols.push_back(m);
		}
#ifdef CLOCK_PROFILING
end_t = clock();
std::cerr << "assign CONECTs: " << ((float)(end_t - start_t))/CLOCKS_PER_SEC << "\n";
start_t = end_t;
#endif

		if (!reached_end)
			continue;

		per_model_conects = false;
		for (std::vector<Molecule *>::iterator fmi = file_mols.begin();
		fmi != file_mols.end(); ++fmi) {
			Molecule *fm = *fmi;
			Conects &conects = conect_map[fm];
			Asns &asns = asn_map[fm];
			std::set<Atom *> conect_atoms;
			for (Conects::iterator ci = conects.begin(); ci != conects.end(); ++ci) {
				PDB::Conect_ &conect = *ci;
				int from_serial = conect.serial[0];
				if (asns.find(from_serial) == asns.end()) {
					LOG_PY_ERROR_NULL("warning:  CONECT record for nonexistent atom: "
						<< conect.serial[0] << '\n');
					break;
				}
				bool has_covalent = false;
				for (int i = 1; i < 5; i += 1) {
					add_bond(asns, from_serial, conect.serial[i], log_file);
				}
				// purely cross-residue bonds are not
				// considered to completely specify an
				// atom's connectivity unless it is
				// the only atom in the residue
				Atom *fa = asns[from_serial];
				const Atom::BondsMap &bonds_map = fa->bonds_map();
				for (Atom::BondsMap::const_iterator bi = bonds_map.begin();
				bi != bonds_map.end(); ++bi) {
					Atom *ta = (*bi).first;
					if (ta->residue() == fa->residue()) {
						has_covalent = true;
						break;
					}
				}
				if (has_covalent || fa->residue()->atoms().size() == 1) {
					conect_atoms.insert(fa);
				}
			}

			assign_secondary_structure(fm, ss_map[fm], log_file);

			Links &links = link_map[fm];
			for (Links::iterator li = links.begin(); li != links.end(); ++li)
				link_up(*li, fm, &conect_atoms, log_file);
			connect_molecule(fm, &start_res_map[fm], &end_res_map[fm], &conect_atoms, &mod_res_map[fm]);
			prune_short_bonds(fm);
		}
#ifdef CLOCK_PROFILING
end_t = clock();
std::cerr << "find bonds: " << ((float)(end_t - start_t))/CLOCKS_PER_SEC << "\n";
start_t = end_t;
#endif

		if (eof)
			break;
		file_mols.clear();
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
	return encapsulate_mol_vec(mols);
}

PyObject *
read_pdb_file(PyObject *, PyObject *args, PyObject *keywords)
{
	PyObject *pdb_file, *mols;
	PyObject *log_file = Py_None;
	bool explode = true;
	static const char *kw_list[] = {"file", "log", "explode", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|$Op", (char **) kw_list,
		&pdb_file, &log_file, &explode))
			return NULL;
#if 0
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
	if (log_file != Py_None) {
		is_inst = PyObject_IsInstance(log_file, io_base);
		if (is_inst == 0)
			PyErr_SetString(PyExc_TypeError, "log file is not an instance of IOBase class");
		if (is_inst <= 0) {
			Py_DECREF(io_mod);
			Py_DECREF(io_base);
			return NULL;
		}
	}
#endif
	mols = read_pdb(pdb_file, log_file, explode);
	return mols;
}

static struct PyMethodDef pdbio_functions[] =
{
	{ "read_pdb_file", (PyCFunction)read_pdb_file, METH_VARARGS|METH_KEYWORDS, "" },
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
