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
#include <atomstruct/CoordSet.h>
#include <atomstruct/PBGroup.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Sequence.h>
#include <atomstruct/destruct.h>
#include <atomstruct/tmpl/residues.h>
#include <logger/logger.h>
#include "pdb/connect.h"
#include <pdb/PDB.h>

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

using namespace pdb;
using namespace pdb_connect;

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

void set_res_name_and_chain_id(Residue* res, PDB::ResidueName& out_rn, char* out_cid,
    PyObject* py_logger = nullptr, bool *warned_res_name_length = nullptr,
    bool *warned_chain_id_length = nullptr)
{
    if (res->chain_id().size() == 2) {
        std::string adjusted_name;
        int num_spaces = 3 - res->name().size();
        if (num_spaces > 0)
            adjusted_name.insert(0, num_spaces, ' ');
        adjusted_name.append(res->name());
        adjusted_name.append(1, res->chain_id()[0]);
        strcpy(out_rn, adjusted_name.c_str());
        *out_cid = res->chain_id()[1];
    } else {
        strcpy(out_rn, res->name().c_str());
        *out_cid = res->chain_id()[0];
        if (py_logger != nullptr && res->chain_id().size() > 2 && !*warned_chain_id_length) {
            *warned_chain_id_length = true;
            logger::warning(py_logger, "Chain IDs longer than 2 characters; truncating");
        }
    }
    if (py_logger != nullptr && res->name().size() > 4 && !*warned_res_name_length) {
        *warned_res_name_length = true;
        logger::warning(py_logger, "Residue names longer than 4 characters; truncating");
    }
}

static void
push_helix(std::vector<Residue*>& cur_helix, std::vector<std::string>& helices, int helix_num)
{
    Residue* start = cur_helix[0];
    Residue* end = cur_helix[cur_helix.size()-1];

    PDB hrec(PDB::HELIX);

    hrec.helix.ser_num = helix_num;
    sprintf(hrec.helix.helix_id, "%3d", helix_num);
    set_res_name_and_chain_id(start, hrec.helix.init.name, &hrec.helix.init.chain_id);
    hrec.helix.init.seq_num = start->number();
    hrec.helix.init.i_code = start->insertion_code();
    set_res_name_and_chain_id(end, hrec.helix.end.name, &hrec.helix.end.chain_id);
    hrec.helix.end.seq_num = end->number();
    hrec.helix.end.i_code = end->insertion_code();
    hrec.helix.helix_class = 1;
    hrec.helix.length = cur_helix.size();
    helices.push_back(hrec.c_str());
    cur_helix.clear();
}

static void
push_sheet(std::vector<Residue*>& cur_sheet, std::vector<std::string>& sheets, int sheet_num)
{
    Residue* start = cur_sheet[0];
    Residue* end = cur_sheet[cur_sheet.size()-1];

    PDB srec(PDB::SHEET);

    srec.sheet.strand = sheet_num;
    sprintf(srec.sheet.sheet_id, "%3d", sheet_num);
    srec.sheet.num_strands = 1;
    set_res_name_and_chain_id(start, srec.sheet.init.name, &srec.sheet.init.chain_id);
    srec.sheet.init.seq_num = start->number();
    srec.sheet.init.i_code = start->insertion_code();
    set_res_name_and_chain_id(end, srec.sheet.end.name, &srec.sheet.end.chain_id);
    srec.sheet.end.seq_num = end->number();
    srec.sheet.end.i_code = end->insertion_code();
    srec.sheet.sense = 0;
    sheets.push_back(srec.c_str());
    cur_sheet.clear();
}

static void
compile_helices_sheets(const Structure* s, std::vector<std::string>& helices, std::vector<std::string>& sheets)
{
    Residue* prev_res = nullptr;
    int helix_num = 1, sheet_num = 1;
    std::vector<Residue*> cur_helix, cur_sheet;
    for (auto r: s->residues()) {
        if (prev_res && prev_res->connects_to(r, true)) {
            if (cur_helix.size() > 0 && (!r->is_helix() || prev_res->ss_id() != r->ss_id()))
                push_helix(cur_helix, helices, helix_num++);
            if (cur_sheet.size() > 0 && (!r->is_strand() || prev_res->ss_id() != r->ss_id()))
                push_sheet(cur_sheet, sheets, sheet_num++);
        } else {
            if (cur_helix.size() > 0)
                push_helix(cur_helix, helices, helix_num++);
            if (cur_sheet.size() > 0)
                push_sheet(cur_sheet, sheets, sheet_num++);
        }
        if (r->is_helix())
            cur_helix.push_back(r);
        if (r->is_strand())
            cur_sheet.push_back(r);

        prev_res = r;
    }
    if (cur_helix.size() > 0)
        push_helix(cur_helix, helices, helix_num);
    if (cur_sheet.size() > 0)
        push_sheet(cur_sheet, sheets, sheet_num);
}

static void
push_link(Atom *a1, Atom *a2, Real length, std::vector<std::string>& links)
{
    PDB lrec(PDB::LINK);

    strcpy(lrec.link.name[0], a1->name().c_str());
    strcpy(lrec.link.name[1], a2->name().c_str());
    lrec.link.alt_loc[0] = lrec.link.alt_loc[1] = ' ';
    if (a1->residue()->chain_id().size() < 2) {
        strncpy(lrec.link.res[0].name, a1->residue()->name().c_str(), 3);
        lrec.link.res[0].chain_id = a1->residue()->chain_id().c_str()[0];
    } else {
        auto res_name = a1->residue()->name();
        auto chain_id = a1->residue()->chain_id();
        res_name[3] = chain_id[0];
        strncpy(lrec.link.res[0].name, res_name.c_str(), 4);
        lrec.link.res[0].chain_id = chain_id[1];
    }
    lrec.link.res[0].seq_num = a1->residue()->number();
    lrec.link.res[0].i_code = a1->residue()->insertion_code();
    if (a2->residue()->chain_id().size() < 2) {
        strncpy(lrec.link.res[1].name, a2->residue()->name().c_str(), 3);
        lrec.link.res[1].chain_id = a2->residue()->chain_id().c_str()[0];
    } else {
        auto res_name = a2->residue()->name();
        auto chain_id = a2->residue()->chain_id();
        res_name[3] = chain_id[0];
        strncpy(lrec.link.res[1].name, res_name.c_str(), 4);
        lrec.link.res[1].chain_id = chain_id[1];
    }
    lrec.link.res[1].seq_num = a2->residue()->number();
    lrec.link.res[1].i_code = a2->residue()->insertion_code();
    lrec.link.sym[0] = lrec.link.sym[1] = 1555;
    lrec.link.length = length;
    links.push_back(lrec.c_str());
}

static void
compile_links_ssbonds(const Structure* s, std::vector<std::string>& links, std::vector<std::string>& ssbonds)
{
    // Preserve old LINK and SSBOND records that involved differing symmetry ops
    std::string Ssbond("SSBOND"), Link("LINK");
    int ssbond_serial = 1;
    auto ssbond_recs = s->metadata.find(Ssbond);
    if (ssbond_recs != s->metadata.end()) {
        for (auto rec: ssbond_recs->second) {
            if (rec.length() >= 72 && rec.substr(59, 6) != rec.substr(66, 6)) {
                char buffer[5];
                std::sprintf(buffer, "%4d", ssbond_serial++);
                ssbonds.push_back(rec.substr(0, 7) + std::string(buffer) + rec.substr(10, std::string::npos));
            }
        }
    }
    auto link_recs = s->metadata.find(Link);
    if (link_recs != s->metadata.end()) {
        for (auto rec: link_recs->second) {
            if (rec.length() >= 72 && rec.substr(59, 6) != rec.substr(66, 6))
                links.push_back(rec);
        }
    }

    // Go through inter-residue bonds; put non-polymeric ones into LINK or SSBOND
    for (auto b: s->bonds()) {
        auto a1 = b->atoms()[0];
        auto a2 = b->atoms()[1];
        auto r1 = a1->residue();
        auto r2 = a2->residue();
        if (r1 == r2 || b->polymeric_start_atom() != nullptr)
            continue;

        if (a1->element() == Element::S && a2->element() == Element::S) {
            // SSBOND
            PDB srec(PDB::SSBOND);
            srec.ssbond.ser_num = ssbond_serial++;
            if (r1->chain_id().size() < 2) {
                strncpy(srec.ssbond.res[0].name, r1->name().c_str(), 3);
                srec.ssbond.res[0].chain_id = r1->chain_id()[0];
            } else {
                auto res_name = r1->name();
                auto chain_id = r1->chain_id();
                res_name[3] = chain_id[0];
                strncpy(srec.ssbond.res[0].name, res_name.c_str(), 4);
                srec.ssbond.res[0].chain_id = chain_id[1];
            }
            srec.ssbond.res[0].seq_num = r1->number();
            srec.ssbond.res[0].i_code = r1->insertion_code();
            if (r2->chain_id().size() < 2) {
                strncpy(srec.ssbond.res[1].name, r2->name().c_str(), 3);
                srec.ssbond.res[1].chain_id = r2->chain_id()[0];
            } else {
                auto res_name = r2->name();
                auto chain_id = r2->chain_id();
                res_name[3] = chain_id[0];
                strncpy(srec.ssbond.res[1].name, res_name.c_str(), 4);
                srec.ssbond.res[1].chain_id = chain_id[1];
            }
            srec.ssbond.res[1].seq_num = r2->number();
            srec.ssbond.res[1].i_code = r2->insertion_code();
            srec.ssbond.sym[0] = srec.ssbond.sym[1] = 1555;
            srec.ssbond.length = b->length();
            ssbonds.push_back(srec.c_str());
        } else {
            // LINK
            push_link(a1, a2, b->length(), links);
        }
    }

    // Put metal complex pseudobonds into LINK records
    auto pbg = s->pb_mgr().get_group(Structure::PBG_METAL_COORDINATION);
    if (pbg != nullptr) {
        for (auto pb: pbg->pseudobonds()) {
            auto a1 = pb->atoms()[0];
            auto a2 = pb->atoms()[1];
            auto r1 = a1->residue();
            auto r2 = a2->residue();
            if (r1 == r2)
                continue;
            push_link(a1, a2, pb->length(), links);
        }
    }
}

typedef std::map<char, const char*> CharToResName;
static CharToResName protein_name_map = {
    {'A', "ALA"},
    {'C', "CYS"},
    {'D', "ASP"},
    {'E', "GLU"},
    {'F', "PHE"},
    {'G', "GLY"},
    {'H', "HIS"},
    {'I', "ILE"},
    {'K', "LYS"},
    {'L', "LEU"},
    {'M', "MET"},
    {'N', "ASN"},
    {'P', "PRO"},
    {'Q', "GLN"},
    {'R', "ARG"},
    {'S', "SER"},
    {'T', "THR"},
    {'V', "VAL"},
    {'W', "TRP"},
    {'Y', "TYR"}
};

static void
push_seqres(Chain *chain, size_t start_index, int record_num, std::vector<std::string>& seqres)
{
    PDB sr_rec(PDB::SEQRES);
    sr_rec.seqres.ser_num = record_num;
    std::string chain_id(chain->chain_id());
    if (chain_id.size() > 1) {
        sr_rec.seqres.chain_id[0] = chain_id[0];
        sr_rec.seqres.chain_id[1] = chain_id[1];
        sr_rec.seqres.chain_id[2] = '\0';
    } else {
        sr_rec.seqres.chain_id[0] = ' ';
        sr_rec.seqres.chain_id[1] = chain_id[0];
        sr_rec.seqres.chain_id[2] = '\0';
    }
    sr_rec.seqres.num_res = chain->residues().size();
    int is_rna = -1;
    for (size_t i = 0; i < 13; ++i) {
        size_t index = start_index + i;
        if (index >= chain->residues().size())
            break;
        auto res = chain->residues()[index];
        if (res == nullptr) {
            auto seq_char = chain->characters()[index];
            if (chain->polymer_type() == PT_AMINO) {
                auto char_rn_i = protein_name_map.find(seq_char);
                if (char_rn_i == protein_name_map.end()) {
                    strcpy(sr_rec.seqres.res_name[i], "UNK");
                } else {
                    strcpy(sr_rec.seqres.res_name[i], (*char_rn_i).second);
                }
            } else {
                if (is_rna == -1) {
                    // need to try do figure out if it's RNA or DNA: look through actual residues
                    is_rna = 1; // default to RNA if no residues exist
                    for (auto r: chain->residues()) {
                        if (r != nullptr) {
                            is_rna = r->name().size() == 1 ? 1 : 0;
                            break;
                        }
                    }
                }
                if (is_rna) {
                    ResName rna_name;
                    rna_name.append(1, seq_char);
                    strcpy(sr_rec.seqres.res_name[i], rna_name.c_str());
                } else {
                    ResName dna_name("D");
                    dna_name.append(1, seq_char);
                    strcpy(sr_rec.seqres.res_name[i], dna_name.c_str());
                }
            }
        } else {
            strcpy(sr_rec.seqres.res_name[i], res->name().c_str());
        }
    }
    seqres.push_back(sr_rec.c_str());
}

static void
compile_seqres(const Structure* s, std::vector<std::string>& seqres)
{
    for (auto chain: s->chains()) {
        int record_num = 1;
        for (size_t i = 0; i < chain->characters().size(); i += 13) {
                push_seqres(chain, i, record_num++, seqres);
        }
    }
}

class StringIOStream
{
    PyObject* _string_io_write;
    bool _good;
public:
    StringIOStream(PyObject* string_io): _good(true) {
        _string_io_write = PyObject_GetAttrString(string_io, "write");
        if (_string_io_write == nullptr)
            throw std::logic_error("StringIO object has no 'write' attribute");
    }
    virtual ~StringIOStream() { Py_DECREF(_string_io_write); }
    bool bad() const { return !_good; }
    bool good() const { return _good; }
    void write_char(char c) {
        char buf[2];
        buf[0] = c;
        buf[1] = '\0';
        write_text(buf);
    }
    void write_text(const char* text) {
        PyObject* py_text = PyUnicode_FromString(text);
        auto result = PyObject_CallFunctionObjArgs(_string_io_write, py_text, nullptr);
        if (result == nullptr)
            _good = false;
        else
            Py_DECREF(result);
    }
    StringIOStream& operator<<(const char* text) { write_text(text); return *this; }
    StringIOStream& operator<<(const PDB& p) { *this << p.c_str(); return *this; }
    StringIOStream& operator<<(const std::string& s) { *this << s.c_str(); return *this; }
    StringIOStream& operator<<(char c) { write_char(c); return *this; }
};

class StreamDispatcher
{
    bool _use_fstream;
    std::ofstream* _fstream;
    StringIOStream* _io_stream;
public:
    StreamDispatcher(std::ofstream* fstream) {
        _use_fstream = true;
        _fstream = fstream;
    }
    StreamDispatcher(StringIOStream* io_stream) {
        _use_fstream = false;
        _io_stream = io_stream;
    }
    ~StreamDispatcher() {
        if (_use_fstream)
            delete _fstream;
        else
            delete _io_stream;
    }
    bool bad() const { return _use_fstream ? _fstream->bad() : _io_stream->bad(); }
    bool good() const { return _use_fstream ? _fstream->good() : _io_stream->good(); }
    StreamDispatcher& operator<<(const char* text) {
        if (_use_fstream)
            *_fstream << text;
        else
            *_io_stream << text;
        return *this;
    }
    StreamDispatcher& operator<<(const PDB& p) {
        if (_use_fstream)
            *_fstream << p;
        else
            *_io_stream << p;
        return *this;
    }
    StreamDispatcher& operator<<(const std::string& s) {
        if (_use_fstream)
            *_fstream << s;
        else
            *_io_stream << s;
        return *this;
    }
    StreamDispatcher& operator<<(char c) {
        if (_use_fstream)
            *_fstream << c;
        else
            *_io_stream << c;
        return *this;
    }
};

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
            start_connect = true;
            recent_TER = true;
            break_hets = false;
            if (second_chain_let_okay)
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
            auto cid = segid_chains ? ChainID(record.atom.seg_id) : ChainID({record.atom.res.chain_id});
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
                    logger::error(py_logger, "Residue ", rid, " not in first"
                        " model on line ", *line_num, " of PDB file");
                    goto finished;
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

static int
aniso_u_to_int(Real aniso_u_val)
{
    return static_cast<int>(aniso_u_val < 0.0 ?
        10000.0 * aniso_u_val - 0.5 : 10000.0 * aniso_u_val + 0.5);
}

static void
write_coord_set(StreamDispatcher& os, const Structure* s, const CoordSet* cs,
    std::map<const Atom*, int>& rev_asn, bool selected_only, bool displayed_only, double* xform,
    bool pqr, bool h36, std::set<const Atom*>& written, std::map<const Residue*, int>& polymer_map,
    const std::set<ResName>& polymeric_res_names, PyObject* py_logger, bool *warned_atom_name_length,
    bool *warned_res_name_length, bool *warned_chain_id_length)
{
    Residue* prev_res = nullptr;
    bool prev_standard = false;
    PDB p(h36), p_ter(h36), p_anisou(h36);
    bool need_ter = false;
    bool some_output = false;
    int serial = 0;
    p_ter.set_type(PDB::TER);
    p_anisou.set_type(PDB::ANISOU);
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
            p_ter.ter.serial = ++serial;
            set_res_name_and_chain_id(prev_res, p_ter.ter.res.name, &p_ter.ter.res.chain_id);
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
            if (standard || polymeric_res_names.find(r->name()) != polymeric_res_names.end()) {
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
                primes_to_asterisks(a->name().c_str()) : a->name().c_str();
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
            if (aname.size() > 4 && !*warned_atom_name_length) {
                *warned_atom_name_length = true;
                logger::warning(py_logger, "Atom names longer than 4 characters; truncating");

            }
            set_res_name_and_chain_id(r, res->name, &res->chain_id,
                py_logger, warned_res_name_length, warned_chain_id_length);
            auto seq_num = r->number();
            auto i_code = r->insertion_code();
            // since H36 also works with the non-ATOM fields that require large residue numbers,
            // don't use the hack below anymore
#if 0
            if (seq_num > 9999) {
                // usurp the insertion code...
                i_code = '0' + (seq_num % 10);
                seq_num = seq_num / 10;
            }
#endif
            res->seq_num = seq_num;
            res->i_code = i_code;
            if (pqr) {
                try {
                    p.atomqr.charge = a->get_py_float_attr(pqr_charge);
                } catch (pyinstance::PyAttrError&) {
                    p.atomqr.charge = 0.0;
                }
                p.atomqr.radius = a->radius();
            } else {
                try {
                    auto charge = a->get_py_int_attr(pdb_charge);
                    if (charge > 0.0)
                        p.atom.charge[0] = '+';
                    else if (charge < 0.0)
                        p.atom.charge[0] = '-';
                    else
                        p.atom.charge[0] = ' ';
                    p.atom.charge[1] = '0' + std::abs(charge);
                    p.atom.charge[2] = '\0';
                } catch (pyinstance::PyAttrError&) {
                    p.atom.charge[0] = ' ';
                    p.atom.charge[1] = ' ';
                    p.atom.charge[2] = '\0';
                }
                try {
                    strcpy(p.atom.seg_id, a->get_py_string_attr(pdb_segment));
                } catch (pyinstance::PyAttrError&) { }
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
                    bfactor = a->bfactor(alt_loc);
                    occupancy = a->occupancy(alt_loc);
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
                if (a->has_aniso_u(alt_loc)) {
                    p_anisou.anisou.serial = *rec_serial;
                    strcpy(p_anisou.anisou.name, *rec_name);
                    p_anisou.anisou.alt_loc = *rec_alt_loc;
                    p_anisou.anisou.res = *res;
                    // Atom.aniso_u is row major; whereas PDB is 11, 22, 33, 12, 13, 23
                    auto aniso_u = a->aniso_u(alt_loc);
                    p_anisou.anisou.u[0] = aniso_u_to_int((*aniso_u)[0]);
                    p_anisou.anisou.u[1] = aniso_u_to_int((*aniso_u)[3]);
                    p_anisou.anisou.u[2] = aniso_u_to_int((*aniso_u)[5]);
                    p_anisou.anisou.u[3] = aniso_u_to_int((*aniso_u)[1]);
                    p_anisou.anisou.u[4] = aniso_u_to_int((*aniso_u)[2]);
                    p_anisou.anisou.u[5] = aniso_u_to_int((*aniso_u)[4]);
                    strcpy(p_anisou.anisou.seg_id, p.atom.seg_id);
                    strcpy(p_anisou.anisou.element, p.atom.element);
                    strcpy(p_anisou.anisou.charge, p.atom.charge);
                    os << p_anisou << "\n";
                }
            }
            written.insert(a);
        }
        prev_res = r;
        prev_standard = standard;
    }
    if (prev_res != nullptr && prev_standard && some_output) {
        // Output a final TER if the last residue was in a chain
        p_ter.ter.serial = ++serial;
        set_res_name_and_chain_id(prev_res, p_ter.ter.res.name, &p_ter.ter.res.chain_id);
        int seq_num = prev_res->number();
        char i_code = prev_res->insertion_code();
        if (seq_num > 9999) {
            // usurp the insertion code...
            i_code = '0' + (seq_num % 10);
            seq_num = seq_num / 10;
        }
        p_ter.ter.res.seq_num = seq_num;
        p_ter.ter.res.i_code = i_code;
        os << p_ter << "\n";
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
write_conect(StreamDispatcher& os, const Structure* s, std::map<const Atom*, int>& rev_asn,
    const std::set<const Atom*>& written, const std::set<ResName>& polymeric_res_names)
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
        bool standard = polymeric_res_names.find(r->name()) != polymeric_res_names.end();
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
                    if (polymeric_res_names.find(oar->name()) == polymeric_res_names.end()
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
write_pdb(std::vector<const Structure*> structures, StreamDispatcher& os, bool selected_only,
    bool displayed_only, std::vector<double*>& xforms, bool all_coordsets, bool pqr, bool h36,
    const std::set<ResName>& polymeric_res_names, PyObject* py_logger)
{
    PDB p(h36);
    // non-selected/displayed atoms may not be written out, so we need to track what
    // was written so we know which CONECT records to output
    std::set<const Atom*> written;
    int out_model_num = 0;
    std::string Helix("HELIX"), Sheet("SHEET"), Ssbond("SSBOND"), Link("LINK"), Seqres("SEQRES");
    bool warned_atom_name_length = false;
    bool warned_res_name_length = false;
    bool warned_chain_id_length = false;

    for (std::vector<const Structure*>::size_type i = 0; i < structures.size(); ++i) {
        auto s = structures[i];
        auto xform = xforms[i];
        bool multi_model = (s->coord_sets().size() > 1) && all_coordsets;
        // Output headers only before first MODEL
        if (s == structures[0]) {
            // generate HELIX/SHEET records relevant to current structure
            std::vector<std::string> helices, sheets, ssbonds, links, seqres;
            compile_helices_sheets(s, helices, sheets);
            compile_links_ssbonds(s, links, ssbonds);
            compile_seqres(s, seqres);
            // since we need to munge the headers, make a copy instead of using a const reference
            auto headers = s->metadata;
            headers[Helix] = helices;
            headers[Sheet] = sheets;
            headers[Ssbond] = ssbonds;
            headers[Link] = links;
            headers[Seqres] = seqres;
            // write out known headers first
            for (auto& record_type: record_order) {
                if (record_type == "MODEL")
                    // end of headers
                    break;
                auto hdr_i = headers.find(record_type);
                if (hdr_i == headers.end())
                    continue;
                for (auto hdr: hdr_i->second) {
                    os << hdr;
                    os << '\n';
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
        for (auto polymers: s->polymers()) {
            auto& poly_residues = polymers.first;
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
            write_coord_set(os, s, cs, rev_asn, selected_only, displayed_only, xform, pqr, h36,
                written, polymer_map, polymeric_res_names, py_logger, &warned_atom_name_length,
                &warned_res_name_length, &warned_chain_id_length);
            if (use_MODEL) {
                p.set_type(PDB::ENDMDL);
                os << p << "\n";
            }
        }
        write_conect(os, s, rev_asn, written, polymeric_res_names);
    }
    p.set_type(PDB::END);
    os << p << "\n";
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
" http://cci.lbl.gov/hybrid_36), so up to 87,440,031 atoms."
" 'polymeric_res_names' is a sequence of residue names that"
" should be output using ATOM records rather than HETATM records."
"\n";

extern "C" PyObject*
write_pdb_file(PyObject *, PyObject *args)
{
    PyObject* py_structures;
    PyObject* py_output;
    int selected_only;
    int displayed_only;
    PyObject* py_xforms;
    int all_coordsets;
    int pqr;
    int h36;
    PyObject* py_poly_res_names;
    PyObject* py_logger;
    if (!PyArg_ParseTuple(args, "OOppOpppOO",
            &py_structures, &py_output, &selected_only, &displayed_only, &py_xforms, &all_coordsets,
            &pqr, &h36, &py_poly_res_names, &py_logger))
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

    std::set<ResName> poly_res_names;
    if (!PySequence_Check(py_poly_res_names) && !PyAnySet_Check(py_poly_res_names)) {
        PyErr_SetString(PyExc_TypeError, "'polymeric_res_names' arg must be a sequence or a set");
        return nullptr;
    }
    PyObject *iter = PyObject_GetIter(py_poly_res_names);
    PyObject *py_res_name;
    while ((py_res_name = PyIter_Next(iter))) {
        if (!PyUnicode_Check(py_res_name)) {
            Py_DECREF(py_res_name);
            std::stringstream err_msg;
            err_msg << "Item in 'polymeric_res_names' arg is not a string";
            PyErr_SetString(PyExc_TypeError, err_msg.str().c_str());
            Py_DECREF(py_res_name);
            Py_DECREF(iter);
            return nullptr;
        }
        poly_res_names.insert(PyUnicode_AsUTF8(py_res_name));
        Py_DECREF(py_res_name);
    }
    Py_DECREF(iter);

    PyObject *io_mod = PyImport_ImportModule("io");
    if (io_mod == nullptr)
        return nullptr;
    PyObject *io_base = PyObject_GetAttrString(io_mod, "IOBase");
    if (io_base == nullptr) {
        Py_DECREF(io_mod);
        PyErr_SetString(PyExc_AttributeError, "IOBase class not found in io module");
        return nullptr;
    }
    int is_inst = PyObject_IsInstance(py_output, io_base);
    if (is_inst < 0) {
        Py_DECREF(io_mod);
        Py_DECREF(io_base);
        return nullptr;
    }
    StreamDispatcher* out_stream;
    if (is_inst) {
        out_stream = new StreamDispatcher(new StringIOStream(py_output));
    } else {
        PyBytesObject* fs_path;
        if (PyUnicode_FSConverter(py_output, &fs_path) < 0) {
            return nullptr;
        }
        const char* path = PyBytes_AS_STRING(fs_path);
#ifdef _WIN32
        auto wpath = PyUnicode_AsWideCharString(py_output, nullptr);
        if (wpath == nullptr) {
            std::stringstream err_msg;
            err_msg << "Unable to convert file name '" << path << "'to Windows format string";
            PyErr_SetString(PyExc_IOError, err_msg.str().c_str());
            return nullptr;
        }
        out_stream = new StreamDispatcher(new std::ofstream(wpath));
        PyMem_Free(wpath);
#else
        out_stream = new StreamDispatcher(new std::ofstream(path));
        Py_XDECREF(fs_path);
#endif
        if (!out_stream->good()) {
            std::stringstream err_msg;
            err_msg << "Unable to open file '" << path << "' for writing";
            PyErr_SetString(PyExc_IOError, err_msg.str().c_str());
            return nullptr;
        }
    }

    std::vector<double*> xforms;
    auto array = Numeric_Array();
    if (py_xforms == Py_None) {
        for (int i = 0; i < num_structs; ++i)
            xforms.push_back(nullptr);
    } else {
        if (PySequence_Check(py_xforms) < 0) {
            PyErr_SetString(PyExc_TypeError, "xforms arg is not a sequence");
            return nullptr;
        }
        if (PySequence_Size(py_xforms) != num_structs) {
            PyErr_SetString(PyExc_TypeError,
                "xforms arg sequence is not the same length as the number of structures");
            return nullptr;
        }
        for (int i = 0; i < num_structs; ++i) {
            PyObject* py_xform = PySequence_GetItem(py_xforms, i);
            if (py_xform == Py_None) {
                xforms.push_back(nullptr);
                continue;
            }
            if (!array_from_python(py_xform, 2, Numeric_Array::Double, &array, false)) {
                return nullptr;
            }
            auto dims = array.sizes();
            if (dims[0] != 3 || dims[1] != 4) {
                std::stringstream err_msg;
                err_msg << "Transform #" << i+1 << " is not 3x4, is " << dims[0] << "x" << dims[1];
                PyErr_SetString(PyExc_ValueError, err_msg.str().c_str());
                return nullptr;
            }
            xforms.push_back(static_cast<double*>(array.values()));
        }
    }

    write_pdb(structures, *out_stream, (bool)selected_only, (bool)displayed_only, xforms,
        (bool)all_coordsets, (bool)pqr, (bool)h36, poly_res_names, py_logger);

    if (out_stream->bad()) {
        PyErr_SetString(PyExc_ValueError, "Problem writing output PDB file");
        delete out_stream;
        return nullptr;
    }
    delete out_stream;
    Py_RETURN_NONE;
}

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
