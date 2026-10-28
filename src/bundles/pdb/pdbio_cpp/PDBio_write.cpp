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

const std::vector<std::string> record_order = {
    "HEADER", "OBSLTE", "TITLE", "CAVEAT", "COMPND", "SOURCE", "KEYWDS", "EXPDTA", "AUTHOR",
    "REVDAT", "SPRSDE", "JRNL", "REMARK", "DBREF", "SEQADV", "SEQRES", "MODRES", "FTNOTE",
    "HET", "HETNAM", "HETSYN", "FORMUL", "HELIX", "SHEET", "TURN", "SSBOND", "LINK", "HYDBND",
    "SLTBRG", "CISPEP", "SITE", "CRYST1", "ORIGX1", "ORIGX2", "ORIGX3", "SCALE1", "SCALE2",
    "SCALE3", "MTRIX1", "MTRIX2", "MTRIX3", "TVECT", "MODEL", "ATOM", "SIGATM", "ANISOU",
    "SIGUIJ", "TER", "HETATM", "ENDMDL", "CONECT", "MASTER", "END",
};

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
    std::set<ChainID> incomplete_chains;
    auto pbg = s->pb_mgr().get_group(Structure::PBG_MISSING_STRUCTURE);
    if (pbg != nullptr) {
        for (auto pb: pbg->pseudobonds()) {
            incomplete_chains.insert(pb->atoms()[0]->residue()->chain_id());
        }
    }
    for (auto chain: s->chains()) {
        int record_num = 1;
        for (size_t i = 0; i < chain->characters().size(); i += 13) {
                // only output SEQRES for chains where we believe we know the complete polymer sequence
                if (!chain->from_seqres()
                && incomplete_chains.find(chain->chain_id()) != incomplete_chains.end())
                    continue;
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
        std::sort(ordering.begin(), ordering.end(),
            [](Atom* a1, Atom* a2) { return a1->coord_index() < a2->coord_index(); });

        for (auto a: ordering) {
            if (selected_only && !a->selected())
                continue;
            if (displayed_only && !a->display())
                continue;
            std::string aname = s->asterisks_translated ?
                primes_to_asterisks(a->name().c_str()) : a->name().c_str();
            if (aname.size() > 4 && !*warned_atom_name_length) {
                *warned_atom_name_length = true;
                logger::warning(py_logger, "Atom names longer than 4 characters; truncating");
            }
            if (strlen(a->element().name()) > 1) {
                strncpy(*rec_name, aname.c_str(), 4);
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
                    strncpy(*rec_name, aname.c_str(), 4);
                }
            }
            (*rec_name)[4] = '\0';
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
