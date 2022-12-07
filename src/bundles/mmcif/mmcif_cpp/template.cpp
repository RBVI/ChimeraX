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

#include "mmcif.h"
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/MolResId.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/Sequence.h>
#include <atomstruct/tmpl/restmpl.h>
#if 0
// only needed if using tmpl::find_template_residue
#include <atomstruct/tmpl/residues.h>
#endif
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <algorithm>
#include <sstream>

#undef LEAVING_ATOMS

using std::string;
using std::vector;
using std::hash;
using std::set;

using atomstruct::AtomicStructure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using element::Element;
using atomstruct::MolResId;
using atomstruct::Sequence;
using atomstruct::Coord;
using atomstruct::PolymerType;

namespace mmcif {

using atomstruct::AtomName;
using atomstruct::ResName;

typedef vector<string> StringVector;

// Symbolic names for readcif arguments
static const bool Required = true;  // column is required

tmpl::Molecule* templates;
LocateFunc  locate_func;

const tmpl::Residue*
find_template_residue(const ResName& name, bool start, bool stop)
{
    const tmpl::Residue* tr = nullptr;
    if (name.empty())
        return tr;
    if (templates == nullptr)
        templates = new tmpl::Molecule();
    else
        tr = templates->find_residue(name);
    if (tr == nullptr) {
        if (locate_func == nullptr)
            return nullptr;
        string filename = locate_func(name);
        if (filename.empty())
            return nullptr;
        load_mmCIF_templates(filename.c_str());
        tr = templates->find_residue(name);
    }
    if (tr) {
        if (tr->polymer_type() == PolymerType::PT_AMINO) {
            if (start) {
                ResName terminus = name + "_LSN3";
                const tmpl::Residue* ttr = templates->find_residue(terminus);
                if (ttr)
                    return ttr;
            } else if (stop) {
                ResName terminus = name + "_LEO2H";
                const tmpl::Residue* ttr = templates->find_residue(terminus);
                if (ttr)
                    return ttr;
            }
        }
#if 0
// The atomic module's templates are missing atoms, eg., OP3 in G.
// So this gets rid of some warning about missing hydrogens
// and add others.  Needs more investigation.
        else if (tr->polymer_type() == PolymerType::PT_NUCLEIC) {
            if (start || stop) {
                // Until the PDB has mmCIF templates for RNA/DNA use built-ins
                const tmpl::Residue* ttr = tmpl::find_template_residue(name, stop, start);
                if (ttr)
                    return ttr;
            }
        }
#endif
    }
    return tr;
}

struct ExtractTemplate: public readcif::CIFFile
{
    // TODO? consider alternate atom names?
    // The PDB's mmCIF files use the canonical name,
    // so don't support the alternate names for now.
    ExtractTemplate();
    ~ExtractTemplate();
    virtual void data_block(const string& name);
    virtual void finished_parse();
    void parse_chem_comp();
    void parse_chem_comp_atom();
    void parse_chem_comp_bond();
    void parse_generic_residue_category();

    vector<tmpl::Residue*> all_residues;
    tmpl::Residue* residue;         // current residue
#ifdef LEAVING_ATOMS
    set<tmpl::Atom*> leaving_atoms; // in current residue
#endif
    string type;                    // residue type
    bool is_linking;
};

ExtractTemplate::ExtractTemplate(): residue(nullptr)
{
    all_residues.reserve(32);
    register_category("chem_comp",
        [this] () {
            parse_chem_comp();
        });
    register_category("chem_comp_atom",
        [this] () {
            parse_chem_comp_atom();
        }, { "chem_comp" });
    register_category("chem_comp_bond",
        [this] () {
            parse_chem_comp_bond();
        }, { "chem_comp", "chem_comp_atom" });
    register_category("pdbx_chem_comp_descriptor",
        [this] () {
            parse_generic_residue_category();
        }, { "chem_comp" });
    register_category("pdbx_chem_comp_identifier",
        [this] () {
            parse_generic_residue_category();
        }, { "chem_comp" });
}

ExtractTemplate::~ExtractTemplate()
{
    if (templates == nullptr || residue == nullptr)
        return;
    if (residue->atoms_map().size() == 0)
        templates->delete_residue(residue);
}

void
ExtractTemplate::data_block(const string& /*name*/)
{
    if (residue != nullptr)
        finished_parse();
    residue = nullptr;
#ifdef LEAVING_ATOMS
    leaving_atoms.clear();
#endif
    type.clear();
}

void
ExtractTemplate::finished_parse()
{
    if (residue == nullptr || !is_linking)
        return;
#ifdef LEAVING_ATOMS
    // figure out linking atoms
    //
    // The linking atoms of peptides and nucleotides used to connect
    // residues are "well known".  Links with other residue types are
    // explicitly given, so no need to figure which atoms are the
    // linking atoms.
    std::cout << residue->name() << ' ' << type << '\n';
    for (auto a1: leaving_atoms) {
        for (auto& b: a1->bonds()) {
            auto a2 = b->other_atom(a1);
            if (leaving_atoms.find(a2) != leaving_atoms.end())
                continue;
            std::cout << "    linking heavy atom: " << a2->name() << '\n';
            break;
        }
    }
#endif
    if (residue->polymer_type() == PolymerType::PT_AMINO) {
        tmpl::Atom* n = residue->find_atom("N");
        residue->chief(n);
        tmpl::Atom* c;
        if (type.find("c-gamma") != string::npos)
            c = residue->find_atom("CG");
        else if (type.find("c-delta") != string::npos)
            c = residue->find_atom("CD");
        else
            c = residue->find_atom("C");
        residue->link(c);
    } else if (residue->polymer_type() == PolymerType::PT_NUCLEIC) {
        tmpl::Atom* p = residue->find_atom("P");
        residue->chief(p);
        tmpl::Atom* o3p = residue->find_atom("O3'");
        residue->link(o3p);
    }
}

void
ExtractTemplate::parse_chem_comp()
{
    // TODO: parse "all" columns of chem_comp table and save in TmplMolecule's
    // metadata for extraction when opening a CCD file as an atomic structure
    ResName name;
    string  modres;
    string  description;
    string  code;
    bool    ambiguous = false;
    type.clear();

    CIFFile::ParseValues pv;
    pv.reserve(6);
    try {
        pv.emplace_back(get_column("id", Required),
            [&] (const char* start, const char* end) {
                name = ResName(start, end - start);
            });
        pv.emplace_back(get_column("type"),
            [&] (const char* start, const char* end) {
                type = string(start, end - start);
            });
        pv.emplace_back(get_column("name"),
            [&] (const char* start, const char* end) {
                description = string(start, end - start);
            });
        pv.emplace_back(get_column("mon_nstd_parent_comp_id"),
            [&] (const char* start, const char* end) {
                modres = string(start, end - start);
                if (modres == "?" || modres == ".")
                    modres.clear();
            });
        pv.emplace_back(get_column("one_letter_code"),
            [&] (const char* start, const char* end) {
                code = string(start, end - start);
                if (code == "." || code == "?")
                    code.clear();
            });
        pv.emplace_back(get_column("pdbx_ambiguous_flag"),
            [&] (const char* start) {
                ambiguous = *start == 'Y' || *start == 'y';
            });
    } catch (std::runtime_error& e) {
        std::ostringstream err_msg;
        err_msg << "chem_comp: " << e.what();
        throw std::runtime_error(err_msg.str());
    }
    (void) parse_row(pv);

    // convert type to lowercase
    for (auto& c: type) {
        if (isupper(c))
            c = tolower(c);
    }
    is_linking = type.find(" linking") != string::npos;
    bool is_peptide = type.find("peptide") != string::npos;
    bool is_nucleotide = type.find("dna ") != string::npos
        || type.find("rna ") != string::npos;
    residue = templates->new_residue(name.c_str());
    residue->description(description);
    residue->pdbx_ambiguous = ambiguous;
    all_residues.push_back(residue);
    if (!is_peptide && !is_nucleotide)
        return;
    if (is_peptide)
        residue->polymer_type(PolymerType::PT_AMINO);
    else
        residue->polymer_type(PolymerType::PT_NUCLEIC);
    char old_code;
    if (is_peptide)
        old_code = Sequence::protein3to1(name);
    else
        old_code = Sequence::nucleic3to1(name);
    if (code.size() == 1) {
        ; // FALL THROUGH
    } else {
        if (code.size() > 1 || modres.find(',') != string::npos) {
            code = 'X';
        } else if (!modres.empty()) {
            if (is_peptide)
                code = Sequence::protein3to1(modres);
            else
                code = Sequence::nucleic3to1(modres);
        } else if (code.empty()) {
            return;  // let sequence code pick what unknown residues should be
        }
    }
    if (old_code == 'X')
        Sequence::assign_rname3to1(name, code[0], is_peptide);
    else if (old_code != '?' && old_code != code[0])
        // ChimeraX uses ? for N, DN, and UNK
        // while CCD uses N, N, and X, respectively.
        // So don't warn about those.
        // TODO: log this somehow
        std::cerr << "Not changing " << name <<
            "'s sequence abbreviation (existing: " << old_code << ", new: " <<
            code << ")\n";
}

void
ExtractTemplate::parse_chem_comp_atom()
{
    char    chirality;
    AtomName  name;
    char    symbol[3];
    float   x, y, z;
    float   pdbx_x = std::numeric_limits<float>::quiet_NaN(),
            pdbx_y = std::numeric_limits<float>::quiet_NaN(),
            pdbx_z = std::numeric_limits<float>::quiet_NaN();
#ifdef LEAVING_ATOMS
    bool    leaving = false;
#endif
    if (residue == nullptr)
        return;

    CIFFile::ParseValues pv;
    pv.reserve(8);
    try {
        pv.emplace_back(get_column("atom_id", Required),
            [&] (const char* start, const char* end) {
                name = AtomName(start, end - start);
            });
        //pv.emplace_back(get_column("alt_atom_id", true),
        //    [&] (const char* start, const char* end) {
        //        alt_name = string(start, end - start);
        //    });
        pv.emplace_back(get_column("type_symbol", Required),
            [&] (const char* start) {
                symbol[0] = *start;
                symbol[1] = *(start + 1);
                if (readcif::is_whitespace(symbol[1]))
                    symbol[1] = '\0';
                else
                    symbol[2] = '\0';
            });
#ifdef LEAVING_ATOMS
        pv.emplace_back(get_column("pdbx_leaving_atom_flag"),
            [&] (const char* start) {
                leaving = *start == 'Y' || *start == 'y';
            });
#endif
        pv.emplace_back(get_column("pdbx_stereo_config", Required),
            [&] (const char* start) {
                chirality = *start;
            });
        pv.emplace_back(get_column("model_Cartn_x", Required),
            [&] (const char* start) {
                x = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("model_Cartn_y", Required),
            [&] (const char* start) {
                y = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("model_Cartn_z", Required),
            [&] (const char* start) {
                z = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("pdbx_model_Cartn_x_ideal"),
            [&] (const char* start) {
                pdbx_x = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("pdbx_model_Cartn_y_ideal"),
            [&] (const char* start) {
                pdbx_y = readcif::str_to_float(start);
            });
        pv.emplace_back(get_column("pdbx_model_Cartn_z_ideal"),
            [&] (const char* start) {
                pdbx_z = readcif::str_to_float(start);
            });
    } catch (std::runtime_error& e) {
        std::ostringstream err_msg;
        err_msg << "chem_comp_atom: " << e.what();
        throw std::runtime_error(err_msg.str());
    }
    while (parse_row(pv)) {
        const Element& elem = Element::get_element(symbol);
        tmpl::Atom* a = templates->new_atom(name, elem, chirality);
        if (std::isnan(pdbx_x)) {
            tmpl::Coord c(x, y, z);
            a->set_coord(c);
        } else {
            tmpl::Coord c(pdbx_x, pdbx_y, pdbx_z);
            a->set_coord(c);
        }
        residue->add_atom(a);
#ifdef LEAVING_ATOMS
        if (leaving)
            leaving_atoms.insert(a);
#endif
        Residue::ideal_chirality[residue->name()][name] = chirality;
    }
}

void
ExtractTemplate::parse_chem_comp_bond()
{
    AtomName name1, name2;
    if (residue == nullptr)
        return;

    CIFFile::ParseValues pv;
    pv.reserve(2);
    try {
        pv.emplace_back(get_column("atom_id_1", Required),
            [&] (const char* start, const char* end) {
                name1 = AtomName(start, end - start);
            });
        pv.emplace_back(get_column("atom_id_2", Required),
            [&] (const char* start, const char* end) {
                name2 = AtomName(start, end - start);
            });
    } catch (std::runtime_error& e) {
        std::ostringstream err_msg;
        err_msg << "chem_comp_bond: " << e.what();
        throw std::runtime_error(err_msg.str());
    }
    while (parse_row(pv)) {
        tmpl::Atom* a1 = residue->find_atom(name1);
        tmpl::Atom* a2 = residue->find_atom(name2);
        if (a1 == nullptr || a2 == nullptr)
            continue;
        templates->new_bond(a1, a2);
    }
}

void
ExtractTemplate::parse_generic_residue_category()
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
    StringVector& data = parse_whole_category();
    residue->metadata[category_ci] = colinfo;
    residue->metadata[category_ci + " data"].swap(data);
}

void
load_mmCIF_templates(const char* filename)
{
    if (templates == nullptr)
        templates = new tmpl::Molecule();

    ExtractTemplate extract;
    try {
        extract.parse_file(filename);
    } catch (std::exception& e) {
        std::cerr << "Loading template file failed: " << e.what() << '\n';
    }
#if 0
    // DEBUG
    // for each residue, print out the name, code, and bonds
    for (auto& r: extract.all_residues) {
        char code = r->single_letter_code();
        tmpl::Atom* chief = r->chief();
        tmpl::Atom* link = r->link();
        std::cout << "Residue " << r->name() << ":\n";
        if (code)
            std::cout << "  single letter code: " << code << '\n';
        if (chief)
            std::cout << "  chief atom: " << chief->name() << '\n';
        if (link)
            std::cout << "  link atom: " << link->name() << '\n';
        for (auto& akv: r->atoms_map()) {
            auto& a1 = akv.second;
            for (auto& bkv: a1->bonds_map()) {
                auto& a2 = bkv.first;
                if (a1->name() < a2->name())
                    std::cout << a1->name() << " - " << a2->name() << '\n';
            }
        }
    }
#endif
}

void
set_locate_template_function(LocateFunc function)
{
    locate_func = function;
}

void
set_Python_locate_function(PyObject* function)
{
    static PyObject* save_reference_to_function = nullptr;

    if (function == nullptr || function == Py_None) {
        locate_func = nullptr;
        return;
    }
    if (!PyCallable_Check(function))
        throw std::logic_error("function must be a callable object");

    if (locate_func != nullptr)
        Py_DECREF(save_reference_to_function);
    Py_INCREF(function);
    save_reference_to_function = function;

    locate_func = [function] (const ResName& name) -> std::string {
      PyObject* name_arg = PyUnicode_DecodeUTF8(name.data(), name.size(), "replace");
        PyObject* result = PyObject_CallFunction(function, "O", name_arg);
        Py_XDECREF(name_arg);
        if (result == nullptr)
            throw std::runtime_error("Python Error");
        if (result == Py_None) {
            Py_DECREF(result);
            return std::string();
        }
        if (!PyUnicode_Check(result)) {
            Py_DECREF(result);
            throw std::logic_error("locate function should return a string");
        }
        Py_ssize_t size;
        const char *data = PyUnicode_AsUTF8AndSize(result, &size);
        string cpp_result(data, size);
        Py_DECREF(result);
        return cpp_result;
    };
}

} // namespace mmcif
