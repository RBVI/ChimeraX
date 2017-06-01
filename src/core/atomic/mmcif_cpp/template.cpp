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
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/Sequence.h>
#include <atomstruct/connect.h>
#include <atomstruct/tmpl/restmpl.h>
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <algorithm>
#include <WrapPy3.h>

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

namespace mmcif {

using atomstruct::AtomName;
using atomstruct::ResName;

tmpl::Molecule* templates;
LocateFunc  locate_func;

// standard_residues have standard linkage 
static std::set<ResName> standard_peptides = {
    "ALA", "ARG", "ASN", "ASP", "ASX", "CYS",
    "GLN", "GLU", "GLX", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "UNK", "VAL"
};
static std::set<ResName> standard_nucleotides = {
    "A", "C", "G", "I", "T", "U",
    "DA", "DC", "DG", "DT",
};

const tmpl::Residue*
find_template_residue(const ResName& name)
{
    if (name.empty())
        return nullptr;
    if (templates == nullptr)
        templates = new tmpl::Molecule();
    else {
        tmpl::Residue* tr = templates->find_residue(name);
        if (tr)
            return tr;
    }
    if (locate_func == nullptr)
        return nullptr;
    string filename = locate_func(name);
    if (filename.empty())
        return nullptr;
    load_mmCIF_templates(filename.c_str());
    return templates->find_residue(name);
}

struct ExtractTemplate: public readcif::CIFFile
{
    // TODO? consider alternate atom names?
    // The PDB's mmCIF files use the canonical name,
    // so don't support the alternate names for now.
    ExtractTemplate();
    virtual void data_block(const string& name);
    virtual void finished_parse();
    void parse_chem_comp();
    void parse_chem_comp_atom();
    void parse_chem_comp_bond();

    vector<tmpl::Residue*> all_residues;
    tmpl::Residue* residue;         // current residue
#ifdef LEAVING_ATOMS
    set<tmpl::Atom*> leaving_atoms; // in current residue
#endif
    string type;                    // residue type
    bool is_peptide;
    bool is_nucleotide;
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
    if (residue == nullptr)
        return;
#ifdef LEAVING_ATOMS
    // figure out linking atoms
    //
    // The linking atoms of peptides and nucleotides used to connect
    // residues are "well known".  Links with other residue types are
    // explicitly given, so no need to figure which atoms are the
    // linking atoms.
    for (auto& akv: residue->atoms_map()) {
        auto& a1 = akv.second;
        if (leaving_atoms.find(a1) != leaving_atoms.end())
            continue;
        for (auto& bkv: a1->bonds_map()) {
            auto& a2 = bkv.first;
            if (a2->element() == Element::H
            || leaving_atoms.find(a2) == leaving_atoms.end())
                continue;
            std::cout << residue->name() << " linking atom: " << a1->name() << '\n';
            break;
        }
    }
#endif
    if (is_peptide
    && standard_peptides.find(residue->name()) != standard_peptides.end()) {
        residue->description("peptide");
        tmpl::Atom* n = residue->find_atom("N");
        residue->chief(n);
        tmpl::Atom* c = residue->find_atom("C");
        residue->link(c);
    } else if (is_nucleotide
    && standard_nucleotides.find(residue->name()) != standard_nucleotides.end()) {
        residue->description("nucleotide");
        tmpl::Atom* p = residue->find_atom("P");
        residue->chief(p);
        tmpl::Atom* o3p = residue->find_atom("O3'");
        residue->link(o3p);
    }
}

void
ExtractTemplate::parse_chem_comp()
{
    ResName  name;
    ResName  modres;
    char    code = '\0';
    bool    ambiguous = false;
    type.clear();

    CIFFile::ParseValues pv;
    pv.reserve(6);
    pv.emplace_back(get_column("id", true),
        [&] (const char* start, const char* end) {
            name = ResName(start, end - start);
        });
    pv.emplace_back(get_column("type"),
        [&] (const char* start, const char* end) {
            type = string(start, end - start);
        });
    pv.emplace_back(get_column("three_letter_code", false),
        [&] (const char* start, const char* end) {
            modres = ResName(start, end - start);
            if (modres == "?" || modres == ".")
                modres = "";
        });
    pv.emplace_back(get_column("one_letter_code", false),
        [&] (const char* start) {
            code = *start;
            if (code == '.' || code == '?')
                code = '\0';
        });
    pv.emplace_back(get_column("pdbx_ambiguous_flag"),
        [&] (const char* start) {
            ambiguous = *start == 'Y' || *start == 'y';
        });
    (void) parse_row(pv);

    // convert type to lowercase
    for (auto& c: type) {
        if (isupper(c))
            c = tolower(c);
    }
    is_peptide = type.find("peptide") != string::npos;
    is_nucleotide = type.compare(0, 3, "dna") == 0
        || type.compare(0, 3, "rna") == 0;
    residue = templates->new_residue(name);
    residue->pdbx_ambiguous = ambiguous;
    all_residues.push_back(residue);
    if (!modres.empty()) {
        if (!code) {
            if (is_peptide)
                code = Sequence::protein3to1(modres);
            else if (is_nucleotide)
                code = Sequence::nucleic3to1(modres);
        }
        if (code && code != 'X')
            Sequence::assign_rname3to1(name, code, is_peptide);
    } else if (code) {
        if (is_peptide) {
            if (Sequence::protein3to1(name) == 'X')
                Sequence::assign_rname3to1(name, code, true);
        } else if (is_nucleotide) {
            if (Sequence::nucleic3to1(name) == 'X')
                Sequence::assign_rname3to1(name, code, false);
        }
    }
}

void
ExtractTemplate::parse_chem_comp_atom()
{
    AtomName  name;
    char    symbol[3];
    float   x, y, z;
#ifdef LEAVING_ATOMS
    bool    leaving = false;
#endif

    CIFFile::ParseValues pv;
    pv.reserve(8);
    pv.emplace_back(get_column("atom_id", true),
        [&] (const char* start, const char* end) {
            name = AtomName(start, end - start);
        });
    //pv.emplace_back(get_column("alt_atom_id", true),
    //    [&] (const char* start, const char* end) {
    //        alt_name = string(start, end - start);
    //    });
    pv.emplace_back(get_column("type_symbol", true),
        [&] (const char* start) {
            symbol[0] = *start;
            symbol[1] = *(start + 1);
            if (readcif::is_whitespace(symbol[1]))
                symbol[1] = '\0';
            else
                symbol[2] = '\0';
        });
#ifdef LEAVING_ATOMS
    pv.emplace_back(get_column("pdbx_leaving_atom_flag", false),
        [&] (const char* start) {
            leaving = *start == 'Y' || *start == 'y';
        });
#endif
    pv.emplace_back(get_column("model_Cartn_x", true),
        [&] (const char* start) {
            x = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("model_Cartn_y", true),
        [&] (const char* start) {
            y = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("model_Cartn_z", true),
        [&] (const char* start) {
            z = readcif::str_to_float(start);
        });
    while (parse_row(pv)) {
        const Element& elem = Element::get_element(symbol);
        tmpl::Atom* a = templates->new_atom(name, elem);
        tmpl::Coord c(x, y, z);
        a->set_coord(c);
        residue->add_atom(a);
#ifdef LEAVING_ATOMS
        if (leaving)
            leaving_atoms.insert(a);
#endif
    }
}

void
ExtractTemplate::parse_chem_comp_bond()
{
    AtomName name1, name2;

    CIFFile::ParseValues pv;
    pv.reserve(2);
    pv.emplace_back(get_column("atom_id_1", true),
        [&] (const char* start, const char* end) {
            name1 = AtomName(start, end - start);
        });
    pv.emplace_back(get_column("atom_id_2", true),
        [&] (const char* start, const char* end) {
            name2 = AtomName(start, end - start);
        });
    while (parse_row(pv)) {
        tmpl::Atom* a1 = residue->find_atom(name1);
        tmpl::Atom* a2 = residue->find_atom(name2);
        if (a1 == nullptr || a2 == nullptr)
            continue;
        templates->new_bond(a1, a2);
    }
}

void
load_mmCIF_templates(const char* filename)
{
    if (templates == nullptr)
        templates = new tmpl::Molecule();

    ExtractTemplate extract;
    extract.parse_file(filename);
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
        PyObject* name_arg = wrappy::pyObject((const char*)name);
        PyObject* result = PyObject_CallFunction(function, "O", name_arg);
        Py_XDECREF(name_arg);
        if (result == nullptr)
            throw wrappy::PythonError();
        if (result == Py_None) {
            Py_DECREF(result);
            return std::string();
        }
        if (!PyUnicode_Check(result)) {
            Py_DECREF(result);
            throw std::logic_error("locate function should return a string");
        }
        string cpp_result = wrappy::PythonUnicode_AsCppString(result);
        Py_DECREF(result);
        return cpp_result;
    };
}

} // namespace mmcif
