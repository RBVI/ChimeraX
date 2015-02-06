// vi: set expandtab ts=4 sw=4:
#include "mmcif.h"
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/Sequence.h>
#include <blob/StructBlob.h>
#include <atomstruct/connect.h>
#include <atomstruct/tmpl/restmpl.h>
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>
#include <WrapPy3.h>

using std::string;
using std::vector;
using std::unordered_map;
using std::hash;
using std::set;

using atomstruct::AtomicStructure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using atomstruct::CoordSet;
using atomstruct::Element;
using atomstruct::MolResId;
using atomstruct::Sequence;
using basegeom::Coord;

namespace mmcif {

using atomstruct::AtomName;

tmpl::Molecule* templates;
LocateFunc  locate_func;

const tmpl::Residue*
find_template_residue(const string& name)
{
    if (templates == nullptr)
        templates = new tmpl::Molecule();

    tmpl::Residue* tr = templates->find_residue(name);
    if (tr)
        return tr;
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
    void parse_chem_comp(bool in_loop);
    void parse_chem_comp_atom(bool in_loop);
    void parse_chem_comp_bond(bool in_loop);

    vector<tmpl::Residue*> all_residues;
    tmpl::Residue* residue;         // current residue
    set<tmpl::Atom*> leaving_atoms; // in current residue
    string type;                    // residue type
};

ExtractTemplate::ExtractTemplate(): residue(NULL)
{
    all_residues.reserve(32);
    register_category("chem_comp",
        [this] (bool in_loop) {
            parse_chem_comp(in_loop);
        });
    register_category("chem_comp_atom",
        [this] (bool in_loop) {
            parse_chem_comp_atom(in_loop);
        }, { "chem_comp" });
    register_category("chem_comp_bond",
        [this] (bool in_loop) {
            parse_chem_comp_bond(in_loop);
        }, { "chem_comp", "chem_comp_atom" });
}

void
ExtractTemplate::data_block(const string& /*name*/)
{
    if (residue != NULL)
        finished_parse();
    residue = NULL;
    leaving_atoms.clear();
    type.clear();
}

void
ExtractTemplate::finished_parse()
{
    if (residue == NULL)
        return;
    // figure out linking atoms
    //
    // The linking atoms of peptides and nucleotides used to connect
    // residues are "well known".  Links with other residue types are
    // explicitly given, so no need to figure which atoms are the
    // linking atoms.
#if 0
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
    if (!type.empty()) {
        bool is_peptide = type.find("peptide") != string::npos
                                    || type.find("PEPTIDE") != string::npos;
        bool is_nucleotide = type.find("DNA") != string::npos
                                    || type.find("RNA") != string::npos;
        if (is_peptide) {
            residue->description("peptide");
            residue->chief(residue->find_atom("N"));
            residue->link(residue->find_atom("C"));
        } else if (is_nucleotide) {
            residue->description("nucleotide");
            residue->chief(residue->find_atom("P"));
            residue->link(residue->find_atom("O3'"));
        }
    }
}

void
ExtractTemplate::parse_chem_comp(bool /*in_loop*/)
{
    string  name;
    string  modres;
    char    code = '\0';

    CIFFile::ParseValues pv;
    pv.reserve(2);
    pv.emplace_back(get_column("id", true), true,
        [&] (const char* start, const char* end) {
            name = string(start, end - start);
        });
    pv.emplace_back(get_column("type", false), true,
        [&] (const char* start, const char* end) {
            type = string(start, end - start);
        });
    pv.emplace_back(get_column("mon_nstd_parent_comp_id", false), true,
        [&] (const char* start, const char* end) {
            modres = string(start, end - start);
            if (modres == "?" || modres == ".")
                modres = "";
        });
    pv.emplace_back(get_column("one_letter_code", false), false,
        [&] (const char* start, const char*) {
            code = *start;
            if (code == '.' || code == '?')
                code = '\0';
        });
    (void) parse_row(pv);

    residue = templates->new_residue(name.c_str());
    all_residues.push_back(residue);
    bool is_peptide = !type.empty() && (type.find("peptide") != string::npos
                                    || type.find("PEPTIDE") != string::npos);
    if (!modres.empty()) {
        if (!code)
            code = is_peptide ? Sequence::protein3to1(modres)
                                                : Sequence::nucleic3to1(modres);
        if (code && code != 'X')
            Sequence::assign_rname3to1(name, code, is_peptide);
    } else if (code) {
        char let = is_peptide ? Sequence::protein3to1(name)
                                                : Sequence::nucleic3to1(name);
        if (let == 'X')
            Sequence::assign_rname3to1(name, code, is_peptide);
    }
}

void
ExtractTemplate::parse_chem_comp_atom(bool /*in_loop*/)
{
    AtomName  name;
    char    symbol[3];
    float   x, y, z;
    bool    leaving = false;

    CIFFile::ParseValues pv;
    pv.reserve(8);
    pv.emplace_back(get_column("atom_id", true), true,
        [&] (const char* start, const char* end) {
            name = AtomName(start, end - start);
        });
    //pv.emplace_back(get_column("alt_atom_id", true), true,
    //    [&] (const char* start, const char* end) {
    //        alt_name = string(start, end - start);
    //    });
    pv.emplace_back(get_column("type_symbol", true), false,
        [&] (const char* start, const char*) {
            symbol[0] = *start;
            symbol[1] = *(start + 1);
            if (readcif::is_whitespace(symbol[1]))
                symbol[1] = '\0';
            else
                symbol[2] = '\0';
        });
    pv.emplace_back(get_column("pdbx_leaving_atom_flag", false), false,
        [&] (const char* start, const char*) {
            leaving = *start == 'Y' || *start == 'y';
        });
    pv.emplace_back(get_column("model_Cartn_x", true), false,
        [&] (const char* start, const char*) {
            x = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("model_Cartn_y", true), false,
        [&] (const char* start, const char*) {
            y = readcif::str_to_float(start);
        });
    pv.emplace_back(get_column("model_Cartn_z", true), false,
        [&] (const char* start, const char*) {
            z = readcif::str_to_float(start);
        });
    while (parse_row(pv)) {
        Element elem(symbol);
        tmpl::Atom* a = templates->new_atom(name, elem);
        tmpl::Coord c(x, y, z);
        a->set_coord(c);
        residue->add_atom(a);
        if (leaving)
            leaving_atoms.insert(a);
    }
}

void
ExtractTemplate::parse_chem_comp_bond(bool /*in_loop*/)
{
    AtomName name1, name2;

    CIFFile::ParseValues pv;
    pv.reserve(2);
    pv.emplace_back(get_column("atom_id_1", true), true,
        [&] (const char* start, const char* end) {
            name1 = AtomName(start, end - start);
        });
    pv.emplace_back(get_column("atom_id_2", true), true,
        [&] (const char* start, const char* end) {
            name2 = AtomName(start, end - start);
        });
    while (parse_row(pv)) {
        tmpl::Atom* a1 = residue->find_atom(name1);
        tmpl::Atom* a2 = residue->find_atom(name2);
        if (a1 == NULL || a2 == NULL)
            continue;
        templates->new_bond(a1, a2);
    }
}

void
load_mmCIF_templates(const char* filename)
{
    if (templates == NULL)
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
    static PyObject* save_reference_to_function = NULL;

    if (function == NULL) {
        locate_func = nullptr;
        return;
    }
    if (!PyCallable_Check(function))
        throw std::logic_error("function must be a callable object");

    if (locate_func != nullptr)
        Py_DECREF(save_reference_to_function);
    Py_INCREF(function);
    save_reference_to_function = function;

    locate_func = [function] (const string& name) -> std::string {
        PyObject* name_arg = wrappy::pyObject(name);
        PyObject* result = PyObject_CallFunction(function, "O", name_arg);
        Py_XDECREF(name_arg);
        if (result == NULL)
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
