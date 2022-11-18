// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include "_mmcif.h"
#include "corecif.h"
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Bond.h>
#include <atomstruct/Atom.h>
#include <atomstruct/CoordSet.h>
#include <atomstruct/Pseudobond.h>
#include <atomstruct/PBGroup.h>
#include <pdb/connect.h>
#include <atomstruct/tmpl/restmpl.h>
#include <logger/logger.h>
#include <arrays/pythonarray.h>	// Use python_voidp_array()
#include <readcif.h>
#include <float.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#define MIXED_CASE_BUILTIN_CATEGORIES 0

using std::string;
using std::vector;
using atomstruct::AtomicStructure;
using atomstruct::Structure;
using atomstruct::Residue;
using atomstruct::Bond;
using atomstruct::Atom;
using element::Element;
using atomstruct::Coord;
using atomstruct::Real;

namespace {

#ifndef M_PI
static const double M_PI = 3.14159265358979323846;
#endif

typedef vector<string> StringVector;

// Symbolic names for readcif arguments
static const bool Required = true;  // column is required

double
parse_float(const char* repr)
{
    // CIF floats are optionally followed by a unsigned ESD value in parenthesis.
    // If not a legal float, return NAN.
    char* endptr;
    if (*repr == '"' || *repr == '\'')
        ++repr;
    double d = strtod(repr, &endptr);
    if (repr == endptr)
        d = strtod("nan", nullptr);
    return d;
}

}

namespace mmcif {

struct SmallMolecule: public readcif::CIFFile
{
    static const char* builtin_categories[];
    PyObject* _logger;
    SmallMolecule(PyObject* logger, const StringVector& generic_categories);
    ~SmallMolecule();
    virtual void data_block(const string& name);
    virtual void reset_parse();
    virtual void finished_parse();

    void parse_cell();
    void parse_atom_site();
    void parse_atom_site_aniso();
    void parse_geom_bond();
    void parse_generic_category();

    void compute_cell_matrix();
    void to_cartesian(const double fract_xyz[3], Real xyz[3]);

    vector<Structure*> all_molecules;
    Structure* molecule;
    Residue* residue;
    double length_a, length_b, length_c;  // unit cell lengths
    double alpha, beta, gamma;            // unit cell angles
    double cell[3][3];
    std::map<string, std::pair<Atom*, char>> atom_lookup;
    std::map<string, StringVector> generic_tables;
};

const char* SmallMolecule::builtin_categories[] = {
    "cell", "atom_site", "atom_site_aniso", "geom_bond", 
};
#define MIXED_CASE_BUILTIN_CATEGORIES 0


SmallMolecule::SmallMolecule(PyObject* logger, const StringVector& generic_categories):
    _logger(logger)
{
    register_category("cell",
        [this] () {
            parse_cell();
        });
    register_category("atom_site",
        [this] () {
            parse_atom_site();
        }, { "cell" });
    register_category("atom_site_aniso",
        [this] () {
            parse_atom_site_aniso();
        }, { "atom_site" });
    register_category("geom_bond",
        [this] () {
            parse_geom_bond();
        }, { "atom_site" });
    for (auto& cat: generic_categories) {
#if MIXED_CASE_BUILTIN_CATEGORIES==0
        const string& category_ci = cat;
#else
        string category_ci(cat);
        for (auto& c: category_ci)
            c = tolower(c);
#endif
        if (std::find(std::begin(builtin_categories), std::end(builtin_categories), category_ci) != std::end(builtin_categories)) {
            logger::warning(_logger, "Can not override builtin parsing for "
                            "category: ", cat);
            continue;
        }
        register_category(cat,
            [this] () {
                parse_generic_category();
            });
    }

    reset_parse();
}

SmallMolecule::~SmallMolecule()
{
}

void
SmallMolecule::data_block(const string& /*name*/)
{
    if (molecule)
        finished_parse();
    reset_parse();
}

void
SmallMolecule::reset_parse()
{
    molecule = nullptr;
    residue = nullptr;
    length_a = length_b = length_c = 0;
    alpha = beta = gamma = M_PI / 2;
    atom_lookup.clear();
    generic_tables.clear();
}

void
SmallMolecule::finished_parse()
{
    if (!molecule)
        return;

    std::set<Atom*> has_bonds;
    for (auto& a: residue->atoms()) {
        if (a->bonds().size() > 0)
            has_bonds.insert(a);
    }
    pdb_connect::connect_residue_by_distance(residue, &has_bonds);
    pdb_connect::find_and_add_metal_coordination_bonds(molecule);
    molecule->metadata = generic_tables;
    molecule->use_best_alt_locs();
    all_molecules.push_back(molecule);
}

void
SmallMolecule::parse_cell()
{
    CIFFile::ParseValues pv;
    pv.reserve(6);
    try {
        pv.emplace_back(get_column("length_a", Required),
            [&] (const char* start) {
                length_a = parse_float(start);
            });
        pv.emplace_back(get_column("length_b", Required),
            [&] (const char* start) {
                length_b = parse_float(start);
            });
        pv.emplace_back(get_column("length_c", Required),
            [&] (const char* start) {
                length_c = parse_float(start);
            });
        pv.emplace_back(get_column("alpha"),
            [&] (const char* start) {
                alpha = parse_float(start) * M_PI / 180;
            });
        pv.emplace_back(get_column("beta"),
            [&] (const char* start) {
                beta = parse_float(start) * M_PI / 180;
            });
        pv.emplace_back(get_column("gamma"),
            [&] (const char* start) {
                gamma = parse_float(start) * M_PI / 180;
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping cell category: ", e.what());
        return;
    }
    parse_row(pv);
}

void
SmallMolecule::parse_atom_site()
{
    CIFFile::ParseValues pv;
    string label;
    char symbol[3];               // type_symbol
    double fract_xyz[3];
    double occupancy = NAN;
    int atom_serial = 0;
    char alt_id = 0;
    string disorder_assembly, disorder_group;

    symbol[0] = '\0';
    pv.reserve(10);
    try {
        pv.emplace_back(get_column("label", Required),
            [&] (const char* start, const char* end) {
                label = string(start, end - start);
            });
        pv.emplace_back(get_column("type_symbol"),
            [&] (const char* start) {
                symbol[0] = *start;
                symbol[1] = *(start + 1);
                if (readcif::is_whitespace(symbol[1]))
                    symbol[1] = '\0';
                else
                    symbol[2] = '\0';
            });
        pv.emplace_back(get_column("fract_x", Required),
            [&] (const char* start) {
                fract_xyz[0] = parse_float(start);
            });
        pv.emplace_back(get_column("fract_y", Required),
            [&] (const char* start) {
                fract_xyz[1] = parse_float(start);
            });
        pv.emplace_back(get_column("fract_z", Required),
            [&] (const char* start) {
                fract_xyz[2] = parse_float(start);
            });
        pv.emplace_back(get_column("occupancy"),
            [&] (const char* start) {
                occupancy = parse_float(start);
                if (std::isnan(occupancy))
                    occupancy = 1;
            });
        pv.emplace_back(get_column("disorder_assembly"),
            [&] (const char* start, const char* end) {
                disorder_assembly = string(start, end - start);
                if (disorder_assembly == ".")
                    disorder_assembly = "";
            });
        pv.emplace_back(get_column("disorder_group"),
            [&] (const char* start, const char* end) {
                disorder_group = string(start, end - start);
                if (disorder_group == ".")
                    disorder_group = "";
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping atom_site category: ", e.what());
        return;
    }

    string current_assembly, current_group, first_group;
    compute_cell_matrix();
    molecule = new AtomicStructure(_logger);
    residue = molecule->new_residue("UNK", "A", 1, 0);
    std::vector<Atom*> alt_atoms;
    int alt_count = 0;
    bool alt_warned = false;
    bool too_many = false;
    bool skip_group = false;
    while (parse_row(pv)) {
        if (label.empty())
            continue;
        if (label[0] == '?' || label[label.size() - 1] == '?') {
            // skip disordered atoms that don't have disorder_ tags
            if (!alt_warned) {
                logger::warning(_logger, "Ignoring alternate atoms locations with ? names near line ", line_number());
                alt_warned = true;
            }
            continue;
        }
        const Element* elem;
        if (symbol[0])
            elem = &Element::get_element(symbol);
        else {
            elem = &Element::get_element(label.c_str());
        }
        bool make_new_atom = true;
        Atom* a;
        if (disorder_assembly != current_assembly) {
            if (disorder_assembly.empty())
                alt_id = 0;
            else
                alt_id = 'A';
            current_assembly = disorder_assembly;
            first_group = current_group = disorder_group;
            alt_atoms.clear();
        } else if (!disorder_group.empty() && disorder_group != first_group) {
            if (disorder_group != current_group) {
                alt_id += 1;
                alt_count = 0;
                current_group = disorder_group;
                too_many = false;
            } else if (skip_group) {
                continue;
            } else {
                ++alt_count;
            }
            if (alt_count >= alt_atoms.size()) {
                if (!too_many) {
                    logger::warning(_logger, "More atoms in disorder assembly ", current_assembly,
                                    " group ", current_group, " than group ", first_group, " near line ", line_number());
                    too_many = true;
                }
                continue;
            }
            make_new_atom = false;
            a = alt_atoms[alt_count];
            if (a->element() != *elem) {
                skip_group = true;
                logger::warning(_logger, "Mismatched alternate atom, skipping rest of disorder assembly ", current_assembly,
                                " group ", current_group, " near line ", line_number());
                continue;
            }
            a->set_alt_loc(alt_id, true);
        }
        if (make_new_atom) {
            a = molecule->new_atom(label.c_str(), *elem);
            residue->add_atom(a);
            if (alt_id) {
                a->set_alt_loc(alt_id, true);
                alt_atoms.push_back(a);
            }
            ++atom_serial;
            a->set_serial_number(atom_serial);
        }
        Real xyz[3];
        to_cartesian(fract_xyz, xyz);
        Coord c(xyz);
        a->set_coord(c);
        if (!std::isnan(occupancy))
            a->set_occupancy(occupancy);
        atom_lookup[label] = {a, alt_id};
    }
}

void
SmallMolecule::parse_atom_site_aniso()
{
    CIFFile::ParseValues pv;
    string label;
    double u11, u12, u13, u22, u23, u33;

    pv.reserve(6);
    try {
        pv.emplace_back(get_column("label", Required),
            [&] (const char* start, const char* end) {
                label = string(start, end - start);
            });
        pv.emplace_back(get_column("U_11", Required),
            [&] (const char* start) {
                u11 = parse_float(start);
            });
        pv.emplace_back(get_column("U_22", Required),
            [&] (const char* start) {
                u22 = parse_float(start);
            });
        pv.emplace_back(get_column("U_33", Required),
            [&] (const char* start) {
                u33 = parse_float(start);
            });
        pv.emplace_back(get_column("U_23", Required),
            [&] (const char* start) {
                u23 = parse_float(start);
            });
        pv.emplace_back(get_column("U_13", Required),
            [&] (const char* start) {
                u13 = parse_float(start);
            });
        pv.emplace_back(get_column("U_12", Required),
            [&] (const char* start) {
                u12 = parse_float(start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping atom_site_aniso category: ", e.what());
        return;
    }
    while (parse_row(pv)) {
        const auto& ai = atom_lookup.find(label);
        if (ai == atom_lookup.end())
            continue;
        Atom *a = ai->second.first;
        char alt_id = ai->second.second;
        if (alt_id)
            a->set_alt_loc(alt_id, false);
        a->set_aniso_u(u11, u12, u13, u22, u23, u33);
    }
}

void
SmallMolecule::parse_geom_bond()
{
    CIFFile::ParseValues pv;
    string label1, label2;
    string sym1 = ".", sym2 = ".";

    pv.reserve(6);
    try {
        pv.emplace_back(get_column("atom_site_label_1", Required),
            [&] (const char* start, const char* end) {
                label1 = string(start, end - start);
            });
        pv.emplace_back(get_column("atom_site_label_2", Required),
            [&] (const char* start, const char* end) {
                label2 = string(start, end - start);
            });
        pv.emplace_back(get_column("site_symmetry_1"),
            [&] (const char* start, const char* end) {
                sym1 = string(start, end - start);
            });
        pv.emplace_back(get_column("site_symmetry_2"),
            [&] (const char* start, const char* end) {
                sym2 = string(start, end - start);
            });
    } catch (std::runtime_error& e) {
        logger::warning(_logger, "Skipping geom_bond category: ", e.what());
        return;
    }
    while (parse_row(pv)) {
        if (sym1 != "." || sym2 != ".")
            continue;  // ignore symmetry for now
        auto ai = atom_lookup.find(label1);
        if (ai == atom_lookup.end())
            continue;
        Atom *a1 = ai->second.first;
        ai = atom_lookup.find(label2);
        if (ai == atom_lookup.end())
            continue;
        Atom *a2 = ai->second.first;
        try {
            molecule->new_bond(a1, a2);
        } catch (std::exception&) {
            ; // probably bond for alternate atoms
        }
    }
}

void
SmallMolecule::parse_generic_category()
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
    StringVector& data = parse_whole_category();
    generic_tables[category_ci + " data"].swap(data);
}

void
SmallMolecule::compute_cell_matrix()
{
    // from https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography

    double n2 = (cos(alpha) - cos(gamma) * cos(beta)) / sin(gamma);
    cell[0][0] = length_a;
    cell[0][1] = 0;
    cell[0][2] = 0;
    cell[1][0] = length_b * cos(gamma);
    cell[1][1] = length_b * sin(gamma);
    cell[1][2] = 0;
    cell[2][0] = length_c * cos(beta);
    cell[2][1] = length_c * n2;
    double sin_beta = sin(beta);
    cell[2][2] = length_c * sqrt(sin_beta * sin_beta - n2 * n2);
}

void
SmallMolecule::to_cartesian(const double fract_xyz[3], Real xyz[3])
{
    xyz[0] = xyz[1] = xyz[2] = 0;
    for (auto i = 0; i < 3; ++i)
        for (auto j = 0; j < 3; ++j)
            xyz[i] += fract_xyz[i] * cell[i][j];
}

static PyObject*
structure_pointers(SmallMolecule &e)
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
parse_coreCIF_file(const char *filename, PyObject* logger)
{
    SmallMolecule small(logger, StringVector());
    small.parse_file(filename);
    return structure_pointers(small);
}

PyObject*
parse_coreCIF_file(const char *filename, const StringVector& generic_categories, PyObject* logger)
{
    SmallMolecule small(logger, generic_categories);
    small.parse_file(filename);
    return structure_pointers(small);
}

PyObject*
parse_coreCIF_buffer(const unsigned char *whole_file, PyObject* logger)
{
    SmallMolecule small(logger, StringVector());
    small.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(small);
}

PyObject*
parse_coreCIF_buffer(const unsigned char *whole_file, const StringVector& generic_categories, PyObject* logger)
{
    SmallMolecule small(logger, generic_categories);
    small.parse(reinterpret_cast<const char *>(whole_file));
    return structure_pointers(small);
}

}
