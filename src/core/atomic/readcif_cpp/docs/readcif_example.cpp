// vi: set expandtab shiftwidth=4 softtabstop=4:
#include "readcif.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>	// for getopt

using std::string;
using std::vector;
using namespace readcif;

#define MAX_CHAR_ATOM_NAME 4
#define MAX_CHAR_RES_NAME 4
#define MAX_CHAR_CHAIN_ID 4

static const bool Required = true;

struct Atom
{
    Atom() {
        clear();
    }
    void clear() {
        memset(atom_name, 0, MAX_CHAR_ATOM_NAME);
        memset(residue_name, 0, MAX_CHAR_RES_NAME);
        memset(chain_id, 0, MAX_CHAR_CHAIN_ID);
    }
    char element;
    char atom_name[MAX_CHAR_ATOM_NAME];
    char residue_name[MAX_CHAR_RES_NAME];
    char chain_id[MAX_CHAR_CHAIN_ID];
    int residue_num;
    float x, y, z;
};

struct ExtractCIF: CIFFile {
    ExtractCIF();
    void parse_audit_conform();
    void parse_atom_site();
    std::vector<Atom> atoms;
};

ExtractCIF::ExtractCIF()
{
#if 0
    using std::placeholder;
    register_category("audit_conform", 
                      std::bind(&ExtractCIF::audit_conform, this, _1));
    register_category("atom_site", 
                      std::bind(&ExtractCIF::parse_atom_site, this, _1));
#else
    // Personal preference, I like lambda functions better.
    // The lambda functions are needed because parse_XXXX
    // are member functions.
    register_category("audit_conform", 
                      [this] () {
                          parse_audit_conform();
                      });
    register_category("atom_site", 
                      [this] () {
                          parse_atom_site();
                      });
#endif
}

void
ExtractCIF::parse_audit_conform()
{
    // Looking for a way to tell if the mmCIF file was written
    // in the PDBx/mmCIF stylized format.  The following technique
    // is not guaranteed to work, but is sufficient for this example.
    string dict_name;
    float dict_version = 0;
    bool has_pdbx = false;

    CIFFile::ParseValues pv;
    pv.reserve(2);
    pv.emplace_back(get_column("dict_name"),
                    [&] (const char* start, const char* end) {
                        dict_name = string(start, end - start);
                    });
    pv.emplace_back(get_column("dict_version"),
                    [&] (const char* start) {
                        dict_version = atof(start);
                    });
    pv.emplace_back(get_column("pdbx_keywords"),
                    [&] (const char* start) {
                        has_pdbx = true;
                        set_PDBx_keywords(*start == 'Y' || *start == 'y');
                    });
    pv.emplace_back(get_column("pdbx_fixed_width_columns"),
                    [&] (const char* start, const char* end) {
                        has_pdbx = true;
                        for (const char *cp = start; cp < end; ++cp) {
                            if (isspace(*cp))
                                continue;
                            start = cp;
                            while (cp < end && !isspace(*cp))
                                ++cp;
                            set_PDBx_fixed_width_columns(string(start, cp - start));
                        }
                    });
    parse_row(pv);

    if (!has_pdbx && dict_name == "mmcif_pdbx.dic" && dict_version > 4) {
        set_PDBx_keywords(true);
        set_PDBx_fixed_width_columns("atom_site");
        set_PDBx_fixed_width_columns("atom_site_anisotrop");
    }
}

void
ExtractCIF::parse_atom_site()
{
    CIFFile::ParseValues pv;
    pv.reserve(10);
    Atom atom;
    pv.emplace_back(get_column("type_symbol", Required),
                    [&atom] (const char* start) {
                        atom.element = *start;
                    });
    pv.emplace_back(get_column("label_atom_id", Required),
                    [&atom] (const char* start, const char* end) {
                        size_t count = end - start;
                        if (count > MAX_CHAR_ATOM_NAME)
                            count = MAX_CHAR_ATOM_NAME;
                        strncpy(atom.atom_name, start, count);
                    });
    pv.emplace_back(get_column("label_comp_id", Required),
                    [&atom] (const char* start, const char* end) {
                        size_t count = end - start;
                        if (count > MAX_CHAR_RES_NAME)
                            count = MAX_CHAR_RES_NAME;
                        strncpy(atom.residue_name, start, count);
                    });
    pv.emplace_back(get_column("label_asym_id"),
                    [&atom] (const char* start, const char* end) {
                        size_t count = end - start;
                        if (count > MAX_CHAR_CHAIN_ID)
                            count = MAX_CHAR_CHAIN_ID;
                        strncpy(atom.chain_id, start, count);
                    });
    pv.emplace_back(get_column("label_seq_id", Required),
                    [&atom] (const char* start) {
                        atom.residue_num = readcif::str_to_int(start);
                    });
    // x, y, z are not required by mmCIF, but are by us
    pv.emplace_back(get_column("Cartn_x", Required),
                    [&atom] (const char* start) {
                        atom.x = readcif::str_to_float(start);
                    });
    pv.emplace_back(get_column("Cartn_y", Required),
                    [&atom] (const char* start) {
                        atom.y = readcif::str_to_float(start);
                    });
    pv.emplace_back(get_column("Cartn_z", Required),
                    [&atom] (const char* start) {
                        atom.z = readcif::str_to_float(start);
                    });
    while (parse_row(pv)) {
        atoms.push_back(atom);
        atom.clear();
    }
}

int
main(int argc, char **argv)
{
    bool debug = false;
    int opt;

    while ((opt = getopt(argc, argv, "d")) != -1) {
        switch (opt) {
            case 'd':
                debug = true;
                break;
            default: // '?'
                goto usage;
        }
    }

    if (optind != argc - 1) {
usage:
        std::cerr << "Usage: " << argv[0] << " [-d] mmCIF-filename\n";
        exit(EXIT_FAILURE);
    }

    ExtractCIF extract;

    try {
        extract.parse_file(argv[optind]);
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    size_t n = extract.atoms.size();
    std::cout << n << " atoms\n";

    if (debug) {
        size_t e = n > 10 ? 10 : n;
        for (size_t i = 0 ; i < e ; ++i) {
            Atom &a = extract.atoms[i];
            std::cout << a.atom_name << " " << a.residue_name << " "
                << a.residue_num << " " << a.chain_id << " "
                << a.x << " " << a.y << " " << a.z << '\n';
        }
        if (n > 10) {
            std::cout << "...\n";
            size_t s = n > 20 ? n - 10 : 10;
            for (size_t i = s; i < n ; ++i) {
                Atom &a = extract.atoms[i];
                std::cout << a.atom_name << " " << a.residue_name << " "
                    << a.residue_num << " " << a.chain_id << " "
                    << a.x << " " << a.y << " " << a.z << '\n';
            }
        }
    }

    return EXIT_SUCCESS;
}
