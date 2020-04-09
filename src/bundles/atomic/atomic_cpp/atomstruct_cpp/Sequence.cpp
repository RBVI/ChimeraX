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

#include <cctype>
#include <exception>
#include <iostream>
#include <Python.h>
#include <regex>

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Chain.h"

#include <pyinstance/PythonInstance.instantiate.h>
template class pyinstance::PythonInstance<atomstruct::Sequence>;

namespace atomstruct {

Sequence::_1Letter_Map Sequence::_nucleic3to1 = {
        {"A", 'A'},
        {"+A", 'A'},
        {"ADE", 'A'},
        {"DA", 'A'},
        {"C", 'C'},
        {"+C", 'C'},
        {"CYT", 'C'},
        {"DC", 'C'},
        {"G", 'G'},
        {"+G", 'G'},
        {"GUA", 'G'},
        {"DG", 'G'},
        {"T", 'T'},
        {"+T", 'T'},
        {"THY", 'T'},
        {"DT", 'T'},
        {"U", 'U'},
        {"+U", 'U'},
        {"URA", 'U'},
        {"DN", '?'},
        {"N", '?'}
};
Sequence::_1Letter_Map Sequence::_protein3to1 = {
        {"ALA", 'A'},
        {"ARG", 'R'},
        {"ASH", 'D'}, // Amber (protonated ASP)
        {"ASN", 'N'},
        {"ASP", 'D'},
        {"ASX", 'B'}, // ambiguous ASP/ASN
        {"CYS", 'C'},
        {"CYX", 'C'}, // Amber (disulphide)
        {"GLH", 'E'}, // Amber (protonated GLU)
        {"GLU", 'E'},
        {"GLN", 'Q'},
        {"GLX", 'Z'}, // ambiguous GLU/GLN
        {"GLY", 'G'},
        {"HID", 'H'}, // Amber (delta protonated)
        {"HIE", 'H'}, // Amber (epsilon protonated)
        {"HIP", 'H'}, // Amber (doubly protonated)
        {"HIS", 'H'},
        {"HYP", 'P'}, // hydroxyproline, which in theory
            // has its own letter ('O') but using that is problematic
            // with similarity matrices
        {"ILE", 'I'},
        {"LEU", 'L'},
        {"LYS", 'K'},
        {"MET", 'M'},
        {"MSE", 'M'}, // Selenomethionine
        {"PHE", 'F'},
        {"PRO", 'P'},
        {"SER", 'S'},
        {"THR", 'T'},
        {"TRP", 'W'},
        {"TYR", 'Y'},
        {"UNK", '?'},
        {"VAL", 'V'}
};
Sequence::_1Letter_Map Sequence::_rname3to1;

void
Sequence::_init_rname_map()
{
    for (auto mapping : Sequence::_nucleic3to1) {
        Sequence::_rname3to1[mapping.first] = mapping.second;
    }
    for (auto mapping : Sequence::_protein3to1) {
        Sequence::_rname3to1[mapping.first] = mapping.second;
    }
}

// 3-letter codes
Sequence::Sequence(const std::vector<ResName>& res_names, std::string name): _name(name)
{
    for (auto rn: res_names) {
        this->push_back(rname3to1(rn));
    }
}

void
Sequence::assign_rname3to1(const ResName& rname, char let, bool protein)
{
    if (protein)
        _protein3to1[rname] = let;
    else
        _nucleic3to1[rname] = let;
    if (_rname3to1.empty())
        _init_rname_map();
    _rname3to1[rname] = let;
}

unsigned int
Sequence::gapped_to_ungapped(unsigned int index) const
{
    if (_cache_ungapped.empty()) {
        (void) ungapped();
    }
    auto i = _cache_g2ug.find(index);
    if (i == _cache_g2ug.end())
        throw SeqIndexError("No corresponding ungapped position");
    return i->second;
}

char
Sequence::nucleic3to1(const ResName& rn)
{
    _1Letter_Map::const_iterator l1i = _nucleic3to1.find(rn);
    if (l1i == _nucleic3to1.end()) {
        return 'X';
    }
    return (*l1i).second;
}

Sequence&
Sequence::operator+=(const Sequence& addition)
{
    _clear_cache();
    _contents.insert(_contents.end(), addition.begin(), addition.end());
    return *this;
}

char
Sequence::protein3to1(const ResName& rn)
{
    _1Letter_Map::const_iterator l1i = _protein3to1.find(rn);
    if (l1i == _protein3to1.end()) {
        return 'X';
    }
    return (*l1i).second;
}

void
Sequence::push_front(char c)
{
    _clear_cache();
    Contents pushed;
    pushed.reserve(_contents.size()+1);
    pushed.push_back(c);
    pushed.insert(pushed.end(), _contents.begin(), _contents.end());
    pushed.swap(_contents);
}

char
Sequence::rname3to1(const ResName& rn)
{
    if (_rname3to1.empty())
        _init_rname_map();

    _1Letter_Map::const_iterator l1i = _rname3to1.find(rn);
    if (l1i == _rname3to1.end()) {
        return 'X';
    }
    return (*l1i).second;
}

std::vector<std::pair<int,int>>
Sequence::search(const std::string& pattern, bool case_sensitive) const
{
    // always ignores gap characters

    // C++ regex support doesn't handle overlapping matches, so
    // have to do that part "by hand"
    std::cmatch m;
    auto regex_flags = std::regex_constants::egrep;
    if (!case_sensitive)
        regex_flags |= std::regex_constants::icase;
    std::regex expr(pattern.c_str(), regex_flags);
    std::vector<std::pair<int,int>> results;
    Contents::size_type offset = 0;
    std::smatch match;
    auto& search_contents = ungapped();
    while (offset < search_contents.size()) {
        auto string_seq = std::string(search_contents.begin()+offset, search_contents.end());
        if (std::regex_search(string_seq, match, expr, std::regex_constants::match_not_null)) {
            results.emplace_back(match.position()+offset, match.length());
           offset += match.position()+1;
        } else {
            break;
        }
    }
    // remap to ungapped indices
    decltype(results) gapped_results;
    for (auto start_len: results) {
        auto ungapped_start = start_len.first;
        auto gapped_start  = ungapped_to_gapped(ungapped_start);
        size_t ungapped_end_index = ungapped_start + start_len.second;
        decltype(gapped_start) gapped_end = ungapped_end_index < search_contents.size() ?
            ungapped_to_gapped(ungapped_end_index) : size();
        gapped_results.emplace_back(gapped_start, gapped_end - gapped_start);
    }
    return gapped_results;
}

void
Sequence::session_restore(int version, int** ints, float**)
{
    auto& int_ptr = *ints;

    auto size = int_ptr[0];
    if (version > 2)
        _circular = int_ptr[1];
    int_ptr += SESSION_NUM_INTS(version);
    if (version < 3)
        // pre-version 3 had wrong number of ints declared, and
        // therefore arbitrarily skipped two int positions
        int_ptr += 2;

    _contents.reserve(size);
    for (decltype(size) i = 0; i < size; ++i) {
        _contents.push_back(*int_ptr++);
    }
}

void
Sequence::session_save(int** ints, float**) const
{
    auto& int_ptr = *ints;

    int_ptr[0] = _contents.size();
    int_ptr[1] = _circular;
    int_ptr += SESSION_NUM_INTS();

    for (auto c: _contents)
        *int_ptr++ = c;
}

void
Sequence::set_name(std::string& name)
{
    auto old_name = _name;
    _name = name;
    Py_XDECREF(py_call_method("_cpp_rename", "s", old_name.c_str()));
}

const Sequence::Contents&
Sequence::ungapped() const
{
    if (_cache_ungapped.empty()) {
        unsigned int ug_index = 0;
        auto gi = begin();
        for (unsigned int i = 0; gi != end(); ++gi, ++i) {
            auto c = *gi;
            if (std::isalpha(c) || c == '?') {
                _cache_ungapped.push_back(c);
                _cache_g2ug[i] = ug_index;
                _cache_ug2g[ug_index] = i;
                ug_index++;
            }
        }
    }
    return _cache_ungapped;
}

unsigned int
Sequence::ungapped_to_gapped(unsigned int index) const
{
    if (_cache_ungapped.empty()) {
        (void) ungapped();
    }
    return _cache_ug2g[index];
}

}  // namespace atomstruct
