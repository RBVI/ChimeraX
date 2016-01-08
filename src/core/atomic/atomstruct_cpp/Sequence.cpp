// vi: set expandtab ts=4 sw=4:
#include <cctype>
#include "Sequence.h"

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
        {"URA", 'U'}
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

Sequence::Sequence(const std::vector<ResName>& res_names) // 3-letter codes
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
    return _cache_g2ug[index];
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
Sequence::push_front(unsigned char c)
{
    _clear_cache();
    Contents pushed(_contents.size()+1);
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

void
Sequence::session_restore(int version, int** ints, float**)
{
    auto& int_ptr = *ints;

    auto size = int_ptr[0];
    int_ptr += SESSION_NUM_INTS(version);

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
    int_ptr += SESSION_NUM_INTS();

    for (auto c: _contents)
        *int_ptr++ = c;
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
