// vi: set expandtab ts=4 sw=4:

#include <algorithm>

#include "Chain.h"
#include "Sequence.h"
#include "Residue.h"

namespace atomstruct {

void
Chain::bulk_set(Chain::Residues& residues, std::vector<unsigned char>* chars)
{
    bool del_chars = chars == nullptr;
    if (del_chars) {
        chars = new std::vector<unsigned char>(residues.size());
        auto chars_ptr = chars->begin();
        for (auto ri = residues.begin(); ri != residues.end();
        ++ri, ++chars_ptr) {
            (*chars_ptr) = Sequence::rname3to1((*ri)->name());
        }
    }
    _residues = residues;
    assign(chars->begin(), chars->end());
    _res_map.clear();
    int i = 0;
    for (auto ri = residues.begin(); ri != residues.end(); ++ri, ++i) {
        if (*ri != nullptr)
            _res_map[*ri] = i;
    }

    if (del_chars)
        delete chars;
}

void 
Chain::set(unsigned i, Residue *r, char character)
{
    unsigned char c;
    if (character < 0) {
        c = Sequence::rname3to1(r->name());
    } else {
        c = (unsigned char)character;
    }
    if (i == _residues.size()) {
        _residues.push_back(r);
        push_back(c);
    } else {
        auto& res_at_i = _residues.at(i);
        if (res_at_i != nullptr)
            _res_map.erase(res_at_i);
        res_at_i = r;
        at(i) = c;
    }
    if (r != nullptr)
        _res_map[r] = i;
}

void
Chain::set_from_seqres(bool fs)
{
    if (fs == _from_seqres)
        return;
    if (_from_seqres) {
        // changing from true to false;
        // eliminate seqres parts of sequence...
        if (std::find(_residues.begin(), _residues.end(), nullptr)
        != _residues.end()) {
            // there actually are seqres portions
            _res_map.clear();
            Chain::Residues new_residues;
            Sequence::Contents new_contents;
            auto ri = _residues.begin();
            int i = 0;
            for (auto si = begin(); si != end(); ++si, ++ri) {
                if (*ri == nullptr)
                    continue;
                _res_map[*ri] = ++i;
                new_residues.push_back(*ri);
                new_contents.push_back(*si);
            }
            _residues.swap(new_residues);
            swap(new_contents);
        }
    }
    _from_seqres = fs;
}

}  // namespace atomstruct
