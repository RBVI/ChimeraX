// vim: set expandtab ts=4 sw=4:

#include <algorithm>

#include "Chain.h"
#include "Sequence.h"
#include "Residue.h"

namespace atomstruct {

void
Chain::bulk_set(Chain::Residues& residues, Sequence::Contents* chars)
{
    bool del_chars = chars == nullptr;
    if (del_chars) {
        chars = new Sequence::Contents(residues.size());
        auto chars_ptr = chars->begin();
        for (auto ri = residues.begin(); ri != residues.end();
        ++ri, ++chars_ptr) {
            (*chars_ptr) = Sequence::rname3to1((*ri)->name());
        }
    }
    _residues = residues;
    this->_sequence = *chars;

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
        this->_sequence.push_back(c);
    } else {
        _residues.at(i) = r;
        this->_sequence.at(i) = c;
    }
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
            Chain::Residues new_residues;
            Sequence::Contents new_contents;
            auto ri = _residues.begin();
            for (auto si = _sequence.begin(); si != _sequence.end();
            ++si, ++ri) {
                if (*ri == nullptr)
                    continue;
                new_residues.push_back(*ri);
                new_contents.push_back(*si);
            }
            _residues.swap(new_residues);
            _sequence.swap(new_contents);
        }
    }
    _from_seqres = fs;
}

}  // namespace atomstruct
