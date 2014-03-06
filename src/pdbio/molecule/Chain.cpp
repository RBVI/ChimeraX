// vim: set expandtab ts=4 sw=4:
#include "Chain.h"
#include "Sequence.h"
#include "Residue.h"

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
Chain::bulk_set(std::vector<Residue *> residues, Sequence::Contents* chars)
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
