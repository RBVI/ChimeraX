// vim: set expandtab ts=4 sw=4:
#include "Sequence.h"

namespace atomstruct {

Sequence::_1Letter_Map Sequence::_rname3to1;

unsigned char
Sequence::rname3to1(const char *rn)
{
    if (_rname3to1.size() == 0) {
        _rname3to1["A"] = 'A';
        _rname3to1["+A"] = 'A';
        _rname3to1["ADE"] = 'A';
        _rname3to1["DA"] = 'A';
        _rname3to1["C"] = 'C';
        _rname3to1["+C"] = 'C';
        _rname3to1["CYT"] = 'C';
        _rname3to1["DC"] = 'C';
        _rname3to1["G"] = 'G';
        _rname3to1["+G"] = 'G';
        _rname3to1["GUA"] = 'G';
        _rname3to1["DG"] = 'G';
        _rname3to1["T"] = 'T';
        _rname3to1["+T"] = 'T';
        _rname3to1["THY"] = 'T';
        _rname3to1["DT"] = 'T';
        _rname3to1["U"] = 'U';
        _rname3to1["+U"] = 'U';
        _rname3to1["URA"] = 'U';
        _rname3to1["ALA"] = 'A';
        _rname3to1["ARG"] = 'R';
        _rname3to1["ASH"] = 'D'; // Amber (protonated ASP)
        _rname3to1["ASN"] = 'N';
        _rname3to1["ASP"] = 'D';
        _rname3to1["ASX"] = 'B'; // ambiguous ASP/ASN
        _rname3to1["CYS"] = 'C';
        _rname3to1["CYX"] = 'C'; // Amber (disulphide)
        _rname3to1["GLH"] = 'E'; // Amber (protonated GLU)
        _rname3to1["GLU"] = 'E';
        _rname3to1["GLN"] = 'Q';
        _rname3to1["GLX"] = 'Z'; // ambiguous GLU/GLN
        _rname3to1["GLY"] = 'G';
        _rname3to1["HID"] = 'H'; // Amber (delta protonated)
        _rname3to1["HIE"] = 'H'; // Amber (epsilon protonated)
        _rname3to1["HIP"] = 'H'; // Amber (doubly protonated)
        _rname3to1["HIS"] = 'H';
        _rname3to1["HYP"] = 'P'; // hydroxyproline, which in theory
            // has its own letter ('O') but using that is problematic
            // with similarity matrices
        _rname3to1["ILE"] = 'I';
        _rname3to1["LEU"] = 'L';
        _rname3to1["LYS"] = 'K';
        _rname3to1["MET"] = 'M';
        _rname3to1["MSE"] = 'M'; // Selenomethionine
        _rname3to1["PHE"] = 'F';
        _rname3to1["PRO"] = 'P';
        _rname3to1["SER"] = 'S';
        _rname3to1["THR"] = 'T';
        _rname3to1["TRP"] = 'W';
        _rname3to1["TYR"] = 'Y';
        _rname3to1["VAL"] = 'V';
    }
    _1Letter_Map::const_iterator l1i = _rname3to1.find(rn);
    if (l1i == _rname3to1.end())
        return 'X';
    return (*l1i).second;
}

}  // namespace atomstruct
