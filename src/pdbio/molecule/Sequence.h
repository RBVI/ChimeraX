// vim: set expandtab ts=4 sw=4:
#ifndef molecule_Sequence
#define molecule_Sequence

#include <vector>
#include <map>
#include <string>
#include "imex.h"

class MOLECULE_IMEX Sequence {
public:
    typedef std::vector<unsigned char> Contents;
protected:
    Contents  _sequence;
    typedef std::map<const char *, unsigned char>  _1Letter_Map;
    static _1Letter_Map _rname3to1;

public:
    unsigned char&  operator[](unsigned i) { return _sequence[i]; }
    unsigned char  operator[](unsigned i) const { return _sequence[i]; }
    static unsigned char  rname3to1(const char *);
    static unsigned char  rname3to1(const std::string &rn)
        { return rname3to1(rn.c_str()); }
    Sequence();
    const Contents&  sequence() const { return _sequence; }
};

#endif  // molecule_Sequence
