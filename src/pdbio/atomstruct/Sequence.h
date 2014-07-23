// vim: set expandtab ts=4 sw=4:
#ifndef atomic_Sequence
#define atomic_Sequence

#include <vector>
#include <unordered_map>
#include <string>
#include "imex.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX Sequence {
public:
    typedef std::vector<unsigned char> Contents;
protected:
    Contents  _sequence;
    typedef std::unordered_map<std::string, unsigned char>  _1Letter_Map;
    static _1Letter_Map _rname3to1;

public:
    static void  assign_rname3to1(const std::string& rname, unsigned char let)
        { _rname3to1[rname] = let; }
    unsigned char&  operator[](unsigned i) { return _sequence[i]; }
    unsigned char  operator[](unsigned i) const { return _sequence[i]; }
    static unsigned char  rname3to1(const std::string &rn);
    Sequence();
    const Contents&  sequence() const { return _sequence; }
};

}  // namespace atomstruct

#endif  // atomic_Sequence
