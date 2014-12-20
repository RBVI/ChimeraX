// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Sequence
#define atomstruct_Sequence

#include <vector>
#include <unordered_map>
#include <string>
#include "imex.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX Sequence {
public:
    typedef std::vector<unsigned char> Contents;
protected:
    typedef std::unordered_map<std::string, unsigned char>  _1Letter_Map;
    static void  _init_rname_map();
    static _1Letter_Map  _nucleic3to1;
    static _1Letter_Map  _protein3to1;
    static _1Letter_Map  _rname3to1;
    Contents  _sequence;
public:
    static void  assign_rname3to1(const std::string& rname, unsigned char let,
        bool protein);
    Contents::const_iterator  begin() { return _sequence.begin(); }
    Contents::const_iterator  end() { return _sequence.end(); }
    unsigned char&  operator[](unsigned i) { return _sequence[i]; }
    unsigned char  operator[](unsigned i) const { return _sequence[i]; }
    static unsigned char  nucleic3to1(const std::string &rn);
    static unsigned char  protein3to1(const std::string &rn);
    static unsigned char  rname3to1(const std::string &rn);
    Sequence() {}
    Sequence(Contents& c): _sequence(c) {};
    Sequence(const std::vector<std::string>& res_names);  // 3-letter codes
    virtual  ~Sequence() {}
    const Contents&  sequence() const { return _sequence; }
    Contents::size_type  size() const { return _sequence.size(); }
};

}  // namespace atomstruct

#endif  // atomstruct_Sequence
