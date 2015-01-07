// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_Sequence
#define atomstruct_Sequence

#include <vector>
#include <unordered_map>
#include <string>
#include "imex.h"

namespace atomstruct {

class ATOMSTRUCT_IMEX Sequence: public std::vector<unsigned char> {
protected:
    typedef std::unordered_map<std::string, unsigned char>  _1Letter_Map;
    static void  _init_rname_map();
    static _1Letter_Map  _nucleic3to1;
    static _1Letter_Map  _protein3to1;
    static _1Letter_Map  _rname3to1;
public:
    static void  assign_rname3to1(const std::string& rname, unsigned char let,
        bool protein);
    static unsigned char  nucleic3to1(const std::string &rn);
    static unsigned char  protein3to1(const std::string &rn);
    static unsigned char  rname3to1(const std::string &rn);
    Sequence() {}
    Sequence(const std::vector<std::string>& res_names);  // 3-letter codes
    virtual  ~Sequence() {}
};

}  // namespace atomstruct

#endif  // atomstruct_Sequence
