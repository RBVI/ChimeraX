// vi: set expandtab ts=4 sw=4:
#include <ctype.h>
#include "cmp_nocase.h"

namespace util {

int
cmp_nocase(const std::string &s, const std::string &s2)
{
    std::string::const_iterator p = s.begin();
    std::string::const_iterator p2 = s2.begin();

    while (p != s.end() && p2 != s2.end()) {
        char c = toupper(*p);
        char c2 = toupper(*p2);
        if (c != c2)
            return c < c2 ? -1 : 1;
        ++p;
        ++p2;
    }
    return s2.size() - s.size();
}

}  // namespace util
