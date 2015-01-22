// vi: set expandtab ts=4 sw=4:
#ifndef util_cmp_nocase
# define util_cmp_nocase

# include <string>
# include "imex.h"

namespace chutil {

CHUTIL_IMEX extern int cmp_nocase(const std::string &s, const std::string &s2);

}  // namespace chutil

#endif  // util_cmp_nocase
