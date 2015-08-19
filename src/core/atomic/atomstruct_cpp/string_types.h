// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_string_types
#define atomstruct_string_types

#include <chutil/CString.h>

namespace atomstruct {

using chutil::CString;

// len param includes null
typedef CString<5, 'A', 't', 'o', 'm', ' ', 'N', 'a', 'm', 'e'>  AtomName;
typedef CString<5, 'A', 't', 'o', 'm', ' ', 'T', 'y', 'p', 'e'>  AtomType;
typedef CString<5, 'C', 'h', 'a', 'i', 'n', ' ', 'I', 'D'>  ChainID;
typedef CString<5, 'R', 'e', 's', 'i', 'd', 'u', 'e', ' ', 'n', 'a', 'm', 'e'>  ResName;

}  // namespace atomstruct

#endif  // atomstruct_string_types
