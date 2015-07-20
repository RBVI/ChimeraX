// vi: set expandtab ts=4 sw=4:
#ifndef atomstruct_string_types
#define atomstruct_string_types

#include <chutil/CString.h>

namespace atomstruct {

using chutil::CString;

// len param includes null
typedef CString<5>  AtomName;
typedef CString<5>  AtomType;
typedef CString<5>  ChainID;
typedef CString<5>  ResName;

}  // namespace atomstruct

#endif  // atomstruct_string_types
