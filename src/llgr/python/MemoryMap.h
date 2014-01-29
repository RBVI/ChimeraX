#ifndef Chimera_extra_h
# define Chimera_extra_h

# if defined(_MSC_VER) && (_MSC_VER >= 1020)
#  pragma once
# endif

# include "_llgr.h"

extern "C" {
typedef struct _object PyObject;
}

namespace llgr {

# ifndef WrapPy
LLGR_IMEX extern PyObject *memory_map(unsigned char *data, Py_ssize_t len);
# else
LLGR_IMEX extern PyObject *memory_map(size_t addr, Py_ssize_t len);
# endif

} // namespace llgr

#endif
