#ifndef _mmcif_h
# define _mmcif_h
# if defined(_MSC_VER) && (_MSC_VER >= 1020)
#  pragma once
# endif

// include Python.h first so standard defines are the same
# define PY_SSIZE_T_CLEAN 1
# include <Python.h>
# include <new>

namespace mmcif {

extern void _mmcifError();
extern int _mmcifDebug;

} // namespace mmcif

#endif
