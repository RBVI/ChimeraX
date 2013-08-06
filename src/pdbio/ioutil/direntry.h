#ifndef ioutil_direntry_h
# define ioutil_direntry_h

# if !defined(_WIN32)
#  include <dirent.h>
# else
#  include "win32_dirent.h"
# endif

#endif
