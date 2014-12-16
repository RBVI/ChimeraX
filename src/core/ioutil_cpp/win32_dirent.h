#ifndef ioutil_win32_dirent_h
#define ioutil_win32_dirent_h

/*
 * Extracted from glut-3.7/progs/demos/walker/win32_dirent.h
 *
 * Copyright notice from glut-3.7/NOTICE:
 *
 * NOTICE:  The OpenGL Utility Toolkit (GLUT) distribution contains source
 * code published in a book titled "Programming OpenGL for the X Window
 * System" (ISBN: 0-201-48359-9) published by Addison-Wesley.  The
 * programs and associated files contained in the distribution were
 * developed by Mark J. Kilgard and are Copyright 1994, 1995, 1996 by Mark
 * J. Kilgard (unless otherwise noted).  The programs are not in the
 * public domain, but they are freely distributable without licensing
 * fees.  These programs are provided without guarantee or warrantee
 * expressed or implied.
 */

/* For Win32 that lacks Unix direct support. */

# ifdef __cplusplus
extern "C" {
# endif

/* To avoid Windows namespace pollution we define MAX_PATH here
 * and hide the definition of DIR.
 */
# ifndef MAX_PATH
#  include <stdlib.h>
#  ifndef MAX_PATH
#   define MAX_PATH _MAX_PATH
#  endif
# endif

struct dirent {
	char d_name[MAX_PATH];
};

typedef struct _dir DIR;

DIR *opendir(const char *pSpec);
void closedir(DIR * pDir);
struct dirent *readdir(DIR *pDir);

# ifdef __cplusplus
} // extern "C"
# endif

#endif /* __win32_dirent__ */
