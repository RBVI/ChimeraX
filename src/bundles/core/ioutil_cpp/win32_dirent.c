/*
 * Modified from glut-3.7/progs/demos/walker/win32_dirent.h
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

#define WIN32_LEAN_AND_MEAN
#define STRICT
#include <windows.h>

#include "win32_dirent.h"

/* For Win32 that lacks Unix direct support. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct _dir {
	WIN32_FIND_DATA wfd;
	HANDLE hFind;
	struct dirent de;
};

DIR *
opendir(const char *pSpec)
{
#ifdef __cplusplus
	DIR *pDir = new DIR;
#else
	DIR *pDir = malloc(sizeof (DIR));
#endif
	/* XXX Windows 95 has problems opening up "." though Windows NT does
	 * this fine?  Open "*" instead of "." to be safe. -mjk */
	/* Add error-checking for non-existent directories. -gsc
	 * FindFirstFile really wants a filename wildcard. -gsc */
	int len = strlen(pSpec);
#ifdef __cplusplus
	char *fname = new char[len + 3];
#else
	char *fname = malloc(len + 3);
#endif
	if (len == 1 && pSpec[0] == '.')
		strcpy(fname, "*");
	else
		sprintf(fname, "%s\\*", pSpec);
	pDir->hFind = FindFirstFile(fname, &pDir->wfd);
#ifdef __cplusplus
	delete [] fname;
#else
	free(fname);
#endif
	if (pDir->hFind == INVALID_HANDLE_VALUE) {
#ifdef __cplusplus
		delete pDir;
#else
		free(pDir);
#endif
		return NULL;
	}
	return pDir;
}

void
closedir(DIR *pDir)
{
	if (pDir->hFind)
		FindClose(pDir->hFind);
#ifdef __cplusplus
	delete pDir;
#else
	free(pDir);
#endif
}

struct dirent *
readdir(DIR *pDir)
{
	if (pDir->hFind == NULL)
		return NULL;
	strncpy(pDir->de.d_name, pDir->wfd.cFileName, MAX_PATH);
	if (!FindNextFile(pDir->hFind, &pDir->wfd)) {
		FindClose(pDir->hFind);
		pDir->hFind = NULL;
	}
	return &pDir->de;
}
