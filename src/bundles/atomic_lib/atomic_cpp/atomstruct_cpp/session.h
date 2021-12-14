// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef atomstruct_session
#define atomstruct_session

// DON'T FORGET TO UPDATE Structure::copy() AS WELL!!
// ALSO: bump maxSessionVersion in atomic's bundle_info.xml
//
// Each class's SESSION_NUM... methods yield the number of those types that don't vary on
// a per-instance basis and are directly saved/restored by that class and not by a contained
// class such as Rgba.
#define CURRENT_SESSION_VERSION 17

#endif  // atomstruct_session
