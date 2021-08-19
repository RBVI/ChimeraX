..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

map_fit: Fit atomic models in density maps
==========================================

Atomic models or density maps can be fit in density maps.
A local gradient ascent method is used to maximize the average density value
at atom positions or the correlation between two maps.


Map fitting
-----------

 * :func:`.fitcmd.fitmap` - optimize fit of atoms or map in another map

Functions
---------

.. autofunction:: chimerax.map_fit.fitcmd.fitmap
  :noindex:
