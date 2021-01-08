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

Code Conventions
================

Coding Style
------------

ChimeraX uses Python 3.

Python code should follow the Python Style Guide: :pep:`8`.

Documentation Strings should follow Python's documentation style
given in `Chapter 7 <http://docs.python.org/devguide/documenting.html>`_
of the `Python Developer's Guide <http://docs.python.org/devguide/index.html>`_.
So use `reStructuredText (reST) as extended by Sphinx <http://sphinx-doc.org/latest/rest.html>`_.

Editor defaults
---------------

From <http://wiki.python.org/moin/Vim>:
All python files should have the following modeline at the top:

    # vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4:

But modelines are a security risk, so put:

    au FileType python setlocal tabstop=8 expandtab shiftwidth=4 softtabstop=4

in your .vimrc as well.
