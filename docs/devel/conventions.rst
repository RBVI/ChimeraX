..  vim: set expandtab shiftwidth=4 softtabstop=4:

..
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

Code Conventions
================

Style
-----
Python code should follow the Python Style Guide: :pep:`8`.

Documentation Strings should follow Python's documentation style
given in `Chapter 7 <https://docs.python.org/devguide/documenting.html>`_
of the `Python Developer's Guide <https://docs.python.org/devguide/index.html>`_.
Use `reStructuredText (reST) as extended by Sphinx <https://sphinx-doc.org/latest/rest.html>`_.

The docstrings of :class:`.MainToolWindow` and :py:meth:`.MainToolWindow.create_child_window()`
can serve as reference docstrings. Specifying an argument's type is not necessary; if :pep:`484`
style type hints are used, Sphinx (through the ``spinx_autodoc_typehints`` extension) will pick
them up automatically.

Editor Defaults
---------------
.. From <https://wiki.python.org/moin/Vim>:

All python files should have the following modeline at the top: ::

    # vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4:

But modelines are a security risk, so put: ::

    au FileType python setlocal tabstop=8 expandtab shiftwidth=4 softtabstop=4

in your .vimrc as well.

Line Endings
------------
The ChimeraX git repository uses line ending normalization. On checkout, the majority
of files will have LF line endings. Use any editor in any configuration; line endings
in mixed files or CRLF files will be converted to LF on check-in except as specified
in ``.gitattributes``, which you may edit to protect any file that must have its
original line endings.

If you are comfortable, you can set ``core.safecrlf`` to ``false`` in your
``~/.gitconfig`` in order to ignore routine normalization warnings from ``git``
when using CRLF line endings on Windows.

Wheel API Control
-----------------
The canonical way to guard code against being imported in the wheel is to import
``runtime_env_is_chimerax_app`` from ``chimerax.core`` and put app-specific code under
a guard: ::

    if runtime_env_is_chimerax_app():
        # app-specific code

This can be used, among other things, to ensure that when users use tab completion on
modules, what's shown to be available to them isn't polluted by code that won't work
outside the application. If you find yourself feeling the urge to guard code inside a
function, consider how the function could be refactored into an app-specific function
that calls a general one.
