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

Code Repository
---------------

ChimeraX uses `git <http://git-scm.com/>`_ for source code management.

Here are a minimal set of commands to get started using git:

    #. Git associates a user name and email address with all check-ins.
       The defaults are based on your :envvar:`USERNAME` and your computer's
       hostname.
       The email address is usually wrong.
       To explicitly set those values::

            git config --global user.name "Your Name"
            git config --global user.email you@example.com

    #. Make local copy of repository (currently we only use the develop branch)::

        git clone --depth 1 --single-branch --branch develop plato.cgl.ucsf.edu:/usr/local/projects/chimerax/git/chimerax.git

    #. To update repository to latest version::

        git pull

    #. Add current state of file to the repository::

        git add "filename(s)"

    #. Commit all changes to repository (added files and changes to those added files)::

        git commit -a

    #. Copy local repository changes to master repository::

        git push

    #. Diff from previous to current revision of file (ignores additions)::

        git whatchanged -n 1 -p <file>

    #. Diff to previous commit of file::

        git diff HEAD^ <file>

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
