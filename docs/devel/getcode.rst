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

Obtaining Code
==============

Most of ChimeraX is written in Python and that code is included in the ChimeraX distribution::

	Windows: ChimeraX/bin/Lib/site-packages/chimerax
	macOS: ChimeraX.app/Contents/lib/python3.7/site-packages/chimerax
	Linux: chimerax/lib/python3.7/site-packages/chimerax

Small modifications to the code can be tested by simply editing the Python code and restarting ChimeraX.

Git Repository
--------------

The Python and C++ ChimeraX source code is available at the `ChimeraX GitHub repository <https://github.com/RBVI/ChimeraX/>`_.

How to use Git
--------------
Here are a minimal set of commands to get started using git:

    #. Git associates a user name and email address with all check-ins.
       The defaults are based on your :envvar:`USERNAME` and your computer's
       hostname.
       The email address is usually wrong.
       To explicitly set those values::

            git config --global user.name "Your Name"
            git config --global user.email you@example.com

    #. Make local copy of repository::

        git clone https://github.com/RBVI/ChimeraX.git

    #. Use the develop branch (the master branch is only used for releases)::

	git switch develop
	 
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