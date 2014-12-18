..  vim: set expandtab shiftwidth=4 softtabstop=4:

Code Conventions
================

Code Repository
---------------

Chimera2 uses `git <http://git-scm.com/>`_ for source code management.

Here are a minimal set of commands to get started using git:

    #. Git associates a user name and email address with all check-ins.
       The defaults are based on your :envvar:`USERNAME` and your computer's
       hostname.
       The email address is usually wrong.
       To explicitly set those values::

            git config --global user.name "Your Name"
            git config --global user.email you@example.com

    #. Make local copy of repository::

        git clone ssh://plato.cgl.ucsf.edu/usr/local/projects/chimera2/git/chimera2.git

    #. Work on development branch::

        git checkout develop

    #. Update repository to latest version::

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

Python code should follow the Python Style Guide: :pep:`8`.

Use new-style classes.

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
