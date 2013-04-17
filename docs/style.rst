
Coding Style
~~~~~~~~~~~~

Python code should follow the Python Style Guide: :pep:`8`.

Use new-style classes.

Documentation Strings should follow Python's documentation style
given in `Chapter 7 <http://docs.python.org/devguide/documenting.html>`_
of the `Python Developer's Guide <http://docs.python.org/devguide/index.html>`_.
So use `reStructuredText (reST) as extended by Sphinx <http://sphinx-doc.org/latest/rest.html>`_.

From <http://wiki.python.org/moin/Vim>:
All python files should have the following modeline at the top:

    # vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

But modelines are a security risk, so put:

    au FileType python setlocal tabstop=8 expandtab shiftwidth=4 softtabstop=4

in your .vimrc as well.
