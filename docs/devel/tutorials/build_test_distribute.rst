.. include:: references.rst

Building and Testing Bundles
============================

To build a bundle, start ChimeraX and execute the command:

``devel build PATH_TO_SOURCE_CODE_FOLDER``

Python source code and other resource files are copied
into a ``build`` sub-folder below the source code
folder.  C/C++ source files, if any, are compiled and
also copied into the ``build`` folder.
The files in ``build`` are then assembled into a
Python wheel in the ``dist`` sub-folder.
**The file with the** ``.whl`` **extension in the** ``dist``
**folder is the ChimeraX bundle.**

To test the bundle, execute the ChimeraX command:

``devel install PATH_TO_SOURCE_CODE_FOLDER``

This will build the bundle, if necessary, and install
the bundle in ChimeraX.  Bundle functionality should
be available immediately.

To remove temporary files created while building
the bundle, execute the ChimeraX command:

``devel clean PATH_TO_SOURCE_CODE_FOLDER``

Some files, such as the bundle itself, may still remain
and need to be removed manually.

Building bundles as part of a batch process is straightforward,
as these ChimeraX commands may be invoked directly
by using commands such as:

``ChimeraX --nogui --exit --cmd 'devel install PATH_TO_SOURCE_CODE_FOLDER exit true'``

This example executes the ``devel install`` command without
displaying a graphics window (``--nogui``) and exits immediately
after installation (``exit true``).  The initial ``--exit``
flag guarantees that ChimeraX will exit even if installation
fails for some reason.


Distributing Bundles
====================

With ChimeraX bundles being packages as standard Python
wheel-format files, they can be distributed as plain files
and installed using the ChimeraX ``toolshed install``
command.  Thus, electronic mail, web sites and file
sharing services can all be used to distribute ChimeraX
bundles.

Private distributions are most useful during bundle
development, when circulation may be limited to testers.
When bundles are ready for public release, they can be
published on the `ChimeraX Toolshed`_, which is designed
to help developers by eliminating the need for custom
distribution channels, and to aid users by providing
a central repository where bundles with a variety of
functionality may be found.

Customizable information for each bundle on the toolshed
includes its description, screen captures, authors,
citation instructions and license terms.
Automatically maintained information
includes release history and download statistics.

To submit a bundle for publication on the toolshed,
you must first sign in.  Currently, only Google
sign in is supported.  Once signed in, use the
``Submit a Bundle`` link at the top of the page
to initiate submission, and follow the instructions.
The first time a bundle is submitted to the toolshed,
it is held for inspection by the ChimeraX team, which
may contact the authors for more information.
Once approved, all subsequent submissions of new
versions of the bundle are posted immediately on the site.
