..  vim: set expandtab shiftwidth=4 softtabstop=4:

..
    === UCSF ChimeraX Copyright ===
    Copyright 2017 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

##################
IDEs and Debugging
##################
ChimeraX is a mostly Python project and ships mostly Python code alongside
C++ extensions. There are over 260 packages in the distributed ChimeraX,
including over 150 bundles. This is an unwieldy amount of code for any one
person to keep in their head all at once. The use of an IDE, which can
provide code completion and in-editor documentation, is highly encouraged.

====
IDEs
====
Popular IDEs include PyCharm and VS Code. Your choice of IDE is up to you;
many share the same feature set thanks to the *language server protocol* (LSP)
and the *debug adapter protocol* (DAP).

You may set ChimeraX's internal Python interpreter as the interpreter for your IDE:

* Windows: ``C:\path\to\your\repo\ChimeraX.app\bin\python.exe``
* macOS: ``/path/to/your/repo/ChimeraX.app/Contents/bin/python``
* Linux: ``/path/to/your/repo/ChimeraX.app/bin/python``

But this may confuse your IDE, which may report that it cannot find certain
modules. The recommended way to access ChimeraX's Python environment in an
IDE is to use the internal python to create a virtual environment, give
that virtual environment access to ChimeraX's ``site-packages`` directory, and
then set that virtual environment's Python as the project's Python interpreter
instead.

=========
Debugging
=========
.. TODO: nogui debugging

Since PyCharm uses its own wrapper around pydevd instead of debugy, we'll start
with steps for PyCha.

* Create a new Run/Debug Configuration
* Set the script path field to point at a module instead
* Set the module to "chimerax.core"
* Set the Python interpreter to ChimeraX's internal Python
* Uncheck "Add content roots to PYTHONPATH"
* Uncheck "Add source roots to PYTHONPATH"

We recommend the following profile for programs using debugpy as the interface
to pydevd (the majority of them). ::

    {
        "name": "Launch ChimeraX",
        "type": "python",
        "request": "launch",
        "module": "chimerax.core",
        # Ensure we can pause ChimeraX, not just break in our module
        "justMyCode": false
    }

Breaking on raise leads to slow execution in the debugger, but breaking on an
uncaught exception or a user uncaught exception provides a tight enough net
to catch errors in ChimeraX code.

ChimeraX spends the majority of its time in its GUI event loop. Unless you set
a breakpoint in your bundle then upon pausing execution you will likely be dropped
into the context of the GUI module.

There, the GUI module's reference to the program's session will be available so
that you can get an idea of what's going on in your debugging session.
