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

When you first open a new IDE, it will probably try to set its
Python interpreter to the system Python. It is better to use a virtual
environment based on ChimeraX's internal Python as the Python interpreter
instead.

If you set ChimeraX's internal interpreter directly as your IDE interpreter,
the IDE may become confused and report that it cannot find certain modules or
that certain modules which are present in the ChimeraX distribution are not
actually installed.

Virtual environments solve both problems, and many others. You can install
packages into the virtual environment without corrupting the internal Python
environment, and trash the virtual environment and start again if something
goes wrong.

First, build ChimeraX or obtain a built ChimeraX. If you have built
ChimeraX, then you can use the makefile to create a virtual environment
with the correct settings in your repository directory: ::

    make venv

If you are working off of the ChimeraX distribution, use the internal Python
to create a virtual environment in your project's folder. The internal
Python can be found at the following locations:

* Windows: ``C:\path\to\ChimeraX.app\bin\python.exe``
* macOS: ``/path/to/ChimeraX.app/Contents/bin/python``
* Linux: ``/path/to/ChimeraX.app/bin/python``

Ensure that the virtual environment has access to ChimeraX's ``site-packages``
directory, and then set that virtual environment's Python as the project's
Python interpreter in your IDE.

Example commands: ::

    # Windows
    C:\path\to\ChimeraX.app\bin\python.exe -m venv .venv --system-site-packages
    # macOS
    /path/to/ChimeraX.app/Contents/bin/python -m venv .venv --system-site-packages
    # Linux
    /path/to/ChimeraX.app/bin/python -m venv .venv --system-site-packages

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
