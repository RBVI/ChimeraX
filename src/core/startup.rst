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

Custom Startup Python Code
**************************

To have ChimeraX run custom Python code each time you start it you can put Python files
in directory

	~/chimerax_start

Each Python file will be executed with the variable "session" added to the global namespace.
For example, the following line put in ~/chimerax_start/starttools.py automatically starts
the File History panel and Density Map toolbar.

        session.tools.start_tools(('File History', 'Density Map Toolbar'))

Python code can be used to register new commands, add mouse modes and file readers that you
develop.  In addition to executing Python files in the directory, the startup directory will
be appended to the Python sys.path search path so Python modules in the subdirectory can be
imported.  Subdirectories in the startup directory that contain an __init__.py file will be
imported and if they contain a function named "start" it will be called with session as
an argument.

To use a directory other than ~/chimerax_start as the startup directory set the environment
variable CHIMERAX_START to the desired directory in the shell where Chimera is started.
