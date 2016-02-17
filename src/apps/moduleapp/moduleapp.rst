..  vim: set expandtab shiftwidth=4 softtabstop=4:

===================
Module Applications
===================

ChimeraX modules can be easily turned into applications on Linux and Apple
Mac OS X with moduleapp's :download:`Makefile` and :download:`mod_app.sh.in`
shell script.  Just set **MOD_NAME** to the name of your Python module,
and include the **moduleapp/Makefile**.  Then **make install** will
install the module in the chimerax namespace and put a shell script that
invokes that script in ChimeraX bin directory (ChimeraX.app/Contents/bin
on Apple Mac OS X, ChimeraX/bin on Linux).

The **MOD_NAME** module is installed in the same chimerax namespace that
tool bundles are installed in, so the name should not be the same unless
the application is for the tool.

The script uses the ChimeraX application's ``-m`` option which is
like Python's ``-m`` option.

Code Organization
=================

Makefile
--------

If the module application's source is in
a parallel directory to the ChimeraX application,
then the Makefile is simply::

    MOD_NAME = application_name

    include ../moduleapp/Makefile

And the directory should be added to the parent's directory's Makefile,
so it will be installed when ChimeraX is built.

Module Source
-------------

If the code is simple, then the module can be in one Python file,
**MOD_NAME**.py, which is imported to run the application.
For complex modules, **MOD_NAME** will be a directory
and **MOD_NAME**/__main__.py is imported to run the application.
In the first case, to guard against accidently running the application
if imported outside the application, you should use the guard::

    if __name__ == `__main__':

for the code that actually runs the application.

Session Access
++++++++++++++

The ChimeraX session is available in the ``session`` variable.
If you pylint your code, add::

    session = session  # noqa

at the top of your file, so pylint will know that the ``session``
variable exists.

Application for Existing Tool
-----------------------------

There are several options for the case that an application name corresponds to
an existing tool.  The __main__.py file can be kept either in the tool
source, or in the application source in a subdirectory since the subdirectory's
files augment those that the tool installs first.
Regardless, the Makefile is just the standard module application Makefile.

Downloads
=========

* :download:`Makefile`  -- the included Makefile
* :download:`mod_app.sh.in` -- the shell script template
