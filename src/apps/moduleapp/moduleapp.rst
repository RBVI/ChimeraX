..  vim: set expandtab shiftwidth=4 softtabstop=4:

===================
Module Applications
===================

Chimera modules can be easily turned into applications on Linux and Apple
Mac OS X with moduleapp's :download:`Makefile` and :download:`mod_app.sh.in`
shell script.  Just set **MOD_APP** to the name of your Python module,
and include the **moduleapp/Makefile**.  Then **make install** will
install the module in the chimerax namespace and put a shell script that
invokes that script in ChimeraX bin directory (ChimeraX.app/Contents/bin
on Apple Mac OS X, ChimeraX/bin on Linux).

The script uses the ChimeraX application's ``-m`` option.
