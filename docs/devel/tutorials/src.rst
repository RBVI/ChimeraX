``src`` is the folder containing the source code for the
Python package that implements the bundle functionality.
The ChimeraX ``devel`` command, used for building and
installing bundles, automatically includes all ``.py``
files in ``src`` as part of the bundle.  (Additional
files may also be included using bundle information tags
such as ``DataFiles`` as shown in :doc:`tutorials_tool`.)
The only required file in ``src`` is ``__init__.py``.
Other ``.py`` files are typically arranged to implement
different types of functionality.  For example, ``cmd.py``
is used for command-line commands; ``tool.py`` or ``gui.py``
for graphical interfaces; ``io.py`` for reading and saving
files, etc.
