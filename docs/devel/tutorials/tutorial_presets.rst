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

.. _Bundle Example\: Define Presets:


==============================
Bundle Example: Define Presets
==============================

This example describes how to create a ChimeraX bundle
that defines presets that will appear in the ChimeraX
Presets menu and be usable in the ``preset`` command.

The steps in implementing the bundle are:

#. Create a ``bundle_info.xml`` containing information
   about the bundle,
#. Create a Python package that interfaces with ChimeraX
   and implements the preset functionality, and
#. Install and test the bundle in ChimeraX.

The final step builds a Python wheel that ChimeraX uses
to install the bundle.  So if the bundle passes testing,
it is immediately available for sharing with other users.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_preset>`_
containing a folder named `tut_preset`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_preset`` folder are:

- ``tut_preset`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``presets.py`` - source code to define/execute presets

The file contents are shown below.


``bundle_info.xml``
-------------------

``bundle_info.xml`` is an `eXtensible Markup Language`_
format file whose tags are listed in :doc:`bundle_info`.
While there are many tags defined, only a few are needed
for bundles written completely in Python.  The
``bundle_info.xml`` in this example is similar to the one
from the :doc:`tutorial_tool` example with changes highlighted.
For explanations of the unhighlighted sections, please
see :doc:`tutorial_hello`, :doc:`tutorial_command` and
:doc:`tutorial_tool`.

.. literalinclude:: ../../../src/examples/tutorials/tut_preset/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,19-23,26-28,31-34,36-39

The ``BundleInfo``, ``Synopsis``, ``Description`` and ``Category`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10, 19-23, and 26-28).

Since the presets use functionality from the PresetMgr bundle, the ``Dependencies``
section has been changed to reflect that (lines 31-34).

The ``Providers`` section on lines 36-39 informs ChimeraX that
this bundle defines two presets, named "thin sticks"  and "ball and stick",
and that their category is "small molcule" for organizing them in the Presets
menu and for the ``category`` argument of the
`preset command <../../user/commands/preset.html>`_.
More details about the ``Providers`` section for presets can be found
in :ref:`Defining Presets`.

There are also deletions relative to previous examples.  For instance,
there are no ``DataFiles`` or ``ChimeraXClassifier`` tags since this
bundle does not provide a help page nor does it implement a tool.


``src``
-------

.. include:: src.rst


``src/__init__.py``
-------------------

As described in :doc:`tutorial_hello`, ``__init__.py`` contains
the initialization code that defines the ``bundle_api`` object
that ChimeraX needs in order to invoke bundle functionality.
ChimeraX expects ``bundle_api`` class to be derived from
:py:class:`chimerax.core.toolshed.BundleAPI` with methods
overridden for registering commands, tools, etc.

.. literalinclude:: ../../../src/examples/tutorials/tut_preset/src/__init__.py
    :language: python
    :linenos:

The :py:meth:`run_provider` method is called by the presets manager (``session.presets``)
whenever one of the presets your bundle provides is requested, typically via the Presets
menu or the `preset command <../../user/commands/preset.html>`_.

The arguments to :py:meth:`run_provider`
are ``session``, the current :py:class:`chimerax.core.session.Session` instance,
``name``, the name of the preset to execute,
``mgr``, a :py:class:`~chimerax.preset_mgr.manager.PresetsManager` instance (*a.k.a.* ``session.presets``),
and ``kw``, a keyword dictionary which is always empty in the case of presets.
Since all managers that your bundle offers ``Providers`` for call this method,
other calls to this method may have different ``mgr`` and ``kw`` argument values
(and the run_provider code would have to be more complex).
Since this example bundle only provides presets, it simply calls its ``run_preset``
function to execute the requested preset, which is discussed below.


``src/presets.py``
-------------------

``presets.py`` defines the function :py:func:`run_preset`
that is invoked (from :py:meth:`run_provider`, above) when either of our presets are requested.

.. literalinclude:: ../../../src/examples/tutorials/tut_preset/src/presets.py
    :language: python
    :linenos:

The :py:func:`run_preset` function needs to in turn call
the preset manager's
:py:meth:`~chimerax.preset_mgr.manager.PresetsManager.execute`
method to actually execute the preset, so that the proper information
about the preset can be logged.
The single argument to 
:py:meth:`~chimerax.preset_mgr.manager.PresetsManager.execute`
is either a Python function taking no arguments,
or a string containing a ChimeraX command.
The string is typically not just a single command, but multiple
commands separated by semi-colon (';') characters,
which is the approach our :py:func:`run_presets` function uses to
execute both a 
`size command <../../user/commands/size.html>`_
(to make the sticks/balls thin/small) and a
`style command <../../user/commands/style.html>`_
to switch to stick or ball-and-stick style.

.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command`
- :doc:`tutorial_tool`
- :doc:`tutorial_read_format`
- :doc:`tutorial_save_format`
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector` (previous topic)
- :doc:`tutorial_presets` (current topic)
