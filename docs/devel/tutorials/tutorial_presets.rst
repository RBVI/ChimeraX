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
        - ``preset.py`` - source code to define/execute presets

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
    :emphasize-lines: 8-10,19-23,26-28,31-35,37-39

The ``BundleInfo``, ``Synopsis``, ``Description`` and ``Category`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10, 19-23, and 26-28).

Since the preset uses functionality from the PresetMgr and Atomic bundles, the ``Dependencies``
section has been changed to reflect that (lines 31-35).

The ``Providers`` section on lines 37-39 informs ChimeraX that
this bundle defines one preset named "thin sticks" and that the
preset's category is "small molcule" for organizing it in the Presets
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

..
    TODO

.. literalinclude:: ../../../src/examples/tutorials/tut_preset/src/__init__.py
    :language: python
    :linenos:

The :py:meth:`register_selector` method is called by ChimeraX,
once for each selector listed in ``bundle_info.xml``,
before the first time a command target specification is parsed.
In this example, the method is called a single time
with selector name ``endres``.

The arguments to :py:meth:`register_selector`, in bundle API version 1,
are ``bi``, a :py:class:`chimerax.core.toolshed.BundleInfo` instance,
``si``, a :py:class:`chimerax.core.toolshed.SelectorInfo` instance, and
``logger``, a :py:class:`chimerax.core.logger.Logger` instance.
The method is expected to call
:py:meth:`chimerax.core.commands.atomspec.register_selector` to define
a selector whose name is given by ``si.name``.
Note that there is no ``session`` argument because, like commands,
selectors are session-independent; that is, once registered, a selector
may be used in any session.


``src/selector.py``
-------------------

``selector.py`` defines both the callback function, ``_select_endres``,
that is invoked when ``endres`` is encountered in a target specification,
as well as the function for registering ``select_endres`` with ChimeraX.

.. literalinclude:: ../../../src/examples/tutorials/tut_sel/src/selector.py
    :language: python
    :linenos:

The code in ``selector.py`` is designed to register multiple
selector callback functions using the same registration function.
When :py:func:`register` is called from
:py:meth:`__init__.bundle_api.register_selector`,
it looks up the callback function associated
with the given selector name using the ``_selector_func`` dictionary,
and registers it using
:py:class:`chimerax.core.commands.atomspec.register_selector`.

A selector callback function is invoked with three arguments:
``session``, a :py:class:`chimerax.core.session.Session` instance,
``models``, a list of :py:class:`chimerax.core.models.Model` instances, and
``results``, a :py:class:`chimerax.core.objects.Objects` instance.
The callback function is expected to process all the given ``models``
and add items of interest to ``results``.  Currently, the only items
that can be added are instances of :py:class:`chimerax.core.models.Model`,
:py:class:`chimerax.atomic.Atom` and
:py:class:`chimerax.atomic.Bond`.
Typically, :py:class:`~chimerax.core.models.Model` instances
are only added explicitly for non-atomic models.
More commonly, atoms (and bonds) are added
using the :py:meth:`~chimerax.core.objects.Objects.add_atoms` method.


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command`
- :doc:`tutorial_tool`
- :doc:`tutorial_read_format`
- :doc:`tutorial_save_format`
- :doc:`tutorial_fetch` (previous topic)
- :doc:`tutorial_selector` (current topic)
