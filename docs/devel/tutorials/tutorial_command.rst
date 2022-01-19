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

.. include:: references.rst

.. _Bundle Example\: Add a Command:


==============================
Bundle Example: Add a Command
==============================

This tutorial builds on the material from :doc:`tutorial_hello`.

This example describes how to create a ChimeraX bundle
that defines two new commands, ``tutorial cofm`` and
``tutorial highlight``.  The steps in implementing the
bundle are:

#. Create a ``bundle_info.xml`` containing information
   about the bundle,
#. Create a Python package that interfaces with ChimeraX
   and implements the command functionality, and
#. Install and test the bundle in ChimeraX.

The final step builds a Python wheel that ChimeraX uses
to install the bundle.  So if the bundle passes testing,
it is immediately available for sharing with other users.

Before deciding on the name and syntax of your own command,
you should peruse the :doc:`command style guide <../command_style>`.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_cmd>`_
containing a folder named `tut_cmd`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_cmd`` folder are:

- ``tut_cmd`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``cmd.py`` - source code to implement two ``tutorial`` commands
        - ``docs/users/commands/tutorial.html`` - help file for the
          ``tutorial`` commands

The file contents are shown below.


``bundle_info.xml``
-------------------

``bundle_info.xml`` is an `eXtensible Markup Language`_
format file whose tags are listed in :doc:`bundle_info`.
While there are many tags defined, only a few are needed
for bundles written completely in Python.  The
``bundle_info.xml`` in this example is similar to the one
from the ``hello world`` example with changes highlighted.
For explanations of the unhighlighted lines, please see
:doc:`tutorial_hello`.

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,33-35,49-52

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-25).  The ``DataFiles`` tag is added
to include documentation files (lines 33-35).
The only other change is replacing
the ``ChimeraXClassifier`` tags to declare the two commands
in this bundle (lines 49-52).

Note that the two command, ``tutorial cofm`` (Center OF Mass)
and ``tutorial highlight``, are multi-word commands that share
the same initial word.  Most bundles that provide multiple
commands should add multi-word commands that share the same
"umbrella" name, e.g., ``tutorial`` in this example.
All names in the command may be shortened, so ``tut high``
is an accepted alternative to ``tutorial highlight``, which
minimizes the typing burden on the user.
Note also that the ``ChimeraXClassifier`` tag text may be split
over multiple lines for readability.  Whitespace characters
around ``::`` are ignored.


``src``
-------

.. include:: src.rst


``src/__init__.py``
-------------------

The command registration code is essentially the same as
:doc:`tutorial_hello`, except that the command
information, ``ci``, is used to get the full name (as listed
in ``bundle_info.xml``) of the command to be registered,
and the corresponding function and description are retrieved
from the ``cmd`` module.

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/__init__.py
    :language: python
    :linenos:


``src/cmd.py``
--------------

``cmd.py`` contains the functions that implement the bundle commands.
For example, the ``cofm`` function is called when the user issues a
``tutorial cofm`` command.  To report the center of mass of a set of
atoms, ``cofm`` requires several parameters supplied by the user:

#. the atoms of interest,
#. in which coordinate system to do the computation
   (using the atomic coordinates from the input file,
   or include geometric transformations relative to other models), and
#. whether the center calculation is weighted by the atomic masses.

It then takes the parameters, computes the center of mass, and
reports the result to the ChimeraX log.  The missing link is
how the user-typed command gets translated into a call to ``cofm``.
This is the purpose of the call to
:py:class:`chimerax.core.commands.cli.register`
in the ``register_command`` method in ``__init__.py``.
The ``register`` call tells ChimeraX to associate a function and
description with a command name.  In this case, ``cofm`` and
``cofm_desc`` are the function and description associated with
the command ``tutorial cofm``.  When the user types a command that
starts with ``tutorial cofm`` (or some abbreviation thereof), ChimeraX
parses the input text according to a standard syntax, maps the input
words to function arguments using the command description,
and then calls the function.

The standard syntax of ChimeraX commands is of the form:

  *command* *required_arguments* *optional_arguments*

*command* is the command name, possibly abbreviated and multi-word.
Required arguments appear immediately after the command.
If there are multiple required arguments, they must be specified in a
prespecified order, i.e., they must all be present and are positional.
Optional arguments appear after required arguments.  They are
typically keyword-value pairs and, because they are keyword-based,
may be in any order.

A command description instance describes how to map input text to
Python values.  It contains a list of 2-tuples for required arguments
and another for optional arguments.  The first element of the 2-tuple
is a string that matches one of the command function parameter names.
The second element is a "type class".  ChimeraX provides a variety
of built-in type classes such as
:py:class:`~chimerax.core.commands.cli.BoolArg` (Boolean),
:py:class:`~chimerax.core.commands.cli.IntArg` (integer),
:py:class:`~chimerax.core.commands.cli.AtomsArg` (container of atoms),
and :py:class:`~chimerax.core.commands.atomspec.AtomSpecArg` (atom specifier).
See :py:mod:`chimerax.core.commands` for the full
list.  The order of the required parameters list (in the command
description) must match the expected order for required arguments
(in the input text).

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/cmd.py
    :language: python
    :linenos:

:py:func:`cofm` is the function called from ``__init__.py`` when the
user enters the ``cofm`` command.  It retrieves the array of atoms,
their coordinates, and their center of mass by calling the internal
function :py:func:`_get_cofm` and reports the result via
:py:attr:`session.logger`, an instance of
:py:class:`chimerax.core.logger.Logger`.

:py:attr:`cofm_desc` contains the description of what arguments are
required or allowed for the ``cofm`` command.  The details of its
declaration are described in the comments in the example.

:py:func:`highlight` is the function called from ``__init__.py`` when the
user enters the ``highlight`` command.  Like :py:func:`cofm`, it retrieves
the array of atoms, their coordinates, and their center of mass by
calling :py:func:`_get_cofm`.  It then

#. computes the distances from each atom to the center of mass
   using Numpy (line 88),
#. sorts the atom indices by distances so that indices of atoms that
   are closer to the center of mass are towards the front of the
   sort result (:code:`argsort(distances)`), and select the first
   :code:`count` indices (line 94),
#. turn the array of indices into an array of atoms (line 97),
   and
#. finally, set the color of the selected atoms (line 101).
   The :py:attr:`colors` attribute of the atomic array is an
   Nx4 array of integers, where N is the number of atoms and
   the rows (of 4 elements) are the RGBA values for each atom.
   The :code:`color` argument to :py:func:`highlight` is an instance
   of :py:class:`chimerax.core.colors.Color`, whose :py:meth:`uint8x4`
   returns its RGBA value as an array of four (:code:`x4`)
   8-bit integers (:code:`uint8`).

:py:func:`_get_cofm`, used by both :py:func:`cofm` and
:py:func:`highlight`, is passed three arguments:

- :code:`atoms`, the atoms specified by the user, if any.
- :code:`transformed`, whether to retrieve transformed (scene) or
  untransformed (original) coordinates.  Untransformed coordinates
  can typically be used when only a single model is involved because
  the atoms are fixed relative to each other.  Transformed coordinates
  must be used when distances among multiple models are being computed
  (*i.e.*, the models must all be in same coordinate system).
- :code:`weighted`, whether to include atomic mass as part of the
  center of mass computation.  Frequently, an unweighted average of
  atomic coordinates, which is simpler and faster to compute, is
  sufficient for qualitative analysis.

If the user did not choose specific atoms (when :code:`atoms`
is :code:`None`), the usual ChimeraX interpretation is that all
atoms should be used (lines 123-125).
:py:func:`chimerax.core.commands.atomspec.all_objects` returns
an instance of `chimerax.core.objects.Object` that contains
all open models in the current ChimeraX session, and whose
:py:attr:`atoms` attribute is an array of atoms in the included
models.  Transformed and untransformed coordinates are accessed
using the :py:attr:`scene_coords` and :py:attr:`coords` attributes
of the atom array, respectively (lines 132-135).  If atomic mass
need not be included, the "center of mass" is simply the average
of the coordinates (line 141); if a weighted calculation is required,
(a) the atomic masses are retrieved by :code:`atoms.elements.masses`
(line 143),
(b) the coordinates are scaled by the corresponding atomic masses
(line 144), and
(c) the weighted average is computed (line 145).

For performance, ChimeraX makes use of `NumPy`_ arrays in many contexts.
The container for atoms is typically a
:py:class:`chimerax.atomic.molarray.Collection`
instance, as are those for bonds, residues, and atomic structures.
Fetching the same attribute, e.g., coordinates, from a collection
of molecular data, e.g., atoms, usually results in a NumPy array.
Although code involving NumPy arrays are sometimes opaque, they are
typically much more efficient than using Python loops.

.. _command help:

``src/docs/user/commands/tutorial.html``
----------------------------------------

The documentation for the ``tutorial`` command should be written
in `HTML 5`_ and saved in a file whose name matches the command
name and has suffix ``.html``, i.e., ``tutorial.html``.
If the bundle command is a subcommand of an existing command
(*e.g.* **color bundlecoloring**) then any spaces should be
replaced by underscores.
When help files are included in bundles, documentation for
the commands may be displayed using the **help** command,
the same as built-in ChimeraX commands.
The directory structure is chosen to allow for multiple types
of documentation for a bundle.
For example, developer documentation such as
the bundle API are saved in a ``devel`` directory instead of
``user``; documentation for graphical tools are saved in
``user/tools`` instead of ``user/commands``.

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/docs/user/commands/tutorial.html
    :language: html
    :linenos:

While the only requirement for documentation is that it be written
as HTML, it is recommended that developers write command help files
following the above template, with:

- a banner linking to the documentation index,
- a usage section with a summary of the command syntax,
- text describing each command in the bundle, and
- an address for contacting the bundle author.

Note that the target links used in the HTML file are all relative
to ``..``.
Even though the command documentation HTML file is stored with the
bundle, ChimeraX treats the links as if the file were located in
the ``commands`` directory in the developer documentation tree.
This creates a virtual HTML documentation tree where command HTML
files can reference each other without having to be collected
together.


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello` (previous topic)
- :doc:`tutorial_command` (current topic)
- :doc:`tutorial_tool` (next topic)
- :doc:`tutorial_read_format`
- :doc:`tutorial_save_format`
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector`
- :doc:`tutorial_presets`
