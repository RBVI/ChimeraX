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

``tut_cmd`` - bundle folder
    ``bundle_info.xml`` - bundle information read by ChimeraX
    ``src`` - source code to Python package for bundle
        ``__init__.py`` - package initializer and interface to ChimeraX
        ``cmd.py`` - source code to implement two ``tutorial`` commands

The file contents are shown below.


``bundle_info.xml``
-------------------

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,41-45

``bundle_info.xml`` is an `_eXtensible Markup Language`_
format file whose tags are listed in :doc:`bundle_info`.
While there are many tags defined, only a few are needed
for bundles written completely in Python.  The
``bundle_info.xml`` in this example is similar to the one
from the ``hello world`` example with changes highlighted.
For explanations of the unhighlighted lines, please see
:doc:`tutorial_hello`.
The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-25).  The only other change is replacing
the ``ChimeraXClassifier`` tags to declare the two commands
in this bundle (lines 41-45).

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


``__init__.py``
---------------

The command registration code is essentially the same as
:doc:`tutorial_hello`, except that the command
information, ``ci``, is used to get the full name (as listed
in ``bundle_info.xml``) of the command to be registered,
and the corresponding function and description are retrieved
from the ``cmd`` module.

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/__init__.py
    :language: python
    :linenos:


``cmd.py``
----------

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
This is the purpose of the call to ``chimerax.core.commands.register``
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
of built-in type classes such as ``BoolArg`` (Boolean), ``IntArg``
(integer), ``AtomsArg`` (container of atoms), and ``AtomspecArg``
(atom specifier).  See ``chimerax.core.commands`` for the full
list.  The order of the required parameters list (in the command
description) must match the expected order for required arguments
(in the input text).

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/cmd.py
    :language: python
    :linenos:

For performance, ChimeraX makes use of `Numpy`_ arrays in many contexts.
The container for atoms is typically a ``chimerax.core.atomic.Collection``
instance, as are those for bonds, residues, and atomic structures.
Fetching the same attribute, e.g., coordinates, from a collection
of molecular data, e.g., atoms, usually results in a NumPy array.
Although code involving NumPy arrays are sometimes opaque, they are
typically much more efficient than using Python loops.


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
