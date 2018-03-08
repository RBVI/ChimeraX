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

.. _Bundle Example\: Fetch from Network Database:


============================================
Bundle Example: Fetch from Network Database
============================================

This example describes how to create a ChimeraX bundle
that retrieves data from a network source.
In this example, the network source is `HomoloGene`_
from `NCBI`_.

The steps in implementing the bundle are:

#. Create a ``bundle_info.xml`` containing information
   about the bundle,
#. Create a Python package that interfaces with ChimeraX
   and implements the file-reading functionality, and
#. Install and test the bundle in ChimeraX.

The final step builds a Python wheel that ChimeraX uses
to install the bundle.  So if the bundle passes testing,
it is immediately available for sharing with other users.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_fetch>`_
containing a folder named `tut_fetch`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_fetch`` folder are:

- ``tut_fetch`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``fetch.py`` - source code to retrieve HomoloGene entries

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

.. literalinclude:: ../../../src/examples/tutorials/tut_fetch/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-24,41-47

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-24).

The ``ChimeraXClassifier`` tags on lines 41-47 informs ChimeraX that
this bundle supports fetching data from a source named ``HomoloGene``.
Note that there is no **DataFormat** or **Open** classifiers
because HomoloGene data can be read using the built-in **FASTA**
format parser.  The fields after **Fetch** are:

- the name of the source (**HomoloGene**),
- the format of the fetched data (**FASTA**),
- a prefix for the format if reading from a file (**homologene**),
- an example identifier for fetching data from the source (**87131**).

``src``
-------

.. include:: src.rst


``__init__.py``
---------------

As described in :doc:`tutorial_hello`, ``__init__.py`` contains
the initialization code that defines the ``bundle_api`` object
that ChimeraX needs in order to invoke bundle functionality.
ChimeraX expects ``bundle_api`` class to be derived from
:py:class:`chimerax.core.toolshed.BundleAPI` with methods
overridden for registering commands, tools, etc.

.. literalinclude:: ../../../src/examples/tutorials/tut_fetch/src/__init__.py
    :language: python
    :linenos:
    :emphasize-lines: 13-26

The ``fetch_from_database`` method is called by ChimeraX to
retrieve the content associated with an identifier from
a network source.
The first two arguments are **session**, a
:py:class:`chimerax.core.session.Session` instance; and
**identifier**, a string.
Optionally provided arguments include:

- **format**, the data format name, and
- **ignore_cache**, whether to use any cached information.

For this example, the optional arguments are omitted because
the bundle only supports FASTA format and does no caching.
All arguments are passed through to ``fetch.fetch_homologene``
to actually retrieve and process the data.


``fetch.py``
------------

.. literalinclude:: ../../../src/examples/tutorials/tut_fetch/src/fetch.py
    :language: python
    :linenos:

The ``fetch_homologene`` function performs the following steps:

- create a URL for fetching content for the given identifier
  and an output file name where the content will be saved
  (lines 18-20),
- call :py:func:`chimerax.core.fetch.fetch_file` to retrieve
  the actual contents (lines 21-24),
- update status line (line 26),
- open the saved file using :py:func:`chimerax.core.io.open_data`
  (lines 27-29), where the default format is ``FASTA`` but may
  be overridden by caller,
- finally, return the list of models created and status message
  from :py:func:`~chimerax.core.io.open_data` (lines 147-149).


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command`
- :doc:`tutorial_tool` (previous topic)
- :doc:`tutorial_read_format` (current topic)
- :doc:`tutorial_save_format` (next topic)
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector`
