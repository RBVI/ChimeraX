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
    :emphasize-lines: 8-10,17-24,37-44

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-24).

The ``Providers`` sections on lines 42 through 44 use the
:ref:`Manager <Manager>`/:ref:`Provider <Provider>` protocol to inform
the "open command" manager that this bundle supports fetching data from
a database named ``homologene`` (really `HomoloGene`_,
but the user will type "homologene").

The attributes usable with the "open command" manager (with ``type="fetch"``)
are described in detail in :ref:`fetch command`.


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

.. literalinclude:: ../../../src/examples/tutorials/tut_fetch/src/__init__.py
    :language: python
    :linenos:
    :emphasize-lines: 13-32

The :py:meth:`run_provider` method is called by a ChimeraX manager
when it needs additional information from a provider or it needs a
provider to execute a task.
The *session* argument is a :py:class:`~chimerax.core.session.Session` instance,
the *name* argument is the same as the ``name`` attribute in your Provider
tag, and the *mgr* argument is the manager instance.
These arguments can be used to decide what to do when your bundle offers
several Provider tags (to possibly several managers), but since this bundle
only declares one provider to one manager, we know it will be called by the
"open command" manager to fetch HomoloGene data and don't need to check
the :py:meth:`run_provider` arguments.

When called by the "open command" manager (that was given the ``type="fetch"`` Provider tag),
:py:meth:`run_provider` must return an instance of a subclass of
:py:class:`chimerax.open_command.FetcherInfo`.
The methods of the class are thoroughly documented if you click the preceding
link, but briefly:

1. The :py:meth:`fetch` method is called to actually fetch the data and
   should return a (models, status message) tuple.  Do *not* add the models
   to the session — that will done by the calling function.
2. The *ignore_cache* argument indicates whether your routine should use locally
   cached data (if any) or instead ignore the cache and fetch the data again.
   Some types of data fetches may not amenable to caching at all, but for those
   that are the caching is usually implemented automatically by having the fetching
   function use the :py:func:`chimerax.core.fetch.fetch_file` routine, which takes an
   *ignore_cache* keyword argument.
3. If there are fetch-specific keyword arguments that the ``open`` command should
   handle, then a :py:meth:`fetch_args` property should be implemented, which
   returns a dictionary mapping **Python** keyword names to :ref:`Annotation <Type Annotations>`
   subclasses.  Such keywords will be passed to your :py:meth:`fetch` method,
   along with format-specific keywords.
   Note that format-specific keywords are known from the :py:meth:`open_args` property
   of the bundle that opens the data's format, and should *not* be included in the
   dictionary returned by :py:meth:`fetch_args`, so therefore it is rarely necessary
   to actually implement the :py:meth:`fetch_args` property.

For this example, the *format_name* argument is omitted because
the bundle only supports FASTA format.
All other arguments are passed through to ``fetch.fetch_homologene``
to actually retrieve and process the data.


``src/fetch.py``
----------------

.. literalinclude:: ../../../src/examples/tutorials/tut_fetch/src/fetch.py
    :language: python
    :linenos:

The ``fetch_homologene`` function performs the following steps:

- create an URL for fetching content for the given identifier
  and an output file name where the content will be saved
  (lines 17-19),
- call :py:func:`chimerax.core.fetch.fetch_file` to retrieve
  the actual contents (lines 20-22),
- update status line (line 24),
- open the saved file using the "open command" manager's
  :py:func:`~chimerax.open_command.manager.OpenManager.open_data`
  method (line 25), which return a (models, status message) tuple.
- return the list of models created and status message (line 26)


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command`
- :doc:`tutorial_tool`
- :doc:`tutorial_read_format`
- :doc:`tutorial_save_format` (previous topic)
- :doc:`tutorial_fetch` (current topic)
- :doc:`tutorial_selector` (next topic)
