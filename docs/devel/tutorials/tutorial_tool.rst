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

.. _Bundle Example\: Add a Tool:


===========================
Bundle Example: Add a Tool
===========================

This example describes how to create a ChimeraX bundle
that defines a graphical interface.  Graphical interfaces
can either be based on HTML or on the Qt windowing toolkit.


Qt-based
========

ChimeraX's graphical interface is written using the
`Qt` windowing toolkit (version 5),
accessed through a Python wrapping of the toolkit
(the toolkit itself is written in C++).
The :doc:`Qt-based tool example <tutorial_tool_qt>`
demonstrates how to develop a graphical interface
using the Qt toolkit's capabilities.

For the sake of simplicity the :doc:`Qt example <tutorial_tool_qt>`
does not create an interface for the commands developed
earlier in the tutorial, but instead displays a text-input
field which logs text typed in by the user, using the
appropriate `log command <../../user/commands/log.html>`_.


HTML-based
==========

For developers who are experts in both developing HTML-based
user input forms and in Javascript development, it may be
easier to deploy an HTML-based graphical user interface.
Other developers are strongly advised to create a Qt-based
interface instead, as described in the preceding section.
The Javascript communication between the HTML front end and
Python back end is particularly challenging for non-experts.
That said, the :doc:`HTML-based tool example <tutorial_tool_html>`
demonstrates developing a graphical interface whose
layout is specified using HTML.

The :doc:`HTML example <tutorial_tool_html>` takes the commands
developed earlier in the tutorial and creates a graphical
interface for them.


What's Next
-----------

- :doc:`tutorial_hello`
- :doc:`tutorial_command` (previous topic)
- :doc:`tutorial_tool` (current topic)
- :doc:`tutorial_read_format` (next topic)
- :doc:`tutorial_save_format`
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector`
- :doc:`tutorial_presets`
