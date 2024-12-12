..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

===========================================
gui: Main ChimeraX graphical user interface
===========================================

.. module:: chimerax.ui

.. automodule:: chimerax.ui.gui
    :show-inheritance:
    :members:
    :exclude-members: closeEvent, customEvent, dragEnterEvent, dropEvent, keyPressEvent, moveEvent, resizeEvent, event, quit

Widgets for choosing models, etc.
+++++++++++++++++++++++++++++++++
.. automodule:: chimerax.ui.widgets.item_chooser
    :show-inheritance:
    :members:

Mouse modes
+++++++++++
.. automodule:: chimerax.mouse_modes.mousemodes
    :members:
    :exclude-members: MouseBinding
    :special-members: __init__
    :show-inheritance:

.. _option_widgets:

Widgets that interoperate with Settings
+++++++++++++++++++++++++++++++++++++++
.. automodule:: chimerax.ui.options.containers
    :members:
    :exclude-members: sizeHint
    :show-inheritance:

.. automodule:: chimerax.ui.options.options
    :members:
    :exclude-members: Signal
    :show-inheritance:

.. _python_data_tables:

Table widget for Python objects
+++++++++++++++++++++++++++++++
.. automodule:: chimerax.ui.widgets.item_table
    :members:
    :exclude-members: Signal
    :show-inheritance:


HTML-based tools/widgets
++++++++++++++++++++++++
.. automodule:: chimerax.ui.htmltool
    :show-inheritance:
    :members:

.. automodule:: chimerax.ui.widgets.htmlview
    :show-inheritance:
    :members:

Font-size convenience function
++++++++++++++++++++++++++++++
.. automodule:: chimerax.ui.font
    :show-inheritance:
    :members:

The 'ui' command
++++++++++++++++
.. automodule:: chimerax.ui.cmd
    :show-inheritance:
    :members:
