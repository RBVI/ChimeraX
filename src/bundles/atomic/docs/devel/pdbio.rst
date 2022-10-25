..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

pdb: Read and write PDB files
=============================

Directly calling the routines below would typically be about your third choice for reading/writing PDB files.  The first choice would be to execute the equivalent ChimeraX command directly with code like::

	from chimerax.core.commands import run
	opened_models = run(session, "open /path/to/file.pdb")

or::

	from chimerax.core.commands import run
	run(session, "save /path/to/file.pdb #1")

This approach has the advantage of not needing to find equivalent API calls to every command you want to execute, or figuring out what arguments are needed for those calls.  This approach is discussed in the :ref:`Pro Tip<pro-tip>` section near the top of the Developer Tutorial.

The second approach would be to use the open- or save-command managers.  Those managers know which bundles provide support for opening/saving various file formats, and provide a generic interface for opening/saving files, *e.g.*::

	models, status_message = session.open_command.open_data("/path/to/file.pdb", [other Python format-specific keywords])

or::

	session.save_command.save_data("/path/to/file.pdb", models=[model1],  [other Python format-specific keywords])

This second approach has the disadvantage that the values for the keywords may not be obvious in some cases (*i.e.* you would have to look at the underlying API).  Also, the models returned by ``open_data()`` have not been added to the session.  Details like this are discussed in the :ref:`Python Functions implementing User Commands<user_commands>` documentation, under :ref:`open<open_command>` and :ref:`save<save_command>`.

And finally, the third approach would be to call the PDB-saving API directly.  Something like::

	from chimerax.pdb import open_pdb
	models, status_message = open_pdb(session, "/path/to/file.pdb")

or::

	from chimerax.pdb import save_pdb
	save_pdb(session, "/path/to/file.pdb", models=[model1])

The only advantage of this third approach is in the rare case where you need to use an esoteric Python-only keyword argument that isn't supported in the equivalent open/save command.  For instance, ``save_pdb()`` has a ``polymeric_res_names argument`` for when you need to output residues in ATOM records that would otherwise be output as HETATM records (this capability is used by the modeller bundle).

.. automodule:: chimerax.pdb.pdb
    :members:
    :show-inheritance:
