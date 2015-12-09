..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. automodule:: chimerax.core.io
    :members:
    :show-inheritance:

Example::

    io.register_format("mmCIF", "Molecular structure",
        (".cif",), ("mmcif", "cif"),
        mime=("chemical/x-cif", "chemical/x-mmcif"),
        reference="http://mmcif.wwpdb.org/",
        requires_seeking=True, open_func=open_mmCIF)
