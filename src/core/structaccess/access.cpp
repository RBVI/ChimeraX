// vi: set expandtab ts=4 sw=4:
#include <blob/AtomBlob.h>
#include <blob/BondBlob.h>
#include <blob/PseudoBlob.h>
#include <blob/ResBlob.h>
#include <blob/StructBlob.h>

extern "C" {

static struct PyMethodDef structaccess_functions[] =
{
    { NULL, NULL, 0, NULL }
};

static const char* doc_string =
":mod:`structaccess` -- Python access to C++ molecular data\n"
"==========================================================\n"
"\n"
".. module:: structaccess\n"
"   :synopsis: Access to C++ molecular aggregates (blobs)\n"
"\n"
"The structaccess module provides access to collections of C++\n"
"molecular data objects.  These collections are referred to as \"blobs\".\n"
"Each blob contains only molecular data of a single type, *e.g.*\n"
"an AtomBlob contains only atoms.\n"
"\n"
"Although it is possible to construct empty blobs with this module (*e.g.*\n"
"ab = structaccess.AtomBlob()), one typically gets blobs from other sources,\n"
"such as by reading a PDB file.\n"
"Data items in a blob are ordered and are not necessarily unique.\n"
"For instances, the :meth:`residues` method of an :class:`AtomBlob` returns\n"
"a :class:`ResBlob` containing the corresponding residues for the atoms\n"
"and in the same order as the atoms -- and obviously probably not unique.\n"
"\n"
"Blobs\n"
"-----\n"
"\n"
"The available blob types are:\n"
"\n"
"* :class:`AtomBlob` -- encapsulates C++ Atoms\n"
"* :class:`BondBlob` -- encapsulates C++ Bonds\n"
"* :class:`ResBlob` -- encapsulates C++ Residues\n"
"* :class:`StructBlob` -- encapsulates C++ AtomicStructures\n"
"* :class:`PseudoBlob` -- encapsulates C++ PBonds (pseudobonds)\n"
"\n"
"Universal blob methods/operations\n"
"---------------------------------\n"
"\n"
"*blob*\\ .filter(\\ *bools*\\ )\n"
"   Returns a new blob of the same type containing the data items\n"
"   corresponding to :const:`True` values in *bools*\\ .  *bools* must be a\n"
"   :mod:`numpy` array (or a value convertible to a :mod:`numpy` array) of\n"
"   the same size as the number of data items in the blob.\n"
"\n"
"*blob*\\ .intersect(\\ *other_blob*\\ )\n"
"   Returns a new blob that consists of the items of the second blob\n"
"   that are also found in the first blob.\n"
"\n"
"len(\\ *blob*\\ )\n"
"   Returns the number of data items in *blob*\\ .\n"
"\n"
"*blob*\\ .merge(\\ *other_blob*\\ )\n"
"   Returns a new blob that contains the items of the first blob, with\n"
"   items from the second blob appended if they don't occur in the first\n"
"   blob.\n"
"\n"
"*blob*\\ .subtract(\\ *other_blob*\\ )\n"
"   Returns a new blob that contains the items of the first blob that\n"
"   are not found in the second blob.\n"
"\n"
".. warning::\n"
"\n"
"   The intersect(), merge(), and subtract() methods do *exactly* what they\n"
"   are documented to do and therefore are typically not useful when used\n"
"   with blobs whose contents aren't unique (\\ *i.e.* that\n"
"   contain multiple copies of one or more data items).\n"
"\n"
"Blob classes\n"
"------------\n"
"\n"
".. class:: AtomBlob\n"
"\n"
"      Holds a list of C++ Atoms and provides access to some of\n"
"      their attributes.\n"
"\n"
"      .. attribute:: colors\n"
"\n"
"         Returns a :mod:`numpy` Nx4 array of RGBA values.  Can be\n"
"         set with such an array (or equivalent sequence), or with a single\n"
"         RGBA value.\n"
"\n"
"      .. attribute:: coords\n"
"\n"
"         Read only.  Returns a :mod:`numpy` Nx3 array of XYZ values.\n"
"\n"
"      .. attribute:: displays\n"
"\n"
"         Controls whether the Atoms should be displayed.\n"
"\n"
"         Returns a :mod:`numpy` array of boolean values.  Can be\n"
"         set with such an array (or equivalent sequence), or with a\n"
"         single boolean value.\n"
"\n"
"      .. attribute:: draw_modes\n"
"\n"
"         Controls how the Atoms should be depicted, *e.g.* sphere,.\n"
"         ball, *etc.*  The values are integers, whose semantic meaning\n"
"         is documented in the\n"
"         :class:`~chimera.core.structure.StructureModel` class.\n"
"\n"         
"         Returns a :mod:`numpy` array of integers.  Can be\n"
"         set with such an array (or equivalent sequence), or with a\n"
"         single integer value.\n"
"\n"
"      .. attribute:: element_names\n"
"\n"
"         Read only.  Returns a list of chemical element symbols.\n"
"\n"
"      .. attribute:: element_numbers\n"
"\n"
"         Read only.  Returns a :mod:`numpy` array\n"
"         of atomic numbers (integers).\n"
"\n"
"      .. attribute:: names\n"
"\n"
"         Read only.  Returns a list of atom names\n"
"\n"
"      .. attribute:: radii\n"
"\n"
"         Returns a :mod:`numpy` array of atomic radii.  Can be\n"
"         set with such an array (or equivalent sequence), or with a single\n"
"         floating-point number.\n"
"\n"
"      .. attribute:: residues\n"
"\n"
"         Read only.  Returns a :class:`ResBlob` whose data items\n"
"         correspond in a 1-to-1 fashion with the items in the\n"
"         AtomBlob.\n"
"\n"
".. class:: BondBlob\n"
"\n"
"      Holds a list of C++ Bonds and provides access to some of\n"
"      their attributes.\n"
"\n"
"      .. attribute:: atoms\n"
"\n"
"         Read only.  Returns a two-tuple of :class:`AtomBlob`\\ s.\n"
"         For each bond in the BondBlob, its endpoint atoms\n"
"         are in the corresponding slots in the :class:`AtomBlob`\\ s.\n"
"\n"
"      .. attribute:: colors\n"
"\n"
"         Returns a :mod:`numpy` Nx4 array of RGBA values.  Can be\n"
"         set with such an array (or equivalent sequence), or with a single\n"
"         RGBA value.\n"
"\n"
"      .. attribute:: displays\n"
"\n"
"         Controls whether the Bonds should be displayed.\n"
"         The values are integers, whose semantic meaning\n"
"         is documented in the\n"
"         :class:`~chimera.core.structure.StructureModel` class.\n"
"\n"         
"         Returns a :mod:`numpy` array of integers.  Can be\n"
"         set with such an array (or equivalent sequence), or with a\n"
"         single integer value.\n"
"\n"
"      .. attribute:: halfbonds\n"
"\n"
"         Controls whether the Bonds should be colored in \"halfbond\"\n"
"         mode, *i.e.* each half colored the same as its endpoint Atom.\n"
"\n"
"         Returns a :mod:`numpy` array of boolean values.  Can be\n"
"         set with such an array (or equivalent sequence), or with a\n"
"         single boolean value.\n"
"\n"
"      .. attribute:: radii\n"
"\n"
"         Returns a :mod:`numpy` array of bond radii (half thicknesses).\n"
"         Can be set with such an array (or equivalent sequence), or with a\n"
"         single floating-point number.\n"
"\n"
".. class:: ResBlob\n"
"\n"
"      Holds a list of C++ Residues and provides access to some of\n"
"      their attributes.\n"
"\n"
"      .. attribute:: chain_ids\n"
"\n"
"         Read only.  Returns a list of chain IDs.\n"
"\n"
"      .. attribute:: names\n"
"\n"
"         Read only.  Returns a list of residue names\n"
"\n"
"      .. attribute:: numbers\n"
"\n"
"         Read only.  Returns a :mod:`numpy` array\n"
"         of residue sequence numbers, as provided by\n"
"         whatever data source the structure came from,\n"
"         so not necessarily consecutve, or starting from 1, *etc.*\n"
"\n"
"      .. attribute:: strs\n"
"\n"
"         Read only.  Returns a list of strings that encapsulates each\n"
"         residue's name, sequence position, and chain ID in a readable\n"
"         form.\n"
"\n"
"      .. attribute:: unique_ids\n"
"\n"
"         Read only.  A :mod:`numpy` array of integers.\n"
"         Multiple copies of the same residue in the blob\n"
"         will have the same integer value in the list.\n"
"\n"
".. class:: StructBlob\n"
"\n"
"      Holds a list of C++ AtomicStructures and provides access to some of\n"
"      their attributes.\n"
"\n"
"      .. attribute:: atoms\n"
"\n"
"         Read only.  Returns an :class:`AtomBlob`\\ .\n"
"\n"
"      .. attribute:: ball_scales\n"
"\n"
"         Controls the scaling of the ball depiction (in ball-and-stick\n"
"         mode) from the default.  The default is documented as the\n"
"         :attr:`ball_scale <chimera.core.structure.StructureModel.ball_scale>`\n"
"         attribute of the :class:`~chimera.core.structure.StructureModel`\n"
"         class.\n"
"\n"
"         Returns a :mod:`numpy` array of ball scales.\n"
"         Can be set with such an array (or equivalent sequence), or with a\n"
"         single floating-point number.\n"
"\n"
"      .. attribute:: bonds\n"
"\n"
"         Read only.  Returns a :class:`BondBlob`\\ .\n"
"\n"
"      .. attribute:: displays\n"
"\n"
"         Controls whether the AtomicStructures should be displayed.\n"
"\n"
"         Returns a :mod:`numpy` array of boolean values.  Can be\n"
"         set with such an array (or equivalent sequence), or with a\n"
"         single boolean value.\n"
"\n"
"      .. attribute:: num_atoms\n"
"\n"
"         Read only.  Returns the total number of Atoms in the \n"
"         AtomicStructures.\n"
"\n"
"      .. attribute:: num_bonds\n"
"\n"
"         Read only.  Returns the total number of Bonds in the \n"
"         AtomicStructures.\n"
"\n"
"      .. attribute:: num_hyds\n"
"\n"
"         Read only.  Returns the total number of hydrogen Atoms in the \n"
"         AtomicStructures.\n"
"\n"
"      .. attribute:: num_residues\n"
"\n"
"         Read only.  Returns the total number of Residues in the \n"
"         AtomicStructures.\n"
"\n"
"      .. attribute:: num_chains\n"
"\n"
"         Read only.  Returns the total number of Chains in the \n"
"         AtomicStructures.\n"
"\n"
"      .. attribute:: num_coord_sets\n"
"\n"
"         Read only.  Returns the total number of CoordSets in the \n"
"         AtomicStructures.\n"
"\n"
"      .. attribute:: pbg_map\n"
"\n"
"         Read only.  Returns a dictionary whose keys are pseudobond\n"
"         group categories (strings) and whose values are\n"
"         :class:`PseudoBlob`\\ s.  This attribute only works with\n"
"         single-structure StructBlobs (see the\n"
"         :attr:`~StructBlob.structures` attribute).\n"
"\n"
"      .. attribute:: residues\n"
"\n"
"         Read only.  Returns a :class:`ResBlob`\\ .\n"
"\n"
"      .. attribute:: structures\n"
"\n"
"         Read only.  Returns a list of StructBlobs, each blob containing\n"
"         exactly one of the structures in this blob.\n"
"\n"
".. class:: PseudoBlob\n"
"\n"
"      Holds a list of C++ PBonds (pseudobonds) and provides access to some of\n"
"      their attributes. It has exactly the same attributes as the\n"
"      :class:`BondBlob` class and works in an analogous fashion.\n"
"";


static struct PyModuleDef structaccess_module =
{
    PyModuleDef_HEAD_INIT,
    "structaccess",
    doc_string,
    -1,
    structaccess_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_structaccess()
{
    using blob::StructBlob_type;
    using blob::ResBlob_type;
    using blob::AtomBlob_type;
    using blob::BondBlob_type;
    using blob::PseudoBlob_type;
    StructBlob_type.tp_new = blob::PyType_NewBlob<blob::StructBlob>;
    ResBlob_type.tp_new = blob::PyType_NewBlob<blob::ResBlob>;
    AtomBlob_type.tp_new = blob::PyType_NewBlob<blob::AtomBlob>;
    BondBlob_type.tp_new = blob::PyType_NewBlob<blob::BondBlob>;
    PseudoBlob_type.tp_new = blob::PyType_NewBlob<blob::PseudoBlob>;
    if (PyType_Ready(&StructBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&ResBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&AtomBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&BondBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&PseudoBlob_type) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&structaccess_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&StructBlob_type);
    Py_INCREF(&ResBlob_type);
    Py_INCREF(&AtomBlob_type);
    Py_INCREF(&BondBlob_type);
    Py_INCREF(&PseudoBlob_type);
    // make blob types visible so their doc strings can be accessed
    // and so that empty blobs can be constructed
    PyModule_AddObject(m, "StructBlob", (PyObject *)&StructBlob_type);
    PyModule_AddObject(m, "ResBlob", (PyObject *)&ResBlob_type);
    PyModule_AddObject(m, "AtomBlob", (PyObject *)&AtomBlob_type);
    PyModule_AddObject(m, "BondBlob", (PyObject *)&BondBlob_type);
    PyModule_AddObject(m, "PseudoBlob", (PyObject *)&PseudoBlob_type);
    return m;
}

}  // extern "C"
