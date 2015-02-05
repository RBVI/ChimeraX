// vi: set expandtab ts=4 sw=4:
#include <blob/StructBlob.h>
#include <blob/ResBlob.h>
#include <blob/AtomBlob.h>
#include <blob/BondBlob.h>
#include <atomstruct/AtomicStructure.h>
#include <atomstruct/Residue.h>
#include <atomstruct/Atom.h>
#include <atomstruct/Bond.h>
#include <basegeom/Coord.h>
#include <atomstruct/Element.h>
#include <vector>
#include <stdexcept>
#include <sstream>  // std::ostringstream

extern "C" {

static struct PyMethodDef structaccess_functions[] =
{
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef structaccess_module =
{
    PyModuleDef_HEAD_INIT,
    "structaccess",
    "Access functions for molecular aggregates",
    -1,
    structaccess_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_structaccess()
{
std::cerr << "sizeof Atom: " << sizeof(atomstruct::Atom) << "\n";
std::cerr << "sizeof BaseSphere: " << sizeof(basegeom::BaseSphere<atomstruct::Bond, atomstruct::Atom>) << "\n";
std::cerr << "sizeof Connectible: " << sizeof(basegeom::Connectible<atomstruct::Bond, atomstruct::Atom>) << "\n";
typedef basegeom::Connectible<atomstruct::Bond, atomstruct::Atom> Conn;
std::cerr << "sizeof Rgba: " << sizeof(basegeom::Rgba) << "\n";
std::cerr << "sizeof bool: " << sizeof(bool) << "\n";
std::cerr << "sizeof vector: " << sizeof(std::vector<void*>) << "\n";
std::cerr << "sizeof string: " << sizeof(std::string) << "\n";
std::cerr << "sizeof Element: " << sizeof(atomstruct::Element) << "\n";
std::cerr << "sizeof map: " << sizeof(std::map<int, int>) << "\n";
std::cerr << "sizeof unordered_map: " << sizeof(std::unordered_map<int, int>) << "\n";
std::cerr << "sizeof Bond: " << sizeof(atomstruct::Bond) << "\n";
std::cerr << "sizeof CString<5>: " << sizeof(chutil::CString<5>) << "\n";
    using blob::StructBlob_type;
    using blob::ResBlob_type;
    using blob::AtomBlob_type;
    using blob::BondBlob_type;
    StructBlob_type.tp_new = PyType_GenericNew;
    ResBlob_type.tp_new = PyType_GenericNew;
    AtomBlob_type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&StructBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&ResBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&AtomBlob_type) < 0)
        return NULL;
    if (PyType_Ready(&BondBlob_type) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&structaccess_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&StructBlob_type);
    Py_INCREF(&ResBlob_type);
    Py_INCREF(&AtomBlob_type);
    Py_INCREF(&BondBlob_type);
    // make blob types visible so their doc strings can be accessed
    PyModule_AddObject(m, "StructBlob", (PyObject *)&StructBlob_type);
    PyModule_AddObject(m, "ResBlob", (PyObject *)&ResBlob_type);
    PyModule_AddObject(m, "AtomBlob", (PyObject *)&AtomBlob_type);
    PyModule_AddObject(m, "BondBlob", (PyObject *)&BondBlob_type);
    return m;
}

}  // extern "C"
