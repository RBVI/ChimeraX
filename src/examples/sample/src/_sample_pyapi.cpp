#include <Python.h>
#include <atomstruct/AtomicStructure.h>

#include "./_sample.h"

//
// "repr" returns "repr(o)"
//
static PyObject*
_sample_repr(PyObject* self, PyObject* args)
{
	PyObject* value;
	if (!PyArg_ParseTuple(args, "O", &value))
		return NULL;
	return PyObject_Repr(value);
}


//
// "counts" returns number of atoms and bonds in a Structure
//
static PyObject*
_sample_counts(PyObject* self, PyObject* args)
{
	PyObject* as;		// as = Python AtomicStructure
	if (!PyArg_ParseTuple(args, "O", &as))
		return NULL;
	//
	// Essentially, code below does (in pidgin code):
	// void* p = (void *) as._c_pointer.value;
	//
	PyObject* pointer = PyObject_GetAttrString(as, "_c_pointer");
	if (pointer == NULL) {
		PyErr_SetString(PyExc_ValueError, "structure deleted");
		return NULL;
	}
	PyObject* value = PyObject_GetAttrString(pointer, "value");
	Py_DECREF(pointer);
	if (value == NULL) {
		PyErr_SetString(PyExc_ValueError, "structure pointer missing");
		return NULL;
	}
	if (!PyLong_Check(value)) {
		Py_DECREF(value);
		PyErr_SetString(PyExc_ValueError, "structure pointer mis-typed");
		return NULL;
	}
	void *p = PyLong_AsVoidPtr(value);
	Py_DECREF(value);

	// Now get the actual values. The function atom_and_bond_count() is defined
	// in _sample.h
	size_t atom_counts, bond_counts;
	std::tie(atom_counts, bond_counts) = atom_and_bond_count(p);
	return Py_BuildValue("ii", atom_counts, bond_counts);
}


//
// List of all methods in module
//
static PyMethodDef _sample_methods[] = {
	{"repr", _sample_repr, METH_VARARGS, _sample_repr_doc},
	{"counts", _sample_counts, METH_VARARGS, _sample_counts_doc},
	{NULL, NULL}
};


//
// Module instance
//
static struct PyModuleDef _sample_module = {
	PyModuleDef_HEAD_INIT,
	"_sample_pyapi",				// m_name
	_sample_module_doc,		// m_doc
	-1,					// m_size
	_sample_methods,			// m_methods
	NULL,					// m_reload
	NULL,					// m_traverse
	NULL,					// m_clear
	NULL,					// m_free
};


//
// Module initialization function (called when module is loaded)
// Two underscores in function name because the first is a required
// separator and the second is the first character in the module
// name "_sample"
//
PyMODINIT_FUNC
PyInit__sample_pyapi(void)
{
	PyObject* m = PyModule_Create(&_sample_module);
	// Sometimes you want to stick constants into the module
	return m;
}
