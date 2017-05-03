#include <Python.h>
#include <atomstruct/AtomicStructure.h>

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

static char _sample_repr_doc[] =
"repr(o)\n"
"\n"
"Return string representation of o.";

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

	// Now we cast void* to our ChimeraX type
	atomstruct::Structure *s = static_cast<atomstruct::Structure *>(p);
	return Py_BuildValue("ii", s->num_atoms(), s->num_bonds());
}

static char _sample_counts_doc[] =
"repr(o)\n"
"\n"
"Return 2-tuple of number of atoms and bonds in a structure.";


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
	"_sample",				// m_name
	"Sample support module.",		// m_doc
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
PyInit__sample(void)
{
	PyObject* m = PyModule_Create(&_sample_module);
	// Sometimes you want to stick constants into the module
	return m;
}
