#include "_mmcif.h"
#include "mmcif.h"
#include "corecif.h"
#include <stdlib.h>	/* for getenv() and atoi() */
#include <string.h>
#include <typeinfo>
#include <atomstruct/tmpl/restmpl.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

namespace mmcif {

PyObject* _mmcifErrorObj;
int _mmcifDebug;

void
_mmcifError()
{
	// generic exception handler
	try {
		// rethrow exception to look at it
		throw;
	} catch (std::bad_alloc&) {
		PyErr_SetString(PyExc_MemoryError, "not enough memory");
	} catch (std::invalid_argument& e) {
		PyErr_SetString(PyExc_TypeError, e.what());
	} catch (std::length_error& e) {
		PyErr_SetString(PyExc_MemoryError, e.what());
	} catch (std::out_of_range& e) {
		PyErr_SetString(PyExc_IndexError, e.what());
	} catch (std::overflow_error& e) {
		PyErr_SetString(PyExc_OverflowError, e.what());
	} catch (std::range_error& e) {
		PyErr_Format(_mmcifErrorObj, "range_error: %s", e.what());
	} catch (std::underflow_error& e) {
		PyErr_Format(_mmcifErrorObj, "underflow_error: %s", e.what());
	} catch (std::logic_error& e) {
		PyErr_SetString(PyExc_ValueError, e.what());
	} catch (std::ios_base::failure& e) {
		PyErr_SetString(PyExc_IOError, e.what());
	} catch (std::runtime_error& e) {
		const char *what = e.what();
		if (strcmp(what, "Python Error") == 0)
			; // nothing to do, already set
		else
			PyErr_SetString(_mmcifErrorObj, what);
	} catch (std::exception& e) {
		PyErr_Format(_mmcifErrorObj, "unknown error (%s): %s", typeid(e).name(), e.what());
	} catch (...) {
		if (_mmcifDebug)
			throw; // fatal exception
		PyErr_SetString(_mmcifErrorObj, "unknown C++ exception");
	}
}

static bool
sequence_to_vector_string(PyObject *seq, std::vector<std::string> *vec)
{
	if (!PySequence_Check(seq))
		return false;
	Py_ssize_t count = PySequence_Length(seq);
	vec->reserve(count);
	for (auto i = 0; i < count; ++i) {
		PyObject *o = PySequence_GetItem(seq, i);
		if (!PyUnicode_Check(o)) {
			Py_XDECREF(o);
			return false;
		}
		Py_ssize_t size;
		const char *data = PyUnicode_AsUTF8AndSize(o, &size);
		std::string inst(data, size);
		vec->push_back(inst);
		Py_DECREF(o);
	}
	return true;
}

static PyObject*
_mmcif_extract_CIF_tables(PyObject*, PyObject* _args)
{
	PyObject* _ptArg1;
	PyObject* _ptArg2;
	int _ptArg3 = false;
	if (!PyArg_ParseTuple(_args, "OO|p:extract_CIF_tables", &_ptArg1, &_ptArg2, &_ptArg3))
		return NULL;
	try {
		if (!PyUnicode_Check(_ptArg1))
			throw std::invalid_argument("argument 1 should be a str");
		Py_ssize_t size;
		const char *data = PyUnicode_AsUTF8AndSize(_ptArg1, &size);
		std::string cppArg1(data, size);
		std::vector<std::string> cppArg2;
		if (!sequence_to_vector_string(_ptArg2, &cppArg2))
			throw std::invalid_argument("argument 2 should be a sequence of str");
		bool cppArg3 = bool(_ptArg3);
		PyObject* _result = extract_CIF_tables(cppArg1.c_str(), cppArg2, cppArg3);
		return _result;
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmcifextract_CIF_tables_doc[] = "extract_CIF_tables(filename: str, categories: list of str) -> object";

static PyObject *
_mmcif_load_mmCIF_templates(PyObject*, PyObject* _ptArg)
{
	try {
		if (!PyUnicode_Check(_ptArg))
			throw std::invalid_argument("argument 1 should be a str");
		Py_ssize_t size;
		const char *data = PyUnicode_AsUTF8AndSize(_ptArg, &size);
		std::string cppArg1(data, size);
		load_mmCIF_templates(cppArg1.c_str());
		return (Py_INCREF(Py_None), Py_None);
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmcifload_mmCIF_templates_doc[] = "load_mmCIF_templates(filename: str)";

static PyObject*
_mmcif_parse_mmCIF_buffer(PyObject*, PyObject* _args, PyObject* _keywds)
{
	switch (PyTuple_Size(_args)) {
	  default:
		PyErr_SetString(PyExc_TypeError, "parse_mmCIF_buffer() expected 4 or 5 arguments");
		return NULL;
	  case 4: {
		Py_buffer _ptArg1;
		PyObject* _ptArg2;
		int _ptArg3;
		int _ptArg4;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_mmCIF_buffer() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "s*Oii:parse_mmCIF_buffer", &_ptArg1, &_ptArg2, &_ptArg3, &_ptArg4))
			return NULL;
		try {
			const unsigned char* cppArg1 = reinterpret_cast<unsigned char*>(_ptArg1.buf);
			PyObject* cppArg2 = _ptArg2;
			bool cppArg3(_ptArg3);
			bool cppArg4(_ptArg4);
			PyObject* _result = parse_mmCIF_buffer(cppArg1, cppArg2, cppArg3, cppArg4);
			PyBuffer_Release(&_ptArg1);
			return _result;
		} catch (...) {
			PyBuffer_Release(&_ptArg1);
			_mmcifError();
		}
		break;
	  }
	  case 5: {
		Py_buffer _ptArg1;
		PyObject* _ptArg2;
		PyObject* _ptArg3;
		int _ptArg4;
		int _ptArg5;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_mmCIF_buffer() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "s*OOii:parse_mmCIF_buffer", &_ptArg1, &_ptArg2, &_ptArg3, &_ptArg4, &_ptArg5))
			return NULL;
		try {
			const unsigned char* cppArg1 = reinterpret_cast<unsigned char*>(_ptArg1.buf);
			std::vector<std::string> cppArg2;
			if (!sequence_to_vector_string(_ptArg2, &cppArg2))
				throw std::invalid_argument("argument 2 should be a sequence of str");
			PyObject* cppArg3 = _ptArg3;
			bool cppArg4(_ptArg4);
			bool cppArg5(_ptArg5);
			PyObject* _result = parse_mmCIF_buffer(cppArg1, cppArg2, cppArg3, cppArg4, cppArg5);
			PyBuffer_Release(&_ptArg1);
			return _result;
		} catch (...) {
			PyBuffer_Release(&_ptArg1);
			_mmcifError();
		}
		break;
	  }
	}
	return NULL;
}

static const char _mmcifparse_mmCIF_buffer_doc[] = "parse_mmCIF_buffer(buffer: bytes, logger: object, coordsets: bool, atomic: bool) -> object\n\
parse_mmCIF_buffer(buffer: bytes, extra_categories: list of str, logger: object, coordsets: bool, atomic: bool) -> object";

static PyObject*
_mmcif_parse_mmCIF_file(PyObject*, PyObject* _args, PyObject* _keywds)
{
	switch (PyTuple_Size(_args)) {
	  default:
		PyErr_SetString(PyExc_TypeError, "parse_mmCIF_file() expected 4 or 5 arguments");
		return NULL;
	  case 4: {
		PyObject* _ptArg1;
		PyObject* _ptArg2;
		int _ptArg3;
		int _ptArg4;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_mmCIF_file() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "OOii:parse_mmCIF_file", &_ptArg1, &_ptArg2, &_ptArg3, &_ptArg4))
			return NULL;
		try {
			if (!PyUnicode_Check(_ptArg1))
				throw std::invalid_argument("argument 1 should be a str");
			Py_ssize_t size;
			const char *data = PyUnicode_AsUTF8AndSize(_ptArg1, &size);
			std::string cppArg1(data, size);
			PyObject* cppArg2 = _ptArg2;
			bool cppArg3(_ptArg3);
			bool cppArg4(_ptArg4);
			PyObject* _result = parse_mmCIF_file(cppArg1.c_str(), cppArg2, cppArg3, cppArg4);
			return _result;
		} catch (...) {
			_mmcifError();
		}
		break;
	  }
	  case 5: {
		PyObject* _ptArg1;
		PyObject* _ptArg2;
		PyObject* _ptArg3;
		int _ptArg4;
		int _ptArg5;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_mmCIF_file() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "OOOii:parse_mmCIF_file", &_ptArg1, &_ptArg2, &_ptArg3, &_ptArg4, &_ptArg5))
			return NULL;
		try {
			if (!PyUnicode_Check(_ptArg1))
				throw std::invalid_argument("argument 1 should be a str");
			Py_ssize_t size;
			const char *data = PyUnicode_AsUTF8AndSize(_ptArg1, &size);
			std::string cppArg1(data, size);
			std::vector<std::string> cppArg2;
			if (!sequence_to_vector_string(_ptArg2, &cppArg2))
				throw std::invalid_argument("argument 2 should be a sequence of str");
			PyObject* cppArg3 = _ptArg3;
			bool cppArg4(_ptArg4);
			bool cppArg5(_ptArg5);
			PyObject* _result = parse_mmCIF_file(cppArg1.c_str(), cppArg2, cppArg3, cppArg4, cppArg5);
			return _result;
		} catch (...) {
			_mmcifError();
		}
		break;
	  }
	}
	return NULL;
}

static const char _mmcifparse_mmCIF_file_doc[] = "parse_mmCIF_file(filename: str, logger: object, coordsets: bool, atomic: bool) -> object\n\
parse_mmCIF_file(filename: str, extra_categories: list of str, logger: object, coordsets: bool, atomic: bool) -> object";

static PyObject*
_mmcif_set_Python_locate_function(PyObject*, PyObject* _ptArg)
{
	try {
		if (!true)
			throw std::invalid_argument("argument 1 should be a object");
		PyObject* cppArg1 = _ptArg;
		set_Python_locate_function(cppArg1);
		return (Py_INCREF(Py_None), Py_None);
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmcifset_Python_locate_function_doc[] = "set_Python_locate_function(function: object)";

static PyObject*
_mmcif_find_template_residue(PyObject*, PyObject* _ptArg)
{
	try {
		if (!PyUnicode_Check(_ptArg))
			throw std::invalid_argument("argument 1 should be a str");
		Py_ssize_t size;
		const char *name = PyUnicode_AsUTF8AndSize(_ptArg, &size);
		std::string cppArg(name, size);
		const tmpl::Residue* r = find_template_residue(cppArg);
		if (!r) {
			PyErr_Format(PyExc_ValueError, "No template for residue type %s", name);
			return NULL;
		}
		PyObject* _result = r->py_instance(true);
		return _result;
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmciffind_template_residue_doc[] = "find_template_residue(name: str)";

static PyObject*
_mmcif_quote_value(PyObject*, PyObject* _args)
{
	PyObject* _ptArg1;
	int _ptArg2 = 60;
	if (!PyArg_ParseTuple(_args, "O|i:quote", &_ptArg1, &_ptArg2))
		return NULL;
	try {
		return quote_value(_ptArg1, _ptArg2);
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmcifquote_value_doc[] = "quote_value(obj: objects, max_len: int=60)";

static PyObject*
_mmcif_non_standard_bonds(PyObject*, PyObject* _args)
{
	static bool need_init = true;

	if (need_init) {
		import_array();
		need_init = false;
	}
	PyObject* _ptArg1;
	int _ptArg2 = false;
	int _ptArg3 = false;
	if (!PyArg_ParseTuple(_args, "O|ii:non_standard_bonds", &_ptArg1, &_ptArg2, &_ptArg3))
		return NULL;
	try {
		const char* tp_name = Py_TYPE(_ptArg1)->tp_name;
		// chimerax.atomic.molarray.Bonds
		if (strcmp(tp_name, "Bonds") != 0)
			throw std::invalid_argument("argument 1 should be a Bonds collection");
		PyObject* pointers = PyObject_GetAttrString(_ptArg1, "_pointers");
		if (pointers == NULL)
			return NULL;
		if (!PyArray_Check(pointers) || PyArray_NDIM((PyArrayObject*) pointers) != 1) {
			PyErr_Format(PyExc_ValueError, "unable to extract Bonds _pointers");
			return NULL;
		}
		const Bond** bonds = static_cast<const Bond**>(PyArray_DATA((PyArrayObject*) pointers));
		size_t bond_count = PyArray_DIMS((PyArrayObject*) pointers)[0];
		std::vector<const Bond*> disulfide, covalent;
		non_standard_bonds(bonds, bond_count, _ptArg2, _ptArg3, disulfide, covalent);
		PyObject* _result = PyTuple_New(2);
		PyObject* array;
		npy_intp size[1];
		size[0] = disulfide.size();
		if (size[0] == 0) {
			Py_INCREF(Py_None);
			PyTuple_SET_ITEM(_result, 0, Py_None);
		} else {
			array = PyArray_SimpleNew(1, size, NPY_UINTP);
			memcpy(PyArray_DATA((PyArrayObject*) array), &disulfide[0], size[0] * sizeof (void*));
			PyTuple_SET_ITEM(_result, 0, array);
		}
		size[0] = covalent.size();
		if (size[0] == 0) {
			Py_INCREF(Py_None);
			PyTuple_SET_ITEM(_result, 1, Py_None);
		} else {
			array = PyArray_SimpleNew(1, size, NPY_UINTP);
			memcpy(PyArray_DATA((PyArrayObject*) array), &covalent[0], size[0] * sizeof (void*));
			PyTuple_SET_ITEM(_result, 1, array);
		}
		return _result;
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmcifnon_standard_bonds_doc[] = "non_standard_bonds(bonds: Bonds, selected_only: bool, displayed_only: bool) -> tuple[disulfide, covalent]";

static PyObject*
_mmcif_parse_coreCIF_buffer(PyObject*, PyObject* _args, PyObject* _keywds)
{
	switch (PyTuple_Size(_args)) {
	  default:
		PyErr_SetString(PyExc_TypeError, "parse_coreCIF_buffer() expected 2 or 3 arguments");
		return NULL;
	  case 2: {
		Py_buffer _ptArg1;
		PyObject* _ptArg2;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_coreCIF_buffer() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "s*O:parse_coreCIF_buffer", &_ptArg1, &_ptArg2))
			return NULL;
		try {
			const unsigned char* cppArg1 = reinterpret_cast<unsigned char*>(_ptArg1.buf);
			PyObject* cppArg2 = _ptArg2;
			PyObject* _result = parse_coreCIF_buffer(cppArg1, cppArg2);
			PyBuffer_Release(&_ptArg1);
			return _result;
		} catch (...) {
			PyBuffer_Release(&_ptArg1);
			_mmcifError();
		}
		break;
	  }
	  case 3: {
		Py_buffer _ptArg1;
		PyObject* _ptArg2;
		PyObject* _ptArg3;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_coreCIF_buffer() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "s*OO:parse_coreCIF_buffer", &_ptArg1, &_ptArg2, &_ptArg3))
			return NULL;
		try {
			const unsigned char* cppArg1 = reinterpret_cast<unsigned char*>(_ptArg1.buf);
			std::vector<std::string> cppArg2;
			if (!sequence_to_vector_string(_ptArg2, &cppArg2))
				throw std::invalid_argument("argument 2 should be a sequence of str");
			PyObject* cppArg3 = _ptArg3;
			PyObject* _result = parse_coreCIF_buffer(cppArg1, cppArg2, cppArg3);
			PyBuffer_Release(&_ptArg1);
			return _result;
		} catch (...) {
			PyBuffer_Release(&_ptArg1);
			_mmcifError();
		}
		break;
	  }
	}
	return NULL;
}

static const char _mmcifparse_coreCIF_buffer_doc[] = "parse_coreCIF_buffer(buffer: bytes, logger: object) -> object\n\
parse_coreCIF_buffer(buffer: bytes, extra_categories: list of str, logger: object) -> object";

static PyObject*
_mmcif_parse_coreCIF_file(PyObject*, PyObject* _args, PyObject* _keywds)
{
	switch (PyTuple_Size(_args)) {
	  default:
		PyErr_SetString(PyExc_TypeError, "parse_coreCIF_file() expected 2 or 3 arguments");
		return NULL;
	  case 2: {
		PyObject* _ptArg1;
		PyObject* _ptArg2;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_coreCIF_file() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "OO:parse_coreCIF_file", &_ptArg1, &_ptArg2))
			return NULL;
		try {
			if (!PyUnicode_Check(_ptArg1))
				throw std::invalid_argument("argument 1 should be a str");
			Py_ssize_t size;
			const char *data = PyUnicode_AsUTF8AndSize(_ptArg1, &size);
			std::string cppArg1(data, size);
			PyObject* cppArg2 = _ptArg2;
			PyObject* _result = parse_coreCIF_file(cppArg1.c_str(), cppArg2);
			return _result;
		} catch (...) {
			_mmcifError();
		}
		break;
	  }
	  case 3: {
		PyObject* _ptArg1;
		PyObject* _ptArg2;
		PyObject* _ptArg3;
		if (_keywds != NULL && PyDict_Size(_keywds) != 0) {
			PyErr_SetString(PyExc_TypeError, "parse_coreCIF_file() expected no keyword arguments");
			return NULL;
		}
		if (!PyArg_ParseTuple(_args, "OOO:parse_coreCIF_file", &_ptArg1, &_ptArg2, &_ptArg3))
			return NULL;
		try {
			if (!PyUnicode_Check(_ptArg1))
				throw std::invalid_argument("argument 1 should be a str");
			Py_ssize_t size;
			const char *data = PyUnicode_AsUTF8AndSize(_ptArg1, &size);
			std::string cppArg1(data, size);
			std::vector<std::string> cppArg2;
			if (!sequence_to_vector_string(_ptArg2, &cppArg2))
				throw std::invalid_argument("argument 2 should be a sequence of str");
			PyObject* cppArg3 = _ptArg3;
			PyObject* _result = parse_coreCIF_file(cppArg1.c_str(), cppArg2, cppArg3);
			return _result;
		} catch (...) {
			_mmcifError();
		}
		break;
	  }
	}
	return NULL;
}

static const char _mmcifparse_coreCIF_file_doc[] = "parse_coreCIF_file(filename: str, logger: object) -> object\n\
parse_coreCIF_file(filename: str, extra_categories: list of str, logger: object) -> object";

PyMethodDef _mmcifMethods[] = {
	{
		"extract_CIF_tables", (PyCFunction) _mmcif_extract_CIF_tables,
		METH_VARARGS, _mmcifextract_CIF_tables_doc
	},
	{
		"load_mmCIF_templates", (PyCFunction) _mmcif_load_mmCIF_templates,
		METH_O, _mmcifload_mmCIF_templates_doc
	},
	{
		"parse_mmCIF_buffer", (PyCFunction) _mmcif_parse_mmCIF_buffer,
		METH_VARARGS | METH_KEYWORDS, _mmcifparse_mmCIF_buffer_doc
	},
	{
		"parse_mmCIF_file", (PyCFunction) _mmcif_parse_mmCIF_file,
		METH_VARARGS | METH_KEYWORDS, _mmcifparse_mmCIF_file_doc
	},
	{
		"set_Python_locate_function", (PyCFunction) _mmcif_set_Python_locate_function,
		METH_O, _mmcifset_Python_locate_function_doc
	},
	{
		"find_template_residue", (PyCFunction) _mmcif_find_template_residue,
		METH_O, _mmciffind_template_residue_doc
	},
	{
		"quote_value", (PyCFunction) _mmcif_quote_value,
		METH_VARARGS, _mmcifquote_value_doc
	},
	{
		"non_standard_bonds", (PyCFunction) _mmcif_non_standard_bonds,
		METH_VARARGS, _mmcifnon_standard_bonds_doc
	},
	{
		"parse_coreCIF_buffer", (PyCFunction) _mmcif_parse_coreCIF_buffer,
		METH_VARARGS | METH_KEYWORDS, _mmcifparse_coreCIF_buffer_doc
	},
	{
		"parse_coreCIF_file", (PyCFunction) _mmcif_parse_coreCIF_file,
		METH_VARARGS | METH_KEYWORDS, _mmcifparse_coreCIF_file_doc
	},
	{ NULL, NULL, 0, NULL }
};

static const char _mmcif_doc[] = "Function signature documentation only.\n\
See C++ headers for more documentation.";

} // namespace mmcif

static PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"_mmcif", // m_name 
	mmcif::_mmcif_doc, // m_doc
	-1, // m_size
	mmcif::_mmcifMethods, // m_methods
	NULL, // m_reload
	NULL, // m_traverse
	NULL, // m_clear
	NULL, // m_free
};

PyMODINIT_FUNC
PyInit__mmcif()
{
	PyObject* module = PyModule_Create(&moduledef);
	if (module == NULL)
		return NULL;

	const char* debug = getenv("_mmcifDebug");
	if (debug != NULL)
		mmcif::_mmcifDebug = atoi(debug);

	mmcif::_mmcifErrorObj = PyErr_NewException(PY_STUPID "_mmcif.error", NULL, NULL);
	if (mmcif::_mmcifErrorObj == NULL)
		return NULL;
	Py_INCREF(mmcif::_mmcifErrorObj);
	PyModule_AddObject(module, "error", mmcif::_mmcifErrorObj);

	return module;
}
