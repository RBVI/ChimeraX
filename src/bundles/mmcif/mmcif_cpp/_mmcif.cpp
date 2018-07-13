#include "_mmcif.h"
#include "mmcif.h"
#include <stdlib.h>	/* for getenv() and atoi() */
#include <string.h>
#include <typeinfo>

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
	int count = PySequence_Length(seq);
	vec->reserve(count);
	for (int i = 0; i < count; ++i) {
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
_mmcif_extract_mmCIF_tables(PyObject*, PyObject* _args)
{
	PyObject* _ptArg1;
	PyObject* _ptArg2;
	if (!PyArg_ParseTuple(_args, "OO:extract_mmCIF_tables", &_ptArg1, &_ptArg2))
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
		PyObject* _result = extract_mmCIF_tables(cppArg1.c_str(), cppArg2);
		return _result;
	} catch (...) {
		_mmcifError();
	}
	return NULL;
}

static const char _mmcifextract_mmCIF_tables_doc[] = "extract_mmCIF_tables(filename: str, categories: list of str) -> object";

static PyObject*
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

PyMethodDef _mmcifMethods[] = {
	{
		"extract_mmCIF_tables", (PyCFunction) _mmcif_extract_mmCIF_tables,
		METH_VARARGS, _mmcifextract_mmCIF_tables_doc
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
