// Copyright (c) 1998-2007 The Regents of the University of California.
// All rights reserved.
//
// Redistribution and use in source and binary forms are permitted
// provided that the above copyright notice and this paragraph are
// duplicated in all such forms and that any documentation,
// distribution and/or use acknowledge that the software was developed
// by the Computer Graphics Laboratory, University of California,
// San Francisco.  The name of the University may not be used to
// endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
// WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
// IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY OF CALIFORNIA BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS SOFTWARE.

// $Id: WrapPyModule.cpp 37506 2012-09-25 05:45:02Z gregc $

#define WRAPPY_EXPORT
#include "WrapPy3.h"
#include <structmember.h>
#include <unicodeobject.h>
#ifdef _WIN32
# include <stdio.h>
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
# include <eh.h>
#endif

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char*)
#endif

namespace wrappy {

PythonError::PythonError(): std::runtime_error("Python Error")
{
}
  
#ifdef _WIN32
SE_Exception::SE_Exception(SE_Exception& e) throw ()
{
	nSE = e.nSE;
}

SE_Exception::SE_Exception(unsigned int n) throw (): nSE(n)
{
}

SE_Exception::~SE_Exception() throw ()
{
}

const char *
SE_Exception::what() throw ()
{
	static char message[64];
	snprintf(message, sizeof message, "Microsoft SE exception %X", nSE);
	return message;
}
#endif

char
PythonUnicode_AsCChar(PyObject *obj)
{
	if (!PyUnicode_Check(obj))
		throw std::invalid_argument("expected str object");
	if (PyUnicode_GetLength(obj) > 1)
		throw std::invalid_argument("too many characters");
	Py_UCS4 result = PyUnicode_ReadChar(obj, 0);
	if (result > 0x7f)
		throw std::invalid_argument("does not fit in C character");
	return result;
}

std::string
PythonUnicode_AsCppString(PyObject *obj)
{
	if (!PyUnicode_Check(obj))
		throw std::invalid_argument("expected str object");
	Py_ssize_t size;
	const char *data = PyUnicode_AsUTF8AndSize(obj, &size);
	std::string result(data, size);
	return result;
}

int
PyType_AddObject(PyTypeObject* tp, const char* name, PyObject* o)
{
	PyObject *s;
	int err = 0;

	if (o == NULL) {
		if (!PyErr_Occurred())
			PyErr_SetString(PyExc_TypeError,
				"PyType_AddObject: need non-NULL value");
		return -1;
	}
	
	if (tp->tp_dict == NULL && PyType_Ready(tp) < 0) {
		err = -1;
		goto cleanup;
	}

	if (tp->tp_setattr != NULL) {
		err = tp->tp_setattr(reinterpret_cast<PyObject*>(tp),
				PY_STUPID name, o);
		goto cleanup;
	}

	s = PyUnicode_InternFromString(name);
	if (s == NULL) {
		err = -1;
		goto cleanup;
	}
	if (tp->tp_setattro != NULL) {
		err = (*tp->tp_setattro)(reinterpret_cast<PyObject*>(tp), s, o);
	} else {
		if (!PyErr_Occurred())
			PyErr_SetString(PyExc_TypeError,
				"PyType_AddObject: type has no setattr method");
		err = -1;
	}
	Py_DECREF(s);
cleanup:
	Py_DECREF(o);
	return err;
}

int
MutableType_Ready(PyTypeObject* type)
{
	/* Initialize ob_type if NULL.  This means extensions that want to be
	   compilable separately on Windows can call PyType_Ready() instead of
	   initializing the ob_type field of their type objects. */
	if (Py_TYPE(type) == NULL) {
		Py_TYPE(type) = &Mutable_Type;
		if (Py_TYPE(&Mutable_Type) == NULL) {
			if (PyType_Ready(&Mutable_Type) < 0)
				return -1;
		}
	}

	return PyType_Ready(type);
}

} // namespace wrappy

extern "C" {

#if 0
static PyObject *
Mutable_GetAttr(PyObject *obj, PyObject *name)
{
	return PyType_Type.tp_getattro(obj, name);
}
#endif

static int
Mutable_SetAttr(PyObject *obj, PyObject *name, PyObject *value)
{
	// Can't just call PyObject_GenericSetAttr because
	// we need to able to update the __str__ slot as well.
	PyTypeObject *type = reinterpret_cast<PyTypeObject*>(obj);
	type->tp_flags |= Py_TPFLAGS_HEAPTYPE;
	int result = PyType_Type.tp_setattro(obj, name, value);
	type->tp_flags &= ~Py_TPFLAGS_HEAPTYPE;
	return result;
}

static const char Mutable_doc[] = "\n\
Modified 'type' that allows methods to be added/changed";

PyTypeObject Mutable_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"libwrappy2.mutable", // tp_name
	sizeof (PyHeapTypeObject), // tp_basicsize
	sizeof (PyMemberDef), // tp_itemsize
	0, // tp_dealloc
	0, // tp_print
	0, // tp_getattr
	0, // tp_setattr
	0, // tp_compare
	0, // tp_repr
	0, // tp_as_number
	0, // tp_as_sequence
	0, // tp_as_mapping
	0, // tp_hash
	0, // tp_call
	0, // tp_str
	0 /*Mutable_GetAttr*/, // tp_getattro
	Mutable_SetAttr, // tp_setattro
	0, // tp_as_buffer
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC
		| Py_TPFLAGS_BASETYPE | Py_TPFLAGS_TYPE_SUBCLASS, // tp_flags
	Mutable_doc, // tp_doc
	0, // tp_traverse
	0, // tp_clear
	0, // tp_richcompare
	offsetof(PyTypeObject, tp_weaklist), // tp_weaklistoffset
	0, // tp_tp_iter
	0, // tp_iternext
	0, // tp_methods
	0, // tp_members
	0, // tp_getset
	0, // tp_base
	0, // tp_dict
	0, // tp_descr_get
	0, // tp_descr_set
	offsetof(PyTypeObject, tp_dict), // tp_dictoffset
	0, // tp_init
	0, // tp_alloc
	0, // tp_new
	0, // tp_free
	0, // tp_is_gc
	0, // tp_bases
	0, // tp_mro
	0, // tp_cache
	0, // tp_subclasses
	0, // tp_weaklist
	0, // tp_del
};

// This module exists primarily so the wrappy shared library can be
// loaded by Python before other wrappy generated modules thus preloading
// the runtime linker/loader.

static PyMethodDef methods[] = { { NULL, NULL, 0, NULL } };

static PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"libwrappy2",
	"wrappy2 Python/C++ glue",
	-1,
	methods,
	NULL,
	NULL,
	NULL
};

} // extern "C"

#ifdef _WIN32
static void
trans_func(unsigned int u, EXCEPTION_POINTERS* /*pExp*/)
{
    throw wrappy::SE_Exception(u);
}
#endif

PyMODINIT_FUNC
PyInit_libwrappy2()
{
#ifdef _WIN32
	// TODO: initialize once per thread
	_set_se_translator(trans_func);
#endif

	static bool initialized = false;
	if (initialized)
		return NULL;

	PyObject* module = PyModule_Create(&moduledef);
	if (module == NULL)
		return NULL;

	wrappy::Mutable_Type.tp_base = &PyType_Type;
	Py_INCREF(wrappy::Mutable_Type.tp_base);
	wrappy::Mutable_Type.tp_new = PyType_Type.tp_new;
	wrappy::Mutable_Type.tp_dealloc = PyType_Type.tp_dealloc;
	wrappy::Mutable_Type.tp_free = PyType_Type.tp_free;
	wrappy::Mutable_Type.tp_is_gc = PyType_Type.tp_is_gc;
	wrappy::Mutable_Type.tp_traverse = PyType_Type.tp_traverse;
	wrappy::Mutable_Type.tp_clear = PyType_Type.tp_clear;
	(void) PyType_Ready(&wrappy::Mutable_Type);
	PyModule_AddObject(module, "mutable",
		reinterpret_cast<PyObject*>(&wrappy::Mutable_Type));

	(void) wrappy::MutableType_Ready(&wrappy::Object_Type);
	PyModule_AddObject(module, "WrapPy",
		reinterpret_cast<PyObject*>(&wrappy::Object_Type));

	initialized = true;
	return module;
}

#ifdef __APPLE__
#ifdef OSX_NEED_INIT
extern "C"
void callStaticInitWrappy2()
{
	void __initialize_Cplusplus();
	__initialize_Cplusplus();
}
#endif
#endif
