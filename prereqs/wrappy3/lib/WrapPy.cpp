// Copyright (c) 1998-2000 The Regents of the University of California.
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

// $Id: WrapPy.cpp 29912 2010-01-28 01:31:29Z gregc $

#define WRAPPY_EXPORT
#include "WrapPy3.h"
#include <stdexcept>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char *)
#endif

namespace wrappy {

bool
Float_Check(PyObject *o)
{
	return o && o->ob_type->tp_as_number
					&& o->ob_type->tp_as_number->nb_float;
}

bool
Long_Check(PyObject *o)
{
	return o && o->ob_type->tp_as_number
					&& o->ob_type->tp_as_number->nb_int;
}

PyObject*
Obj::wpyNew() const
{
	throw std::runtime_error("missing specialization of wrappy::Obj::wpyNew");
}

PyObject*
Obj::wpyGetObject(Create pwc) const
{
	if (pyObj) {
		if (pwc == PWC_CREATE_AND_OWN)
			owner = true;
		else if (pyObj->ob_refcnt == 0)
			// resurrection!
			_Py_NewReference(pyObj);
		else
			Py_INCREF(pyObj);
		return pyObj;
	}
	if (pwc == PWC_DONT_CREATE)
		return NULL;
	pyObj = wpyNew();		// this should throw if an error
	if (pwc == PWC_CREATE_AND_OWN)
		owner = true;
	else
		Py_XINCREF(pyObj);	// just in case use XINCREF
	return pyObj;
}

Obj::~Obj()
{
	if (pyObj == NULL)
		return;

	// called by wrapped class destructor
	PyObject* saveObj = pyObj;
	wpyDisassociate();
	// Note: the cast below is only safe because wrappy guarantees that
	// all objects from classes that subclass from wrappy::Obj have an
	// instance dictionary
	Object* self = static_cast<Object*>(saveObj);
	self->_inst_data = NULL;
#if 0
	// Bad idea because instance dictionary may hold stuff
	// (e.g., a Tk widget) that needs to be explicitly cleaned up.
	if (self->_inst_dict)
		PyDict_Clear(self->_inst_dict);
#else
	// clear cached attributes
	if (self->_inst_dict) {
		Py_ssize_t pos = 0;
		PyObject* key = 0;
		while (PyDict_Next(self->_inst_dict, &pos, &key, NULL)) {
			if (!PyUnicode_Check(key))
				continue;
			char const* s = reinterpret_cast<char *>(PyUnicode_DATA(key));
			if (s[0] == '_' && strncmp(s, "__cached_", 9) == 0)
				PyDict_DelItem(self->_inst_dict, key);
		}
	}
#endif
	if (!owner)
		Py_DECREF(saveObj);
}

void
Obj::wpyDisassociate()
{
	// only called in Python class' destructor method
	pyObj = NULL;
}

void
Obj::wpyAssociate(PyObject* o) const
{
	// only called by a Python class' __new__ method
	pyObj = o;
}


extern "C" {

PyObject*
Obj_destroyed(PyObject* obj, void*)
{
	Object* self = static_cast<Object*>(obj);
	return PyBool_FromLong(self->_inst_data == NULL);
}

PyGetSetDef Obj_getset[] = {
	{
		PY_STUPID "__destroyed__", Obj_destroyed, NULL,
		PY_STUPID "true if underlying C++ object has disappeared", NULL
	},
	{ NULL, NULL, NULL, NULL, NULL }
};

PyObject*
Obj_Obj(PyTypeObject* _type, PyObject*, PyObject*)
{
	PyErr_Format(PyExc_TypeError, "cannot create '%.100s' instances",
								_type->tp_name);
	return NULL;
}

const char Obj_doc[] = "\n\
Not instantiable from Python";

PyTypeObject Object_Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"libwrappy2.Obj", // tp_name
	sizeof (Object), // tp_basicsize
	0, // tp_itemsize
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
	0, // tp_getattro
	0, // tp_setattro
	0, // tp_as_buffer
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE
		| Py_TPFLAGS_IS_ABSTRACT, // tp_flags
	Obj_doc, // tp_doc
	0, // tp_traverse
	0, // tp_clear
	0, // tp_richcompare
	0, // tp_weaklistoffset
	0, // tp_tp_iter
	0, // tp_iternext
	0, // tp_methods
	0, // tp_members
	Obj_getset, // tp_getset
	0, // tp_base
	0, // tp_dict
	0, // tp_descr_get
	0, // tp_descr_set
	0, // tp_dictoffset
	0, // tp_init
	0, // tp_alloc
	Obj_Obj, // tp_new
	0, // tp_free
	0, // tp_is_gc
	0, // tp_bases
	0, // tp_mro
	0, // tp_cache
	0, // tp_subclasses
	0, // tp_weaklist
	0, // tp_del
};

} // extern "C"

} // namespace wrappy
