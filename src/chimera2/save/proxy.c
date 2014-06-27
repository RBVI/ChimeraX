#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

/* CACHE_ARRAYS trades memory for speed, but doesn't appear to help much */
#undef CACHE_ARRAYS

#define PY_STUPID (char *)

static PyTypeObject ProxyType;

typedef struct {
	PyObject_HEAD
	PyObject	*weakreflist;
	PyObject	*aggregate;
#ifdef CACHE_ARRAYS
	PyObject	*arrays;	/* internal */
#endif
	Py_ssize_t	index;
} Proxy;

static int
Proxy_clear(PyObject *inst)
{
	Proxy *self = (Proxy *) inst;
	PyObject *tmp;

	tmp = self->aggregate;
	self->aggregate = NULL;
	Py_XDECREF(tmp);

#ifdef CACHE_ARRAYS
	tmp = self->arrays;
	self->arrays = NULL;
	Py_XDECREF(tmp);
#endif

	if (self->weakreflist != NULL)
		PyObject_ClearWeakRefs(inst);

	return 0;
}

static void
Proxy_dealloc(PyObject *inst)
{
	Proxy_clear(inst);
	Py_TYPE(inst)->tp_free(inst);
}

static PyObject *
Proxy_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	Proxy *self;

	self = (Proxy *) type->tp_alloc(type, 0);
	if (self != NULL) {
		self->weakreflist = NULL;
		Py_INCREF(Py_None);
		self->aggregate = Py_None;
#ifdef CACHE_ARRAYS
		self->arrays = NULL;
#endif
		self->index = 0;
	}

	return (PyObject *) self;
}

static PyObject *arrays_str = NULL;

static int
Proxy_init(PyObject *inst, PyObject *args, PyObject *kwds)
{
	static const char *kwlist[] = {"aggregate", "index", NULL};
	PyObject *aggregate = NULL, *tmp;

	if (arrays_str == NULL)
		arrays_str = PyString_InternFromString("arrays");

	Proxy *self = (Proxy *) inst;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "On", (char **) kwlist, 
						&aggregate, &self->index))
		return -1; 

	if (self->index < 0) {
		PyErr_SetString(PyExc_ValueError, "index must be nonnegative");
		return -1;
	}

	if (aggregate) {
#ifdef CACHE_ARRAYS
		self->arrays = PyObject_GetAttr(aggregate, arrays_str);
		if (self->arrays == NULL) {
			PyErr_SetString(PyExc_ValueError,
				"missing arrays member of aggregate");
			return -1;
		}
#endif
		tmp = self->aggregate;
		Py_INCREF(aggregate);
		self->aggregate = aggregate;
		Py_XDECREF(tmp);
	}
	return 0;
}

static int
Proxy_traverse(PyObject *inst, visitproc visit, void *arg)
{
	Proxy *self = (Proxy *) inst;
	Py_VISIT(self->aggregate);
	return 0;
}

static PyObject *
Proxy_getattro(PyObject *inst, PyObject *name)
{
	Proxy *self = (Proxy *) inst;
#ifndef CACHE_ARRAYS
	PyObject *arrays = NULL;
#endif
	PyObject *array, *result = NULL;
	const char *name_cstr;

	if (!PyString_Check(name)) {
#ifdef Py_USING_UNICODE
		/* The Unicode to string conversion is done here because the
		   existing tp_setattro slots expect a string object as name
		   and we wouldn't want to break those. */
		if (PyUnicode_Check(name)) {
			name = PyUnicode_AsEncodedString(name, NULL, NULL);
			if (name == NULL)
				return NULL;
		} else
#endif
		{
			PyErr_Format(PyExc_TypeError,
				"attribute name must be string, not '%.200s'",
				Py_TYPE(name)->tp_name);
			return NULL;
		}
	} else
		Py_INCREF(name);
	name_cstr = PyString_AS_STRING(name);

	if (name_cstr[0] == '_') {
		if (name_cstr[1] == 'P') {
			if (strcmp(name_cstr + 2, "roxy_index") == 0) {
				result = PyInt_FromSsize_t(self->index);
				goto done;
			}
			if (strcmp(name_cstr + 2, "roxy_aggregate") == 0) {
				Py_INCREF(self->aggregate);
				result = self->aggregate;
				goto done;
			}
		} if (name_cstr[1] == '_') {
			if (strcmp(name_cstr + 2, "class__") == 0) {
				result = (PyObject *) Py_TYPE(inst);
				Py_INCREF(result);
				goto done;
			} else if (strcmp(name_cstr + 2, "doc__") == 0) {
				const char *doc = Py_TYPE(self)->tp_doc;
				if (doc != NULL) {
					result = PyString_FromString(doc);
					goto done;
				}
			}
			PyErr_SetObject(PyExc_AttributeError, name);
			goto done;
		}
	} else if (name_cstr[0] == 'c' && strcmp(name_cstr + 1, "ontainer") == 0) {
		result = PyObject_GetAttr(self->aggregate, name);
		goto done;
	}

#ifdef CACHE_ARRAYS
	array = PyDict_GetItem(self->arrays, name);
#else
	arrays = PyObject_GetAttr(self->aggregate, arrays_str);
	if (arrays == NULL) {
		PyErr_SetObject(PyExc_AttributeError, name);
		goto done;
	}
	array = PyDict_GetItem(arrays, name);
#endif
	if (array == NULL) {
		PyErr_SetObject(PyExc_AttributeError, name);
		goto done;
	}
	result = PySequence_GetItem(array, self->index);
done:
#ifndef CACHE_ARRAYS
	Py_XDECREF(arrays);
#endif
	Py_DECREF(name);
	return result;
}

static int
Proxy_setattro(PyObject *inst, PyObject *name, PyObject *value)
{
	Proxy *self = (Proxy *) inst;
#ifndef CACHE_ARRAYS
	PyObject *arrays = NULL;
#endif
	PyObject *array;
	const char *name_cstr;
	int result = -1;

	if (!PyString_Check(name)) {
#ifdef Py_USING_UNICODE
		/* The Unicode to string conversion is done here because the
		   existing tp_setattro slots expect a string object as name
		   and we wouldn't want to break those. */
		if (PyUnicode_Check(name)) {
			name = PyUnicode_AsEncodedString(name, NULL, NULL);
			if (name == NULL)
				return -1;
		} else
#endif
		{
			PyErr_Format(PyExc_TypeError,
				"attribute name must be string, not '%.200s'",
				Py_TYPE(name)->tp_name);
			return -1;
		}
	} else
		Py_INCREF(name);
	name_cstr = PyString_AS_STRING(name);

	if (self->index == PY_SSIZE_T_MAX) {
		PyErr_SetString(PyExc_ValueError,
					"proxied object has been deleted");
		goto done;
	}
	if (value == NULL) {
		/* del attribute */
		if (name_cstr[0] == '_' && name_cstr[1] == '_') {
			PyErr_SetString(PyExc_TypeError,
					"can not del private attributes");
			goto done;
		}
#ifdef CACHE_ARRAYS
		array = PyDict_GetItem(self->arrays, name);
#else
		arrays = PyObject_GetAttr(self->aggregate, arrays_str);
		if (arrays == NULL) {
			PyErr_SetObject(PyExc_AttributeError, name);
			goto done;
		}
		array = PyDict_GetItem(arrays, name);
#endif
		if (array == NULL) {
			PyErr_SetObject(PyExc_AttributeError, name);
			goto done;
		}
#ifdef TODO
		if isinsance(array, numpy.ma.masked_array):
		    array[self._Proxy_index] = numpy.ma.masked
		else:
		    raise ValueError("can not del required attributes")
#else
		PyErr_SetString(PyExc_NotImplementedError,
					"cannot del proxied attributes");
		goto done;
#endif
	}

	if (name_cstr[0] == '_' && name_cstr[1] == '_') {
		PyErr_SetString(PyExc_TypeError,
					"can not set private attributes");
		goto done;
	}
	if (name_cstr[0] == 'c' && strcmp(name_cstr + 1, "ontainer") == 0) {
		/*TODO: assert value == self._Proxy_aggregate.container*/
		result = 0;
		goto done;
	}
#ifdef CACHE_ARRAYS
	if (!PyDict_Contains(self->arrays, name))
#else
	arrays = PyObject_GetAttr(self->aggregate, arrays_str);
	if (arrays == NULL) {
		PyErr_SetObject(PyExc_AttributeError, name);
		goto done;
	}
	if (!PyDict_Contains(arrays, name))
#endif
	{
		/* unknown attribute, add it as an object */
		PyObject *res;

		PyArray_Descr *dtype = PyArray_DescrFromType(NPY_OBJECT);
		PyObject *info = Py_BuildValue("{OO}", name, dtype);
		res = PyObject_CallMethod(self->aggregate, PY_STUPID "register", PY_STUPID "O", info);
		Py_DECREF(res);
		Py_DECREF(info);
	}
#ifdef CACHE_ARRAYS
	array = PyDict_GetItem(self->arrays, name);
#else
	array = PyDict_GetItem(arrays, name);
#endif
	if (array == NULL) {
		PyErr_SetObject(PyExc_AttributeError, name);
		goto done;
	}
        /*assert self._Proxy_index < self._Proxy_aggregate.size*/
	result = PySequence_SetItem(array, self->index, value);
done:
#ifndef CACHE_ARRAYS
	Py_XDECREF(arrays);
#endif
	Py_DECREF(name);
	return result;
}

static PyMemberDef Proxy_members[] = {
	{ PY_STUPID "_Proxy_aggregate", T_OBJECT_EX, offsetof(Proxy, aggregate),
		READONLY, PY_STUPID "proxy container"},
	{ PY_STUPID "_Proxy_index", T_PYSSIZET, offsetof(Proxy, index), 0,
		PY_STUPID "array index"},
	{NULL}	/* Sentinel */
};

static PyTypeObject ProxyType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"_proxy.Proxy",			/* tp_name*/
	sizeof (Proxy),			/* tp_basicsize */
	0,				/* tp_itemsize */
	Proxy_dealloc,			/* tp_dealloc */
	0,				/* tp_print */
	0,				/* tp_getattr */
	0,				/* tp_setattr */
	0,				/* tp_compare */
	0,				/* tp_repr */
	0,				/* tp_as_number */
	0,				/* tp_as_sequence */
	0,				/* tp_as_mapping */
	0,				/* tp_hash */
	0,				/* tp_call */
	0,				/* tp_str */
	Proxy_getattro,			/* tp_getattro */
	Proxy_setattro,			/* tp_setattro */
	0,				/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
	"Proxy objects",		/* tp_doc */
	Proxy_traverse,			/* tp_traverse */
	Proxy_clear,			/* tp_clear */
	0,				/* tp_richcompare */
	offsetof(Proxy, weakreflist),	/* tp_weaklistoffset */
	0,				/* tp_iter */
	0,				/* tp_iternext */
	0,				/* tp_methods */
	Proxy_members,			/* tp_members */
	0,				/* tp_getset */
	0,				/* tp_base */
	0,				/* tp_dict */
	0,				/* tp_descr_get */
	0,				/* tp_descr_set */
	0,				/* tp_dictoffset */
	Proxy_init,			/* tp_init */
	0,				/* tp_alloc */
	Proxy_new,			/* tp_new */
};

static PyMethodDef _proxy_methods[] = {
	{NULL}	/* Sentinel */
};

PyMODINIT_FUNC
init_proxy(void)
{
	PyObject *m;

	if (PyType_Ready(&ProxyType) < 0)
		return;

	m = Py_InitModule("_proxy", _proxy_methods);
	if (m == NULL)
		return;

	Py_INCREF(&ProxyType);
	PyModule_AddObject(m, "Proxy", (PyObject *) &ProxyType);

	import_array();		/* initialize access to numpy functions */
}
