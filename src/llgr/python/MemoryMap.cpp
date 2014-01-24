#include "MemoryMap.h"
#define PY_SSIZE_T_CLEAN 1
#include <Python.h>
#include <stdexcept>
#include <sstream>
#include <ctype.h>

namespace llgr {

namespace {

/* simple buffer-like object */
struct MemoryMap: public PyObject {
	unsigned char	*buf;
	Py_ssize_t	len;
};

} // namespace

extern "C" void
MemoryMap_dealloc(PyObject *self)
{
	self->ob_type->tp_free(self);
}

extern "C" Py_ssize_t
MemoryMap_length(PyObject *self)
{
	MemoryMap *mm = static_cast<MemoryMap *>(self);
	return mm->len;
}

extern "C" PyObject *
MemoryMap_subscript(PyObject *self, PyObject *item)
{
	MemoryMap *mm = static_cast<MemoryMap *>(self);

	if (PyIndex_Check(item)) {
		Py_ssize_t i = PyNumber_AsSsize_t(item, PyExc_IndexError);
		if (i == -1 && PyErr_Occurred())
			return NULL;
		if (i < 0)
			i += mm->len;
		if (i < 0 || i >= mm->len) {
			PyErr_SetString(PyExc_IndexError,
					"MemoryMap index out of range");
			return NULL;
		}
		unsigned char *ptr = mm->buf + i;
		return PyLong_FromLong(*ptr);
	}

	if (PySlice_Check(item) == 0) {
		PyErr_Format(PyExc_TypeError,
			     "MemoryMap indices must be integers, not %200s",
			     item->ob_type->tp_name);
		return NULL;
	}

	// must be a slice!
	Py_ssize_t start, stop, step, slice_length;
	if (PySlice_GetIndicesEx(item, mm->len,
			       &start, &stop, &step, &slice_length) < 0) {
		return NULL;
	}
	if (step != 1) {
		PyErr_SetString(PyExc_ValueError,
				"only slices with step 1 are supported");
		return NULL;
	}
	return PyBytes_FromStringAndSize(
		 reinterpret_cast<char *>(mm->buf) + start, stop - start);
}

extern "C" int
MemoryMap_ass_subscript(PyObject *self, PyObject *item, PyObject *value)
{
	MemoryMap *mm = static_cast<MemoryMap *>(self);
	if (value == NULL) {
		PyErr_SetString(PyExc_ValueError,
					"MemoryMap's can not change size");
		return -1;
	}

	if (PyIndex_Check(item)) {
		Py_ssize_t i = PyNumber_AsSsize_t(item, PyExc_IndexError);
		if (i == -1 && PyErr_Occurred())
			return -1;
		if (i < 0)
			i += mm->len;
		if (i < 0 || i >= mm->len) {
			PyErr_SetString(PyExc_IndexError,
					 "MemoryMap assignment index out of range");
			return -1;
		}
		unsigned char *ptr = mm->buf + i;
		long iv = PyLong_AsLong(value);
		if (iv == -1 && PyErr_Occurred())
			return -1;
		*ptr = static_cast<unsigned char>(iv);
		return 0;
	}

	if (PySlice_Check(item) == 0) {
		PyErr_Format(PyExc_TypeError,
			     "MemoryMap indices must be integers, not %200s",
			     item->ob_type->tp_name);
		return -1;
	}

	// must be a slice!
	Py_ssize_t start, stop, step, slice_length;
	if (PySlice_GetIndicesEx(item, mm->len,
			       &start, &stop, &step, &slice_length) < 0) {
		return -1;
	}
	if (step != 1) {
		PyErr_SetString(PyExc_ValueError,
				"only slices with step 1 are supported");
		return -1;
	}

	Py_buffer buffer;
	if (PyObject_GetBuffer(value, &buffer, PyBUF_ANY_CONTIGUOUS) == 0) {
		// expected contiguous bytes or buffer
		return -1;
	}
	if (buffer.len != stop - start) {
		PyErr_SetString(PyExc_ValueError,
					"MemoryMap's can not change size");
		PyBuffer_Release(&buffer);
		return -1;
	}
	unsigned char *dest = mm->buf + start;
	memcpy(dest, buffer.buf, buffer.len);
	PyBuffer_Release(&buffer);
	return 0;
}

static PyMappingMethods MemoryMap_as_mapping = {
	MemoryMap_length,		// sq_length
	MemoryMap_subscript,		// sq_item
	MemoryMap_ass_subscript		// sq_ass_item
};

extern "C" PyObject *
MemoryMap_repr(PyObject *self)
{
	MemoryMap *mm = static_cast<MemoryMap *>(self);
	std::ostringstream buf;
	buf << "MemoryMap(b'";
	for (int i = 0; i < mm->len; ++i) {
		unsigned char *ptr = mm->buf + i;
		if (isascii(*ptr) && isprint(*ptr))
			buf << *ptr;
		else
			buf << std::hex << static_cast<unsigned>(*ptr);
	}
	buf << "')";
	const std::string &s = buf.str();
	return PyUnicode_FromStringAndSize(s.c_str(), s.size());
}

const char MemoryMap_doc[] = "Provide memory map";

static PyObject*
MemoryMap_copyfrom(PyObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = { "offset", "buffer", NULL };
	MemoryMap *mm = static_cast<MemoryMap *>(self);
	Py_ssize_t start;
	Py_buffer buffer;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ny*", kwlist,
					 	&start, &buffer)) {
		return NULL;
	}

	if (!PyBuffer_IsContiguous(&buffer, 'C')) {
		PyBuffer_Release(&buffer);
		PyErr_SetString(PyExc_ValueError, "buffer must be contigious");
		return NULL;
	}
	if (start + buffer.len > mm->len) {
		PyBuffer_Release(&buffer);
		PyErr_SetString(PyExc_ValueError, "copy would overflow memorymap");
		return NULL;
	}

	unsigned char *dest = mm->buf + start;
	memcpy(dest, buffer.buf, buffer.len);
	PyBuffer_Release(&buffer);
	return (Py_INCREF(Py_None), Py_None);
}

const char MemoryMap_copyfrom_doc[] = "At offset copy from buffer-like object";

static PyMethodDef MemoryMap_methods[] = {
	{ "copyfrom", (PyCFunction) MemoryMap_copyfrom,
		METH_VARARGS | METH_KEYWORDS, MemoryMap_copyfrom_doc },
	{ NULL, NULL, 0, NULL }
};

PyTypeObject MemoryMapType = {
	PyVarObject_HEAD_INIT(&PyType_Type, 0)
	"MemoryMap",			// tp_name
	sizeof (MemoryMap),		// tp_basicsize
	0,				// tp_itemsize
	MemoryMap_dealloc,		// tp_dealloc
	0,				// tp_print
	0,				// tp_getattr
	0,				// tp_setattr
	0,				// tp_compare
	MemoryMap_repr,			// tp_repr
	0,				// tp_as_number
	0,				// tp_as_sequence
	&MemoryMap_as_mapping,		// tp_as_mapping
	0, 				// tp_hash
	0,				// tp_call
	0,				// tp_str
	0,				// tp_getattro
	0,				// tp_setattro
	0,				// tp_as_buffer
	0,				// tp_xxx4
	MemoryMap_doc,			// tp_doc
	0,				// tp_traverse
	0,				// tp_clear
	0,				// tp_richcompare
	0,				// tp_weaklistoffset
	0,				// tp_iter
	0,				// tp_iternext
	MemoryMap_methods,		// tp_methods
	0,				// tp_members
	0,				// tp_getset
	0,				// tp_base
	0,				// tp_dict
	0,				// tp_descr_get
	0,				// tp_descr_set
	0,				// tp_dictoffset
	0,				// tp_init
	0,				// tp_alloc
	0,				// tp_new
	PyObject_Del,			// tp_free
	0,				// tp_is_gc
	0,				// tp_bases
	0,				// tp_mro
	0,				// tp_cache
	0,				// tp_subclasses
	0,				// tp_weaklist
	0,				// tp_del
	0,				// tp_version_tag
};

#if 0
inline bool
MemoryMapTypeCheck(PyObject *obj)
{
	return (obj)->ob_type == &MemoryMapType;
}
#endif

PyObject *
memory_map(unsigned char *data, Py_ssize_t len)
{
	static bool type_init;

	if (!type_init) {
		// Since wrappy doesn't see the MemoryMap class,
		// it doesn't know to make it ready.
		PyType_Ready(&MemoryMapType);
		type_init = true;
	}
	MemoryMap *mm = PyObject_NEW(MemoryMap, &MemoryMapType);
	if (mm == NULL)
		return NULL;
	mm->len = len;
	mm->buf = data;
	if (mm->buf == NULL)
		mm->len = 0;		// sanity check
	return mm;
}

} // namespace llgr
