/* include Python.h first, so that 64-bit file support is picked up & defined properly
   before include stdio.h */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <vector>
#include <iostream>
extern "C" {
#include "xdrfile_xtc.h"
#include "xdrfile_trr.h"
} // extern "C"

#define PY_STUPID (char*)


#define ERROR_RETURN(msg) { sprintf(error_string, msg); PyErr_SetString(PyExc_ValueError, error_string); return NULL; }
#define ERROR_RETURN3(msg, arg1, arg2) { sprintf(error_string, msg, arg1, arg2); PyErr_SetString(PyExc_ValueError, error_string); return NULL; }
#define FREE_CRDS { for (std::vector<rvec *>::iterator ci=coords_by_frame.begin(); ci != coords_by_frame.end(); ++ci) { free(*ci); } }

static void
dealloc_array(PyObject *capsule)
{
	void *ptr = PyCapsule_GetPointer(capsule, NULL);
	if (ptr != NULL) {
		free(ptr);
	}
}
	
static PyObject *
read_traj_file(PyObject *args, bool is_xtc)
{
	char error_string[256];

	char *file_name;
	if (!PyArg_ParseTuple(args, PY_STUPID "s", &file_name)) {
		if (is_xtc) {
			ERROR_RETURN("readXtcFile: could not parse args");
		} else {
			ERROR_RETURN("readTrrFile: could not parse args");
		}
	}

	int num_atoms, status;
	const char *format;
	if (is_xtc) {
		status = read_xtc_natoms(file_name, &num_atoms);
		format = "xtc";
	} else {
		status = read_trr_natoms(file_name, &num_atoms);
		format = "trr";
	}
	if (status != exdrOK)
		ERROR_RETURN3("read_%s_natoms failure; return code %d", format, status);

	XDRFILE *xd = xdrfile_open(file_name, "r");
	if (xd == NULL)
		ERROR_RETURN("xdrfile_open failure");

	// read frames into float arrays
	std::vector<rvec*> coords_by_frame;
	int step;
	float time, precision;
	matrix box;
	float lambda;
	do {
		rvec *crds = (rvec *)malloc(num_atoms * sizeof(rvec));
		if (crds == NULL) {
			FREE_CRDS;
			ERROR_RETURN("Couldn't allocate enough memory for coords");
		}
		if (is_xtc)
			status = read_xtc(xd, num_atoms, &step, &time, box, crds, &precision);
		else
			status = read_trr(xd, num_atoms, &step, &time, &lambda, box, crds, NULL, NULL);
		if (status != exdrOK) {
			if (is_xtc) {
				if (status != exdrENDOFFILE) {
					FREE_CRDS;
					free(crds);
					ERROR_RETURN3("read_%s failure; return code %d", format, status);
				}
			} else {
				free(crds);
				status = exdrENDOFFILE; // trr doesn't return proper end-of-file status
			}
		}
		if (status != exdrENDOFFILE) {
			coords_by_frame.push_back(crds);
		}
	} while (status == exdrOK);

	// convert vector of float arrays into Python list of numpy objects
	// allocated memory use is transferred, not duplicated
	// (PyArray_SimpleNewFromData)
	PyObject *crd_list = PyList_New(0);
	if (crd_list == NULL) {
		FREE_CRDS;
		ERROR_RETURN("Couldn't create Python list for coords");
	}
	npy_intp dimensions[2];
	dimensions[0] = num_atoms;
	dimensions[1] = 3;
	for (std::vector<rvec *>::iterator ci=coords_by_frame.begin();
			ci != coords_by_frame.end(); ++ci) {
		PyObject *array = PyArray_SimpleNewFromData(2, dimensions, NPY_FLOAT, *ci);
		PyObject *mem_controller = PyCapsule_New(*ci, NULL, dealloc_array);
		if (PyArray_SetBaseObject((PyArrayObject *)array, mem_controller) < 0) {
			Py_XDECREF(array);
			Py_XDECREF(mem_controller);
			FREE_CRDS;
			ERROR_RETURN("Cannot set array base object");
		}
		if (PyList_Append(crd_list, array) < 0) {
			Py_DECREF(array);
			Py_DECREF(mem_controller);
			FREE_CRDS;
			// exception set by PyList_Append call
			return NULL;
		}
		Py_DECREF(array);
	}

	return Py_BuildValue(PY_STUPID "iO", num_atoms, crd_list);
}

static PyObject *
readXtcFile(PyObject *, PyObject *args)
{
	return read_traj_file(args, true);
}

static PyObject *
readTrrFile(PyObject *, PyObject *args)
{
	return read_traj_file(args, false);
}


static PyMethodDef Methods[] =
{
	{PY_STUPID "read_xtc_file", readXtcFile, METH_VARARGS, NULL},
	{PY_STUPID "read_trr_file", readTrrFile, METH_VARARGS, NULL},
	{nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef gromacs_def =
{
	PyModuleDef_HEAD_INIT,
	"_gromacs",
	"Read Gromacs XTC/TRR coordinates",
	-1,
	Methods,
	nullptr,
	nullptr,
	nullptr,
	nullptr
};

PyMODINIT_FUNC
PyInit__gromacs(void)
{
  import_array();
  return PyModule_Create(&gromacs_def);
}
