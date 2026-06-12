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
#include <cassert>
#include <arrays/pythonarray.h>
extern "C" {
#include "xdrfile_xtc.h"
#include "xdrfile_trr.h"
} // extern "C"

#define PY_STUPID (char*)


#define ERROR_RETURN(msg) { sprintf(error_string, msg); PyErr_SetString(PyExc_ValueError, error_string); return NULL; }
#define ERROR_RETURN3(msg, arg1, arg2) { sprintf(error_string, msg, arg1, arg2); PyErr_SetString(PyExc_ValueError, error_string); return NULL; }

#define RVEC_AS_DOUBLE 1

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

#ifdef RVEC_AS_DOUBLE
	static_assert(sizeof(rvec) == 3 * sizeof(double), "rvec has not been redefined to double[3]");
#else
	static_assert(sizeof(rvec) == 3 * sizeof(float), "rvec has been redefined to double[3]");
#endif

	XDRFILE *xd = xdrfile_open(file_name, "r");
	if (xd == NULL)
		ERROR_RETURN("xdrfile_open failure");
	
	// Despite holding two copies of the coordinates in memory at once, this implementation's peak
	// memory use is no worse (and might be better) that one that read the trajectory twice, once
	// to get the number of frames (to allocate a correctly sized numpy array) and a second time to
	// actually transfer the coordinates into the array.  And that implementaion too twice as long!
	std::vector<rvec*> frame_list;
	int step;
	float time, precision;
	matrix box;
	float lambda;
	do {
		rvec *gromacs_crds = (rvec *)malloc(num_atoms * sizeof(rvec));
		if (gromacs_crds == nullptr) {
			for (auto crds: frame_list) free(crds);
			ERROR_RETURN("Cannot allocate memory for coordinates");
		}
		frame_list.push_back(gromacs_crds);
		if (is_xtc)
			status = read_xtc(xd, num_atoms, &step, &time, box, gromacs_crds, &precision);
		else
			status = read_trr(xd, num_atoms, &step, &time, &lambda, box, gromacs_crds, NULL, NULL);
		if (status != exdrOK) {
			if (is_xtc) {
				if (status != exdrENDOFFILE) {
					for (auto crds: frame_list) free(crds);
					xdrfile_close(xd);
					ERROR_RETURN3("read_%s failure; return code %d", format, status);
				}
			} else {
				status = exdrENDOFFILE; // trr doesn't return proper end-of-file status
			}
		}
	} while (status == exdrOK);
	free(frame_list.back());
	frame_list.pop_back();

	double* crd_ptr;
	auto crd_array = python_double_array(frame_list.size(), num_atoms, 3, &crd_ptr);
	for (auto crds: frame_list) {
#ifdef RVEC_AS_DOUBLE
		auto xfer_ptr = reinterpret_cast<double*>(crds);
#else
		auto xfer_ptr = reinterpret_cast<float*>(crds);
#endif
		for (int i=0; i<3*num_atoms; ++i) {
			*crd_ptr++ = *xfer_ptr++;
		}
		free(crds);
	}
	return crd_array;
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
