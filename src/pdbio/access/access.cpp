#include "capsule/capsule.h"
#include "molecule/Molecule.h"
#include "molecule/Residue.h"
#include "molecule/Atom.h"
#include "molecule/Bond.h"
#include "base-geom/Coord.h"
#include "molecule/Element.h"
#include <vector>
#include <map>
#include <Python.h>
#include <stdexcept>
#include <sstream>  // std::ostringstream

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>  // use PyArray_*(), NPY_*

// Need to call NumPy import_array() before using NumPy routines
static void *
initialize_numpy()
{
	static bool first_call = true;
	if (first_call) {
		first_call = false;
		import_array();
	}
	return NULL;
}

static const char *
numpy_type_name(int type)
{
	const char *name = "unknown";
	switch (type) {
		case NPY_BOOL: name = "bool"; break;
		case NPY_BYTE: name = "byte"; break;
		case NPY_UBYTE: name = "ubyte"; break;
		case NPY_SHORT: name = "short"; break;
		case NPY_USHORT: name = "ushort"; break;
		case NPY_INT: name = "int"; break;
		case NPY_UINT: name = "uint"; break;
		case NPY_LONG: name = "long"; break;
		case NPY_ULONG: name = "ulong"; break;
		case NPY_LONGLONG: name = "longlong"; break;
		case NPY_ULONGLONG: name = "ulonglong"; break;
		case NPY_FLOAT: name = "float"; break;
		case NPY_DOUBLE: name = "double"; break;
		case NPY_LONGDOUBLE: name = "longdouble"; break;
		case NPY_CFLOAT: name = "cfloat"; break;
		case NPY_CDOUBLE: name = "cdouble"; break;
		case NPY_CLONGDOUBLE: name = "clongdouble"; break;
		case NPY_OBJECT: name = "object"; break;
		case NPY_STRING: name = "string"; break;
		case NPY_UNICODE: name = "unicode"; break;
		case NPY_VOID: name = "void"; break;
		default: break;
	}
	return name;
}

static PyObject *
allocate_python_array(unsigned int dim, unsigned int *size, int type)
{
	npy_intp *sn = new npy_intp[dim];
	for (int i = 0; i < dim; ++i)
		sn[i] = (npy_intp)size[i];
	
	// array not initialized to zero
	PyObject *array = PyArray_SimpleNew(dim, sn, type);
	delete [] sn;
	if (array == NULL) {
		std::ostringstream msg;
		msg << numpy_type_name(type) << " array allocation of size (";
		for (int i = 0; i < dim; ++i) {
			msg << size[i] << (i < dim-1 ? ", " : "");
		}
		msg << ") failed " << std::endl;
		throw std::runtime_error(msg.str());
	}
	return array;
}

static PyObject *
allocate_python_array(unsigned int dim, unsigned int *size, PyArray_Descr *dtype)
{
	npy_intp *sn = new npy_intp[dim];
	for (int i = 0; i < dim; ++i)
		sn[i] = (npy_intp)size[i];
	
	// array not initialized to zero
	PyObject *array = PyArray_SimpleNewFromDescr(dim, sn, dtype);
	delete [] sn;
	if (array == NULL) {
		std::ostringstream msg;
		msg << "Array allocation of size (";
		for (int i = 0; i < dim; ++i) {
			msg << size[i] << (i < dim-1 ? ", " : "");
		}
		msg << ") failed " << std::endl;
		throw std::runtime_error(msg.str());
	}
	return array;
}

static PyObject *
python_string_array(unsigned int size, int string_length, char **data)
{
	initialize_numpy();  // required before using NumPy

	PyArray_Descr *d = PyArray_DescrNewFromType(NPY_CHAR);
	d->elsize = string_length;
	unsigned int dimensions[1] = {size};
	PyObject *array = allocate_python_array(1, dimensions, d);
	if (data)
		*data = (char *)PyArray_DATA((PyArrayObject *)array);

	return array;
}

static PyObject *
atom_names(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *atom_vec_capsule;
	int numpy = 0; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "atoms_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&atom_vec_capsule, &numpy))
		return NULL;
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms_capsule arg is not a atom-vec capsule");
		return NULL;
	}
	PyObject *atom_names;
	if (numpy) {
		initialize_numpy();
		char *data;
		atom_names = python_string_array(atoms->size(), 4, &data);
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end();
				++ai) {
			const std::string &an = (*ai)->name();
			strncpy(data, an.c_str(), 4);
			data += 4;
		}
	} else {
		atom_names = PyList_New(atoms->size());
		unsigned long i = 0;
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end();
				++ai, ++i) {
			PyList_SET_ITEM(atom_names, i, PyUnicode_FromString((*ai)->name().c_str()));
		}
	}
	return atom_names;
}

static PyObject *
atom_residues(PyObject *, PyObject *atom_vec_capsule)
{
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "first arg is not an atom-vec capsule");
		return NULL;
	}
	std::vector<Residue *> *res_vec = new std::vector<Residue *>(0);
	res_vec->reserve(atoms->size());
	for (std::vector<Atom *>::iterator ai = atoms->begin(); ai != atoms->end(); ++ai) {
		Atom *a = *ai;
		res_vec->push_back(a->residue());
	}
	return encapsulate_res_vec(res_vec);
}

static PyObject *
atoms(PyObject *, PyObject *mol_vec_capsule)
{
	std::vector<Molecule *> *mols;
	try {
		mols = decapsulate_mol_vec(mol_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "first arg is not a mol-vec capsule");
		return NULL;
	}
	int size = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		size += m->atoms().size();
	}
	std::vector<Atom *> *atom_vec = new std::vector<Atom *>(0);
	atom_vec->reserve(size);
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Atoms &m_atoms = m->atoms();
		atom_vec->insert(atom_vec->end(), m_atoms.begin(), m_atoms.end());
	}
	return encapsulate_atom_vec(atom_vec);
}

// atoms_bonds: returns an atom blob and a list of 2-tuples
static PyObject *
atoms_bonds(PyObject *, PyObject *mol_vec_capsule)
{
	std::vector<Molecule *> *mols;
	try {
		mols = decapsulate_mol_vec(mol_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms_bonds arg is not a mol-vec capsule");
		return NULL;
	}
	int atoms_size = 0, bonds_size = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		atoms_size += m->atoms().size();
		bonds_size += m->bonds().size();
	}
	std::vector<Atom *> *atom_vec = new std::vector<Atom *>(0);
	atom_vec->reserve(atoms_size);
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Atoms &m_atoms = m->atoms();
		atom_vec->insert(atom_vec->end(), m_atoms.begin(), m_atoms.end());
	}
	std::map<Atom *, unsigned long> atom_map;
	unsigned long i = 0;
	for (std::vector<Atom *>::iterator ai = atom_vec->begin();
	ai != atom_vec->end(); ++ai, ++i) {
		atom_map[*ai] = i;
	}
	PyObject *bond_list = PyList_New(bonds_size);
	i = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Bonds &m_bonds = m->bonds();
		for (Molecule::Bonds::const_iterator bi = m_bonds.begin(); bi != m_bonds.end();
		++bi) {
			Bond *b = *bi;
			const Bond::Atoms &b_atoms = b->atoms();
			PyObject *index_tuple = PyTuple_New(2);
			PyList_SET_ITEM(bond_list, i++, index_tuple);
			PyTuple_SET_ITEM(index_tuple, 0, PyLong_FromLong(atom_map[b_atoms[0]]));
			PyTuple_SET_ITEM(index_tuple, 1, PyLong_FromLong(atom_map[b_atoms[1]]));
		}
	}
	PyObject *ret_val = PyTuple_New(2);
	PyTuple_SET_ITEM(ret_val, 0, encapsulate_atom_vec(atom_vec));
	PyTuple_SET_ITEM(ret_val, 1, bond_list);
	return ret_val;
}

static PyObject *
coords(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *atom_vec_capsule;
	int numpy = 1; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "atoms_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&atom_vec_capsule, &numpy))
		return NULL;
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms_capsule arg is not an atom-vec capsule");
		return NULL;
	}
	PyObject *coords;
	if (numpy) {
		initialize_numpy();
		static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
		unsigned int shape[2] = {(unsigned int)atoms->size(), 3};
		coords = allocate_python_array(2, shape, NPY_DOUBLE);
		double *crd_data = (double *) PyArray_DATA((PyArrayObject *)coords);
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end(); ++ai) {
			const Coord &crd = (*ai)->coord();
			*crd_data++ = crd[0];
			*crd_data++ = crd[1];
			*crd_data++ = crd[2];
		}
	} else {
		coords = PyList_New(atoms->size());
		unsigned long i = 0;
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end(); ++ai, ++i) {
			const Coord &crd = (*ai)->coord();
			PyObject *crd_tuple = PyTuple_New(3);
			PyList_SET_ITEM(coords, i, crd_tuple);
			PyTuple_SET_ITEM(crd_tuple, 0, PyFloat_FromDouble(crd[0]));
			PyTuple_SET_ITEM(crd_tuple, 1, PyFloat_FromDouble(crd[1]));
			PyTuple_SET_ITEM(crd_tuple, 2, PyFloat_FromDouble(crd[2]));
		}
	}
	return coords;
}

static PyObject *
element_names(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *atom_vec_capsule;
	int numpy = 0; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "atoms_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&atom_vec_capsule, &numpy))
		return NULL;
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms_capsule arg is not a atom-vec capsule");
		return NULL;
	}
	PyObject *element_names;
	if (numpy) {
		initialize_numpy();
		char *data;
		element_names = python_string_array(atoms->size(), 2, &data);
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end();
				++ai) {
			const char *en = (*ai)->element().name();
			*data++ = *en++;
			*data++ = *en++;
		}
	} else {
		element_names = PyList_New(atoms->size());
		unsigned long i = 0;
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end();
				++ai, ++i) {
			Element element = (*ai)->element();
			PyList_SET_ITEM(element_names, i, PyUnicode_FromString(element.name()));
		}
	}
	return element_names;
}

static PyObject *
element_numbers(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *atom_vec_capsule;
	int numpy = 1; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "atoms_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&atom_vec_capsule, &numpy))
		return NULL;
	std::vector<Atom *> *atoms;
	try {
		atoms = decapsulate_atom_vec(atom_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "atoms_capsule arg is not a atom-vec capsule");
		return NULL;
	}
	PyObject *element_numbers;
	if (numpy) {
		initialize_numpy();
		static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
		unsigned int shape[1] = {(unsigned int)atoms->size()};
		element_numbers = allocate_python_array(1, shape, NPY_UBYTE);
		unsigned char *data = (unsigned char *) PyArray_DATA((PyArrayObject *)element_numbers);
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end();
				++ai) {
			*data++ = (*ai)->element().number();
		}
	} else {
		element_numbers = PyList_New(atoms->size());
		unsigned long i = 0;
		for (std::vector<Atom *>::const_iterator ai = atoms->begin(); ai != atoms->end();
				++ai, ++i) {
			Element element = (*ai)->element();
			PyList_SET_ITEM(element_numbers, i, PyLong_FromLong((long)element.number()));
		}
	}
		
	return element_numbers;
}

static PyObject *
molecules(PyObject *, PyObject *mol_vec_capsule)
{
	std::vector<Molecule *> *mols;
	try {
		mols = decapsulate_mol_vec(mol_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "first arg is not a mol-vec capsule");
		return NULL;
	}
	PyObject *molecules = PyList_New(mols->size());
	unsigned long i = 0;
	for (std::vector<Molecule *>::const_iterator mi = mols->begin(); mi != mols->end();
			++mi, ++i) {
		PyList_SET_ITEM(molecules, i,
			encapsulate_mol_vec(new std::vector<Molecule *>(1, *mi)));
	}
	return molecules;
}

static PyObject *
residue_chain_ids(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *res_vec_capsule;
	int numpy = 0; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "residues_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&res_vec_capsule, &numpy))
		return NULL;
	std::vector<Residue *> *residues;
	try {
		residues = decapsulate_res_vec(res_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "residues_capsule arg is not a residue-vec capsule");
		return NULL;
	}
	PyObject *chain_ids;
	if (numpy) {
		initialize_numpy();
		char *data;
		chain_ids = python_string_array(residues->size(), 1, &data);
		for (std::vector<Residue *>::const_iterator ri = residues->begin();
				ri != residues->end(); ++ri) {
			const std::string &rn = (*ri)->chain_id();
			strncpy(data, rn.c_str(), 1);
			data += 1;
		}
	} else {
		chain_ids = PyList_New(residues->size());
		unsigned long i = 0;
		for (std::vector<Residue *>::const_iterator ri = residues->begin();
				ri != residues->end(); ++ri, ++i) {
			PyList_SET_ITEM(chain_ids, i, PyUnicode_FromString((*ri)->chain_id().c_str()));
		}
	}
	return chain_ids;
}

static PyObject *
residue_names(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *res_vec_capsule;
	int numpy = 0; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "residues_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&res_vec_capsule, &numpy))
		return NULL;
	std::vector<Residue *> *residues;
	try {
		residues = decapsulate_res_vec(res_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "residues_capsule arg is not a residue-vec capsule");
		return NULL;
	}
	PyObject *res_names;
	if (numpy) {
		initialize_numpy();
		char *data;
		res_names = python_string_array(residues->size(), 3, &data);
		for (std::vector<Residue *>::const_iterator ri = residues->begin();
				ri != residues->end(); ++ri) {
			const std::string &rn = (*ri)->name();
			strncpy(data, rn.c_str(), 3);
			data += 3;
		}
	} else {
		res_names = PyList_New(residues->size());
		unsigned long i = 0;
		for (std::vector<Residue *>::const_iterator ri = residues->begin();
				ri != residues->end(); ++ri, ++i) {
			PyList_SET_ITEM(res_names, i, PyUnicode_FromString((*ri)->name().c_str()));
		}
	}
	return res_names;
}

static PyObject *
residue_numbers(PyObject *, PyObject *args, PyObject *kw)
{
	PyObject *res_vec_capsule;
	int numpy = 1; // Python arg parsing doesn't support C++ bool
	static const char *kw_list[] = { "residues_capsule", "numpy", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O|$p", (char **) kw_list,
			&res_vec_capsule, &numpy))
		return NULL;
	std::vector<Residue *> *residues;
	try {
		residues = decapsulate_res_vec(res_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "residues_capsule arg is not a res-vec capsule");
		return NULL;
	}
	PyObject *residue_numbers;
	if (numpy) {
		initialize_numpy();
		static_assert(sizeof(unsigned int) >= 4, "need 32-bit ints");
		unsigned int shape[1] = {(unsigned int)residues->size()};
		residue_numbers = allocate_python_array(1, shape, NPY_INT);
		int *data = (int *) PyArray_DATA((PyArrayObject *)residue_numbers);
		for (std::vector<Residue *>::const_iterator ri = residues->begin();
				ri != residues->end(); ++ri) {
			*data++ = (*ri)->position();
		}
	} else {
		residue_numbers = PyList_New(residues->size());
		unsigned int i = 0;
		for (std::vector<Residue *>::const_iterator ri = residues->begin();
				ri != residues->end(); ++ri, ++i) {
			PyList_SET_ITEM(residue_numbers, i, PyLong_FromLong((long)(*ri)->position()));
		}
	}
		
	return residue_numbers;
}

static PyObject *
residues(PyObject *, PyObject *mol_vec_capsule)
{
	std::vector<Molecule *> *mols;
	try {
		mols = decapsulate_mol_vec(mol_vec_capsule);
	} catch (std::invalid_argument &e) {
		PyErr_SetString(PyExc_ValueError, "first arg is not a mol-vec capsule");
		return NULL;
	}
	int size = 0;
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		size += m->residues().size();
	}
	std::vector<Residue *> *res_vec = new std::vector<Residue *>(0);
	res_vec->reserve(size);
	for (std::vector<Molecule *>::iterator mi = mols->begin(); mi != mols->end(); ++mi) {
		Molecule *m = *mi;
		const Molecule::Residues &m_residues = m->residues();
		res_vec->insert(res_vec->end(), m_residues.begin(), m_residues.end());
	}
	return encapsulate_res_vec(res_vec);
}

static struct PyMethodDef access_functions[] =
{
	{ "atom_names", (PyCFunction)atom_names, METH_VARARGS|METH_KEYWORDS, "" },
	{ "atom_residues", (PyCFunction)atom_residues, METH_O, "" },
	{ "atoms", (PyCFunction)atoms, METH_O, "" },
	{ "atoms_bonds", (PyCFunction)atoms_bonds, METH_O, "" },
	{ "coords", (PyCFunction)coords, METH_VARARGS|METH_KEYWORDS, "" },
	{ "element_names", (PyCFunction)element_names, METH_VARARGS|METH_KEYWORDS, "" },
	{ "element_numbers", (PyCFunction)element_numbers, METH_VARARGS|METH_KEYWORDS, "" },
	{ "molecules", (PyCFunction)molecules, METH_O, "" },
	{ "residue_chain_ids", (PyCFunction)residue_chain_ids, METH_VARARGS|METH_KEYWORDS, "" },
	{ "residue_names", (PyCFunction)residue_names, METH_VARARGS|METH_KEYWORDS, "" },
	{ "residue_numbers", (PyCFunction)residue_numbers, METH_VARARGS|METH_KEYWORDS, "" },
	{ "residues", (PyCFunction)residues, METH_O, "" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef access_def =
{
	PyModuleDef_HEAD_INIT,
	"access",
	"Access functions for molecular aggregates",
	-1,
	access_functions,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_access()
{
	return PyModule_Create(&access_def);
}
