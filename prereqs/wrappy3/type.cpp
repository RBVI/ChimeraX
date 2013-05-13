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

// $Id: type.cpp 29912 2010-01-28 01:31:29Z gregc $

// dump output for Python types

#include <fstream>
#include <sstream>
#include "common.h"
#include "method.h"
#include "type.h"
#include "CvtType.h"
#include <stdio.h>
#include <stdlib.h>

using std::string;

static string
commonName(const ClassInfo *ci, const char *replacement, bool changeAll)
{
	// 4 cases:
	//	(1) wrapped and in the module namespace
	//	(2) wrapped and in another namespace (promoted to module)
	//	(3) not wrapped in the module namespace (ignored)
	//	(4) not wrapped in another namespace
	string name(qualify(ci->cd->scope, ci->name.str(), true));
	bool sameNamespace = false;
	if (!nameSpace.empty() && name.size() > nameSpace.size()
	&& name.compare(0, nameSpace.size(), nameSpace) == 0
	&& name.compare(nameSpace.size(), 2, "::") == 0) {
		name = name.substr(nameSpace.size() + 2);
		sameNamespace = true;
	}

	if (ci->skipClass) {
		// case 4, leave alone
	} else if (!sameNamespace) {
		// case 2
		string::size_type i = name.rfind("::");
		if (i != string::npos)
			name.replace(0, i + 2, "");
	}

	// replace all ::'s with replacement string
	string::size_type start = 0;
	if (!changeAll && ci->skipClass)
		start = name.find("::") + 2;
	for (string::size_type i = name.find("::", start); i != string::npos;
							i = name.find("::", i))
		name.replace(i, 2, replacement);
	return name;
}

string
moduleName(const ClassInfo *ci, const string &suffix, bool external/*=false*/)
{
	// C++ name within module (may refer to other module)
	string prefix(commonName(ci, "_", false));
	if (!ci->skipClass && external && !nameSpace.empty())
		return nameSpace + "::" + prefix + '_' + suffix;
	return prefix + '_' + suffix;
}

string
pythonName(const ClassInfo *ci, const string &suffix)
{
	// Python name within module (may refer to other module)
	string prefix(commonName(ci, ".", true));
	if (suffix.empty())
		return prefix;
	return prefix + '.' + suffix;
}

string
moduleScopeName(const ClassInfo *ci, const string &name, bool external/*=false*/)
{
	// like moduleName, but for a module reference for the module
	// containing the class definition
	// C++ name within module (may refer to other module)
	string prefix(commonName(ci, "_", false));
	if (!ci->skipClass && external && !nameSpace.empty())
		prefix = nameSpace + "::" + prefix;
	string::size_type i = prefix.rfind("::");
	if (i != string::npos)
		i += 2;
	else
		i = 0;
	prefix.replace(i, string::npos, "");
	return prefix.substr(0, i) + name;
}

static void
dumpWrapTypeInline(std::ostream &output, const ClassInfo *ci)
{
	string prefix(qualify(ci->cd->scope, ci->name.str(), true));
	string objectType(moduleName(ci, "objectType", true));
	output <<
		"\n"
		"namespace wrappy {\n"
		// type check function
		"\n"
		"template <> inline bool\n"
		"Type<" << prefix <<
					">::check(PyObject* _o, bool noneOk)\n"
		"{\n"
		"\tif (noneOk && _o == Py_None)\n"
		"\t\treturn true;\n"
		"\treturn PyObject_TypeCheck(_o, &" << objectType << ");\n"
		"}\n";

	output <<
		"\n"
		"template <> " << exportTag << "PyObject* pyObject(" << prefix
								<< "* _o);\n";
	if (ci->isEmbedded)
		output <<
			"template <> inline PyObject* pyObject(" << prefix
					<< " _o) { return pyObject(&_o); }\n";
	output <<
		"template <> inline PyObject* pyObject(" << prefix
			<< " const* _o) { return pyObject(const_cast<"
						<< prefix << "*>(_o)); }\n";
	output <<
		"\n"
		"} // namespace wrappy\n";
}

void
dumpWrapTypeOutline(std::ostream &output, const ClassInfo *ci)
{
	string prefix(qualify(ci->cd->scope, ci->name.str(), true));
	string objectName(moduleName(ci, "object", true));
	if (!ci->isWrappySubclass && !ci->isEmbedded)
		output <<
			"\n"
			"#include <memory>\n";	// for auto_ptr
	output <<
		"\n"
		"namespace wrappy {\n";
	string debugPrefix;
	if (!nameSpace.empty())
		debugPrefix = nameSpace + "::";

	output <<
		"\n"
		"template <> " << exportTag << "PyObject*\n"
		"pyObject(" << prefix << "* _o)\n"
		"{\n";
	if (ci->isWrappySubclass) {
		output <<
			"\tif (_o == NULL)\n"
			"\t\tPy_RETURN_NONE;\n"
			"\treturn _o->wpyGetObject();\n";
	} else {
		output <<
			"\t// like _new function, but use copy constructor\n";
		if (!ci->isEmbedded)
			output <<
				"\tstd::auto_ptr<" << prefix << "> tmp(new "
						<< prefix << "(*_o));\n";
		output <<
			"\t" << objectName << "* self = static_cast<"
				<< objectName << "*>(" << objectName
				<< "Type.tp_alloc(&" << objectName
				<< "Type, 0));\n"
			"\tif (" << debugPrefix << module << "Debug >= 6)\n"
			"\t\tstd::cerr << \"Allocate " << ci->name
						<< ": \" << self << '\\n';\n"
			"\tif (self == NULL)\n"
			"\t\tthrow wrappy::PythonError();\n";
		if (!ci->isFinal)
			output <<
				"\tself->_inst_dict = PyDict_New();\n"
				"\tif (self->_inst_dict == NULL) {\n"
				"\t\tPy_DECREF(self);\n"
				"\t\tthrow wrappy::PythonError();\n"
				"\t}\n";
		if (!ci->isEmbedded)
			output <<
				"\tself->_inst_data = tmp.release();\n";
		else {
			output <<
				"\ttry {\n"
				"\t\tnew (&self->_inst_data) " << prefix
								<< "(*_o);\n"
				"\t\tself->_initialized = true;\n"
				"\t} catch (...) {\n"
				"\t\tPy_DECREF(self);\n"
				"\t\tthrow;\n"
				"\t}\n";
		}
		output <<
			"\treturn self;\n";
	}
	output << "}\n";

	output <<
		"\n"
		"} // namespace wrappy\n";
}

bool
dumpTypeHeader(const ClassInfo *ci)
{
	string filename(moduleName(ci, "Object.h"));
	std::auto_ptr<std::ostream> outs(outputStream(filename).release());
	std::ostream &output = *outs.get();
	string objectName(moduleName(ci, "object"));
	string cppName(qualify(ci->cd->scope, ci->name.str()));

	string define(moduleName(ci, "object_h"));
	if (!nameSpace.empty())
		define = nameSpace + '_' + define;
	output <<
		"#ifndef " << define << "\n"
		"# define " << define << "\n"
		"# if defined(_MSC_VER) && (_MSC_VER >= 1020)\n"
		"#  pragma once\n"
		"# endif\n"
		"\n";
	output <<
		"# define PY_SSIZE_T_CLEAN 1\n"
		"#include <Python.h>\n" <<
		ci->includeFile << "\n" <<
		"#include <WrapPy3.h>";
	if (!nameSpace.empty())
		output <<
			"\n"
			"namespace " << nameSpace << " {\n";

	output <<
		"\n"
		<< exportTag << "extern PyTypeObject " << moduleName(ci, "objectType") << ";\n";

	if (ci->isBaseClass) {
		bool first = true;
		for (AttrVec::const_iterator i = ci->attrs.begin();
						i != ci->attrs.end(); ++i) {
			const ClassAttrInfo &ai = *i;
			if (!ai.cache)
				continue;	// skip primitive types
			if (first) {
				first = false;
				output <<
					"\n"
					"// help subclasses initialize cached attributes\n";
			}
			string init = moduleName(ci, "attrInit" + ai.name.str());
			output << exportTag << "extern PyObject* " << init
						<< "(PyObject*, void*);\n";
		}
	}

	// object definition
	output <<
		"\n"
		"struct " << objectName << ": public PyObject {\n";
	if (!ci->isFinal)
		output << "\tPyObject* _inst_dict;\n";
	if (ci->isWrappySubclass)
		output <<
			"\twrappy::Obj* _inst_data;\n"
			"\t" << cppName << "* _inst() { return static_cast<"
				<< cppName << "*>(_inst_data); }\n"
			"\tPyObject* _weaklist;\n";
	else if (!ci->isEmbedded)
		output <<
			"\t" << cppName << "* _inst_data;\n"
			"\t" << cppName << "* _inst() { return _inst_data; }\n";
	else {
		output <<
			"\t" << alignmentType << " _inst_data[(sizeof ("
				<< cppName << ") + sizeof (" << alignmentType
				<< ") - 1) / sizeof (" << alignmentType
				<< ")];\n"
			"\tbool _initialized;\n"
			"\t" << cppName
				<< "* _inst() { return reinterpret_cast<"
				<< cppName << "*>(_inst_data); }\n";
	}
	output <<
		"};\n"
		"\n"
		<< exportTag << "extern " << cppName << "* getInst("
						<< objectName << "* self);\n";

	if (!nameSpace.empty())
		output <<
			"\n"
			"} // namespace " << nameSpace << "\n";

	dumpWrapTypeInline(output, ci);

	output <<
		"\n"
		"#endif\n";
	// flush output and check return status
	output.flush();
	if (output.good())
		return true;
	std::cerr << programName << ": error writing " << filename << '\n';
	return false;
}

static void
dumpConstructor(std::ostream &output, const ClassInfo *ci, const string &fname,
		const string &docName)
{
	MethodInfo mi;
	if (!ci->isAbstractType && !ci->constructors.empty())
		dumpMethod(output, ci, ci->constructors, &mi);
	else {
		output <<
			"\n"
			"static int\n" <<
			fname << "(PyObject* _type, PyObject*, PyObject*)\n"
			"{\n"
			"\tPyErr_SetString(PyExc_TypeError, \"cannot create '"
					<< ci->name << "' instances\");\n"
			"\treturn -1;\n"
			"}\n";
		mi.doc = "Not instantiable from Python";
	}

	output << "\n"
		"static const char " << docName << "[] = \""
						<< stringize(mi.doc) << "\";\n";
}

static void
dumpInaccessibleConstructor(std::ostream &output, const ClassInfo *ci,
			const string &fname, const string &docName)
{
	output <<
		"\n"
		"static PyObject*\n" <<
		fname << "(PyTypeObject*, PyObject*, PyObject*)\n"
		"{\n"
		"\tPyErr_SetString(PyExc_TypeError, \"cannot create '"
				<< ci->name << "' instances\");\n"
		"\treturn NULL;\n"
		"}\n"

		"\n"
		"static const char " << docName
				<< "[] = \"Not instantiable from Python\";\n";
}

static void
dumpDeallocator(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	string objectName(moduleName(ci, "object"));
	output <<
		"\n"
		"static void\n" <<
		fname << "(PyObject* obj)\n"
		"{\n"
		"\t" << objectName << "* self = static_cast<" << objectName
								<< "*>(obj);\n"
		"\tif (" << module << "Debug >= 6)\n"
		"\t\tstd::cerr << \"Deallocate " << ci->name
						<< ": \" << self << '\\n';\n"
		"\tPyObject* err_type;\n"
		"\tPyObject* err_value;\n"
		"\tPyObject* err_traceback;\n"
		"\tbool has_error = PyErr_Occurred() != NULL;\n"
		"\tif (has_error)\n"
		"\t\tPyErr_Fetch(&err_type, &err_value, &err_traceback);\n";
	if (ci->isEmbedded) {
		if (!ci->isFinal)
			output <<
				"\tPy_CLEAR(self->_inst_dict);\n";
		output <<
			"\tif (self->_initialized)\n"
			"\t\ttry {\n"
			"\t\t\tself->_initialized = false;\n"
			"\t\t\tself->_inst()->~" << ci->name
					<< "(); // freed in Python's C layer\n"
			"\t\t} catch (...) {\n"
			"\t\t\t// ignore exceptions in destructors\n"
			"\t\t}\n";
	} else {
		output <<
			"\tif (self->_inst_data) {\n";
		if (ci->cd->members.has(Symbol('~' + ci->name.str()), PUBLIC)) {
			string cppName(qualify(ci->cd->scope, ci->name.str()));
			output <<
				"\t\t" << cppName << "* _save = self->_inst();\n"
				"\t\tself->_inst_data = 0;\n";
			if (!ci->isWrappySubclass)
				output << "\t\tdelete _save;\n";
			else
				output <<
					"\t\tif (_save->pyOwned())\n"
					"\t\t\tdelete _save;\n"
					"\t\telse\n"
					"\t\t\t_save->wpyDisassociate();\n";
		} else {
			output <<
				"\t\tself->_inst_data = 0;\n";
		}
		output <<
			"\t}\n";
		output <<
			"\tif (self->ob_refcnt > 0) {\n"
			"\t\tif (" << module << "Debug >= 6)\n"
			"\t\t\tstd::cerr << \"Resurrected " << ci->name
						<< ": \" << self << '\\n';\n"
			"\t\tif (has_error)\n"
			"\t\t\tPyErr_Restore(err_type, err_value, err_traceback);\n"
			"\t\treturn; // somehow the object was resurrected\n"
			"\t}\n";
		if (!ci->isFinal)
			output <<
				"\tPy_CLEAR(self->_inst_dict);\n";
		if (ci->isWrappySubclass)
			output <<
				"\tif (self->_weaklist)\n"
				"\t\tPyObject_ClearWeakRefs(obj);\n";
	}
	output <<
		"\tif (has_error)\n"
		"\t\tPyErr_Restore(err_type, err_value, err_traceback);\n"
		"\tPy_TYPE(obj)->tp_free(obj);\n"
		"}\n";
}

static void
dumpPrint(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	// TODO: if str() method, then use it
	string objectName(moduleName(ci, "object"));
	output <<
		"\n"
		"static int\n" <<
		fname << "(PyObject* obj, FILE* fp, int /*flags*/)\n"
		"{\n"
		"\t" << objectName << "* self = static_cast<" << objectName
								<< "*>(obj);\n";
	if (!ci->isEmbedded)
		output <<
			"\tif (self->_inst_data == NULL)\n"
			"\t\treturn 0;\n";
	output <<
		"\ttry {\n";
	if (ci->print)
		output <<
			"\t\tstd::ostringstream buf;\n"
			"\t\tbuf << *getInst(self);\n"
			"\t\tfputs(buf.str().c_str(), fp);\n";
	else if (!ci->str.empty()
	&& ci->str.find(Symbol("str")) != ci->str.end())
		output <<
			"\t\tfputs(getInst(self)->str().c_str(), fp);\n";
	else if (!ci->str.empty()
	&& ci->str.find(Symbol("c_str")) != ci->str.end())
		output <<
			"\t\tfputs(getInst(self)->c_str(), fp);\n";
	output <<
		"\t\treturn 0;\n"
		"\t} catch (...) {\n"
		"\t\t" << module << "Error();\n"
		"\t}\n"
		"\treturn -1;\n"
		"}\n";
}

static void
dumpStr(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	string objectName(moduleName(ci, "object"));
	output <<
		"\n"
		"static PyObject*\n" <<
		fname << "(PyObject* obj)\n"
		"{\n"
		"\t" << objectName << "* self = static_cast<" << objectName
								<< "*>(obj);\n";
	if (!ci->isEmbedded)
		output <<
			"\tif (self->_inst_data == NULL)\n"
			"\t\tPy_RETURN_NONE;\n";
	output <<
		"\ttry {\n";
	if (!ci->str.empty()
	&& ci->str.find(Symbol("str")) != ci->str.end())
		output <<
			"\t\treturn wrappy::pyObject(getInst(self)->str());\n";
	else if (!ci->str.empty()
	&& ci->str.find(Symbol("c_str")) != ci->str.end())
		output <<
			"\t\treturn wrappy::pyObject(getInst(self)->c_str());\n";
	else if (ci->print)
		output <<
			"\t\tstd::ostringstream buf;\n"
			"\t\tbuf << " <<  "*getInst(self);\n"
			"\t\treturn wrappy::pyObject(buf.str());\n";
	output <<
		"\t} catch (...) {\n"
		"\t\t" << module << "Error();\n"
		"\t}\n"
		"\treturn NULL;\n"
		"}\n";
}

static void
dumpCompare(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	string objectName(moduleName(ci, "object"));
	string cppName(qualify(ci->cd->scope, ci->name.str()));
	output <<
		"\n"
		"static int\n" <<
		fname << "(PyObject* _obj0, PyObject* _obj1)\n"
		"{\n"
		"\tif (Py_TYPE(_obj0) != Py_TYPE(_obj1))\n"
		"\t\treturn -2;\n"
		"\ttry {\n"
		"\t\t" << cppName << "* _o0 = getInst(static_cast<"
					<< objectName << "*>(_obj0));\n"
		"\t\t" << cppName << "* _o1 = getInst(static_cast<"
					<< objectName << "*>(_obj1));\n"
		"\t\tif (*_o0 < *_o1)\n"
		"\t\t\treturn -1;\n"
		"\t\telse if (*_o1 < *_o0)\n"
		"\t\t\treturn 1;\n"
		"\t\treturn 0;\n"
		"\t} catch (...) {\n"
		"\t\treturn -2;\n"
		"\t}\n"
		"}\n";
}

static void
dumpRichCompare(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	string objectName(moduleName(ci, "object"));
	string cppName(qualify(ci->cd->scope, ci->name.str()));
	output <<
		"\n"
		"static PyObject*\n" <<
		fname << "(PyObject* _obj0, PyObject* _obj1, int _op)\n"
		"{\n"
		"\tif (Py_Type(_obj0) != Py_Type(_obj1)) {\n"
		"\t\tPy_INCREF(Py_NotImplemented);\n"
		"\t\treturn Py_NotImplemented;\n"
		"\t}\n"
		"\tbool _res = false;\n"
		"\ttry {\n"
		"\t\t" << cppName << "* _o0 = getInst(static_cast<"
					<< objectName << "*>(_obj0));\n"
		"\t\t" << cppName << "* _o1 = getInst(static_cast<"
					<< objectName << "*>(_obj1));\n"
		"\t\tswitch (_op) {\n";

	output <<"\t\t  case Py_LT:\n";
	if (!ci->lt.empty())
		output << "\t\t\t_res = *_o0 < *_o1;\n"
			"\t\t\tbreak;\n";
	else
		output << "\t\t\tthrow std::invalid_argument(\"no < relation defined\");\n";

	output << "\t\t  case Py_LE:\n";
	if (!ci->le.empty())
		output << "\t\t\t_res = *_o0 <= *_o1;\n"
			"\t\t\tbreak;\n";
	else if (!ci->lt.empty())
		output << "\t\t\t_res = !(*_o1 < *_o0);\n"
			"\t\t\tbreak;\n";
	else
		output << "\t\t\tthrow std::invalid_argument(\"no <= relation defined\");\n";

	output << "\t\t  case Py_EQ:\n";
	if (!ci->eq.empty())
		output << "\t\t\t_res = *_o0 == *_o1;\n"
			"\t\t\tbreak;\n";
	else if (!ci->lt.empty())
		output << "\t\t\t_res = !((*_o0 < *_o1) || (*_o1 < *_o0));\n"
			"\t\t\tbreak;\n";
	else
		output << "\t\t\tthrow std::invalid_argument(\"no == relation defined\");\n";

	output << "\t\t  case Py_NE:\n";
	if (!ci->ne.empty())
		output << "\t\t\t_res = *_o0 != *_o1;\n"
			"\t\t\tbreak;\n";
	else if (!ci->eq.empty())
		output << "\t\t\t_res = !(*_o0 == *_o1);\n"
			"\t\t\tbreak;\n";
	else if (!ci->lt.empty())
		output << "\t\t\t_res = (*_o0 < *_o1) || (*_o1 < *_o0);\n"
			"\t\t\tbreak;\n";
	else
		output << "\t\t\tthrow std::invalid_argument(\"no != relation defined\");\n";

	output << "\t\t  case Py_GT:\n";
	if (!ci->gt.empty())
		output << "\t\t\t_res = *_o0 > *_o1;\n"
			"\t\t\tbreak;\n";
	else if (!ci->lt.empty())
		output << "\t\t\t_res = *_o1 < *_o0;\n"
			"\t\t\tbreak;\n";
	else
		output << "\t\t\tthrow std::invalid_argument(\"no > relation defined\");\n";

	output << "\t\t  case Py_GE:\n";
	if (!ci->ge.empty())
		output << "\t\t\t_res = *_o0 >= *_o1;\n"
			"\t\t\tbreak;\n";
	else if (!ci->lt.empty())
		output << "\t\t\t_res = !(*_o0 < *_o1);\n"
			"\t\t\tbreak;\n";
	else
		output << "\t\t\tthrow std::invalid_argument(\"no >= relation defined\");\n";

	output <<
		"\t\t}\n"
		"\t\tif (_res)\n"
		"\t\t\tPy_RETURN_TRUE;\n"
		"\t\telse\n"
		"\t\t\tPy_RETURN_FALSE;\n"
		"\t} catch (...) {\n"
		"\t\t" << module << "Error();\n"
		"\t}\n"
		"\treturn NULL;\n"
		"}\n";
}

static void
dumpCall(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	MethodInfo mi;
	dumpMethod(output, ci, ci->call, &mi, fname, "__call__", TernaryFunc);
}

static void
dumpHash(std::ostream &output, const ClassInfo *ci, const string &fname)
{
	string objectName(moduleName(ci, "object"));
	output <<
		"\n"
		"static long\n" <<
		fname << "(PyObject* _o)\n"
		"{\n"
		"\t" << objectName << "* self = static_cast<" << objectName
							<< "*>(_o);\n"
		"\ttry {\n"
		"\t\treturn getInst(self)->hash();\n"
		"\t} catch (...) {\n"
		"\t\t" << module << "Error();\n"
		"\t\treturn -1;\n"
		"\t}\n"
		"}\n";
}

static void
dumpPtrHash(std::ostream &output, const ClassInfo * /*ci*/, const string &fname)
{
	// assert(!ci->isEmbedded);
	output <<
		"\n"
		"static long\n" <<
		fname << "(PyObject* _o)\n"
		"{\n"
		"\treturn _Py_HashPointer(_o);\n"
		"}\n";
}

static void
dumpAttributes(std::ostream &output, const ClassInfo *ci,
		/*OUT*/ string *tp_members, /*OUT*/ string *tp_getset)
{
	std::ostringstream os_members, os_getset;
	string objectName(moduleName(ci, "object"));

	// These are functions for public class variables.
	if (!ci->isFinal)
		os_members << "\t{ PY_STUPID \"__dict__\", T_OBJECT, offsetof("
			<< objectName << ", _inst_dict), READONLY, NULL },\n";
	bool warnFinalCache = true;
	for (AttrVec::const_iterator i = ci->attrs.begin();
						i != ci->attrs.end(); ++i) {
		const ClassAttrInfo &ai = *i;
		if (!ai.offset.empty()) {
			if (ai.get.empty()) {
				std::cerr << "warning: unable to make attribute " << ai.name << " WRITEONLY\n";
			}
			os_members << "\t{ PY_STUPID \"" << ai.name << "\", " 
				<< ai.type << ",\n\t\t" << ai.offset
				<< ",\n\t\t";
			if (ai.set.empty())
				os_members << "READONLY";
			else
				os_members << "0";
			os_members <<
				", PY_STUPID \"" << ai.doc << "\" },\n";
			continue;
		}
		string getter("NULL"), setter("NULL");
		if (!ai.get.empty()) {
			getter = moduleName(ci, "attrGet" + ai.name.str());
			output <<
				"\n"
				"static PyObject*\n" <<
				getter << "(PyObject* self, void*)\n"
				"{\n"
				"\tPyObject* _o = " << ai.get << ";\n";
			if (ai.cache && !ci->isFinal) {
				string objectName(moduleName(ci, "object"));
				output <<
					"\t// cache attribute\n"
					"\tif (_o == NULL)\n"
					"\t\treturn NULL;\n"
					"\tstatic PyObject* key = NULL;\n"
					"\tif (key == NULL)\n"
					"\t\tkey = PyUnicode_InternFromString(\"__cached_"
							<< ai.name << "__\");\n"
					"\tPyDict_SetItem(static_cast<"
						<< objectName
						<< "*>(self)->_inst_dict, key, _o);\n";
			} else if (ai.cache && ci->isFinal && warnFinalCache) {
				warnFinalCache = false;
				std::cerr << "warning: cached attribute " << ai.name << " in FINAL class\n";
			}
			output <<
				"\treturn _o;\n"
				"}\n";
		}
		if (!ai.set.empty()) {
			setter = moduleName(ci, "attrSet" + ai.name.str());
			output <<
				"\n"
				"static int\n" <<
				setter << "(PyObject* self, PyObject* value, void*)\n"
				"{\n"
				"\tif (value == NULL) {\n"
				"\t\tPyErr_SetString(PyExc_TypeError, \"Cannot delete "
						<< ai.name << " attribute\");\n"
				"\t\treturn -1;\n"
				"\t}\n";
			output << ai.set << ";\n";
			if (ai.cache && !ci->isFinal) {
				string objectName(moduleName(ci, "object"));
				output <<
					"\t// cache attribute\n"
					"\tstatic PyObject* key = NULL;\n"
					"\tif (key == NULL)\n"
					"\t\tkey = PyUnicode_InternFromString(\"__cached_"
							<< ai.name << "__\");\n"
					"\tPyDict_SetItem(static_cast<"
						<< objectName
						<< "*>(self)->_inst_dict, key, value);\n";
			} else if (ai.cache && ci->isFinal && warnFinalCache) {
				warnFinalCache = false;
				std::cerr << "warning: cached attribute " << ai.name << " in FINAL class\n";
			}
			output <<
				"\treturn 0;\n"
				"}\n";
		}
		os_getset << "\t{ PY_STUPID \"" << ai.name << "\", " << getter
			<< ", " << setter << ", PY_STUPID \"" << ai.doc
			<< "\", NULL },\n";
	}

	string members = os_members.str();
	if (!members.empty()) {
		*tp_members = moduleName(ci, "members");
		output << 
			"\n"
			"static PyMemberDef " << *tp_members << "[] = {\n" <<
			members <<
			"\t{ NULL, 0, 0, 0, NULL }\n"
			"};\n";
	}
	string getset = os_getset.str();
	if (!getset.empty()) {
		*tp_getset = moduleName(ci, "getset");
		output << 
			"\n"
			"static PyGetSetDef " << *tp_getset << "[] = {\n" <<
			getset <<
			"\t{ NULL, NULL, NULL, NULL, NULL }\n"
			"};\n";
	}
}

static void
dumpMethods(std::ostream &output, const ClassInfo *ci, string *tp_methods)
{
	std::ostringstream method_table;

	// first, create all of the method definitions
	MethodMap::const_iterator next;
	for (MethodMap::const_iterator i = ci->attrMethods.begin();
					i != ci->attrMethods.end(); i = next) {
		next = ci->attrMethods.upper_bound(i->first);
		MethodInfo mi;
		dumpMethod(output, ci, i, next, &mi, string(), string(),
								Attribute);
	}

	for (MethodMap::const_iterator i = ci->methods.begin();
					i != ci->methods.end(); i = next) {
		next = ci->methods.upper_bound(i->first);
		MethodInfo mi;
		dumpMethod(output, ci, i, next, &mi);
		string proto = mi.doc;
		output <<
			"\n"
			"static const char " << moduleName(ci, i->first.str())
				<< "_doc[] = \n\"" << stringize(proto)
				<< "\";\n";
		method_table <<
			"\t{ \"" << i->first << "\", (PyCFunction) "
				<< moduleName(ci, i->first.str()) << ",\n"
			"\t\t" << mi.ml_flags << ", "
				<< moduleName(ci, i->first.str())
				<< "_doc },\n";
	}

	for (MethodMap::const_iterator i = ci->staticMethods.begin();
				i != ci->staticMethods.end(); i = next) {
		next = ci->staticMethods.upper_bound(i->first);
		// except for this check, this is the same as the methods
		// code immediately above
		if (ci->methods.find(i->first) != ci->methods.end()) {
			std::cerr << "skipping static method, " << i->first
				<< ", with same name as normal method in class "
				<< ci->name << '\n';
			continue;
		}
		MethodInfo mi;
		dumpMethod(output, ci, i, next, &mi);
		string proto = mi.doc;
		output <<
			"\n"
			"static const char " << moduleName(ci, i->first.str())
				<< "_doc[] = \n\"" << stringize(proto)
				<< "\";\n";
		method_table <<
			"\t{ \"" << i->first << "\", (PyCFunction) "
				<< moduleName(ci, i->first.str()) << ",\n"
			"\t\t" << mi.ml_flags << ", "
				<< moduleName(ci, i->first.str())
				<< "_doc },\n";
	}

	if (ci->cd->hasPublicCopyConstructor) {
		string fname(moduleName(ci, "copy"));
		output <<
			"\n"
			"static PyObject*\n" <<
			fname << "( PyObject* _obj, PyObject*)\n"
			"{\n"
			"\tPyObject* _tmp = PyTuple_New(1);\n"
			"\tif (_tmp == NULL)\n"
			"\t\treturn NULL;\n"
			"\tPy_INCREF(_obj);\n"
			"\tPyTuple_SetItem(_tmp, 0, _obj);\n"
			"\tPyObject* _result = Py_Type(_obj)->tp_new(Py_Type(_obj), _tmp, NULL);\n"
			"\tif (_result != NULL\n"
			"\t&& Py_Type(_obj)->tp_init(_result, _tmp, NULL) < 0) {\n"
			"\t\tPy_DECREF(_result);\n"
			"\t\t_result = NULL;\n"
			"\t}\n"
			"\tPy_DECREF(_tmp);\n"
			"\treturn _result;\n"
			"}\n"
			"\n"
			"static const char " << fname <<
				"_doc[] = \"__copy__() -> " << ci->name <<
				"\";\n";
		method_table <<
			"\t{ \"__copy__\", (PyCFunction) " << fname <<
									",\n"
			"\t\tMETH_NOARGS, " << fname << "_doc },\n";
	}

	if (!ci->call.empty()) {
		// provide call method as well (with METH_COEXIST), so we can
		// provide a doc string
		string fname(moduleName(ci, "callMethod"));
		MethodInfo mi;
		dumpMethod(output, ci, ci->call, &mi, fname, "__call__");
		string proto = mi.doc;
		output <<
			"\n"
			"static const char " << fname << "_doc[] = \n\""
					<< stringize(proto) << "\";\n";
		method_table <<
			"\t{ \"__call__\", (PyCFunction) " << fname
									<< ",\n"
			"\t\t" << mi.ml_flags << " | METH_COEXIST, "
							<< fname << "_doc },\n";
	}

	const string& methods = method_table.str();
	if (methods.empty())
		return;
	*tp_methods = moduleName(ci, "methods");
	output << "\n"
		"static PyMethodDef " << *tp_methods << "[] = {\n" <<
		methods <<
		"\t{ NULL, NULL, 0, NULL }\n"
		"};\n";
}

static void
dumpUnaryFunc(std::ostream &output, const ClassInfo *, const MethodMap &mm,
					string &fname, const string &pyname)
{
	if (!dumpOperator(output, mm, fname, pyname, UnaryFunc))
		fname = "0";
}

static void
dumpBinaryFunc(std::ostream &output, const ClassInfo *, const MethodMap &mm,
					string &fname, const string &pyname)
{
	if (!dumpOperator(output, mm, fname, pyname, BinaryFunc))
		fname = "0";
}

static void
dumpInquiryFunc(std::ostream &output, const ClassInfo *ci, const MethodMap &mm,
				const string &fname, const string &pyname)
{
	MethodInfo mi;
	dumpMethod(output, ci, mm, &mi, fname, pyname, Inquiry);
}

static void
dumpObjObjProc(std::ostream &output, const ClassInfo *, const MethodMap &mm,
					string &fname, const string &pyname)
{
	if (!dumpOperator(output, mm, fname, pyname, ObjObjProc))
		fname = "0";
}

static void
dumpSsizeArgFunc(std::ostream &output, const ClassInfo *, const MethodMap &mm,
					string &fname, const string &pyname)
{
	if (!dumpOperator(output, mm, fname, pyname, SsizeArgFunc))
		fname = "0";
}

static void
dumpSsizeObjArgProc(std::ostream &output, const ClassInfo *,
		const MethodMap &mm, string &fname, const string &pyname)
{
	if (!dumpOperator(output, mm, fname, pyname, SsizeObjArgProc))
		fname = "0";
}

static void
dumpObjObjArgProc(std::ostream &output, const ClassInfo *,
		const MethodMap &mm, string &fname, const string &pyname)
{
	if (!dumpOperator(output, mm, fname, pyname, ObjObjArgProc))
		fname = "0";
}

static void
dumpNumberMethods(std::ostream &output, const ClassInfo *ci, const string &name)
{
	string nb_add("0");
	string nb_subtract("0");
	string nb_multiply("0");
	string nb_divide("0");
	string nb_remainder("0");
	string nb_divmod("0");
	string nb_power("0");
	string nb_negative("0");
	string nb_positive("0");
	string nb_absolute("0");
	string nb_nonzero("0");
	string nb_invert("0");
	string nb_lshift("0");
	string nb_rshift("0");
	string nb_and("0");
	string nb_xor("0");
	string nb_or("0");
	string nb_coerce("0");
	string nb_int("0");
	string nb_long("0");
	string nb_float("0");
	string nb_oct("0");
	string nb_hex("0");
	// Pytyhon 2.0
	string nb_inplace_add("0");
	string nb_inplace_subtract("0");
	string nb_inplace_multiply("0");
	string nb_inplace_divide("0");
	string nb_inplace_remainder("0");
	string nb_inplace_power("0");
	string nb_inplace_lshift("0");
	string nb_inplace_rshift("0");
	string nb_inplace_and("0");
	string nb_inplace_xor("0");
	string nb_inplace_or("0");
	// Python 2.2
	string nb_floor_divide("0");
	string nb_true_divide("0");
	string nb_inplace_floor_divide("0");
	string nb_inplace_true_divide("0");
	// Python 2.5
	string nb_index("0");

	if (!ci->nb_add.empty()) {
		nb_add = moduleName(ci, "add");
		dumpBinaryFunc(output, ci, ci->nb_add, nb_add, "__add__");
	}
	if (!ci->nb_subtract.empty()) {
		nb_subtract = moduleName(ci, "subtract");
		dumpBinaryFunc(output, ci, ci->nb_subtract, nb_subtract, "__sub__");
	}
	if (!ci->nb_multiply.empty()) {
		nb_multiply = moduleName(ci, "multiply");
		dumpBinaryFunc(output, ci, ci->nb_multiply, nb_multiply, "__mul__");
	}
	if (!ci->nb_divide.empty()) {
		nb_divide = moduleName(ci, "divide");
		dumpBinaryFunc(output, ci, ci->nb_divide, nb_divide, "__div__");
	}
	if (!ci->nb_remainder.empty()) {
		nb_remainder = moduleName(ci, "remainder");
		dumpBinaryFunc(output, ci, ci->nb_remainder, nb_remainder, "__mod__");
	}
	if (!ci->nb_divmod.empty()) {
		nb_divmod = moduleName(ci, "divmod");
		dumpBinaryFunc(output, ci, ci->nb_divmod, nb_divmod, "__divmod__");
	}
#if 0
	if (!ci->nb_power.empty()) {
		nb_power = moduleName(ci, "power");
		dumpTernaryFunc(output, ci, ci->nb_power, nb_power);
	}
#endif
	if (!ci->nb_negative.empty()) {
		nb_negative = moduleName(ci, "negative");
		dumpUnaryFunc(output, ci, ci->nb_negative, nb_negative, "__neg__");
	}
	if (!ci->nb_positive.empty()) {
		nb_positive = moduleName(ci, "positive");
		dumpUnaryFunc(output, ci, ci->nb_positive, nb_positive, "__pos__");
	}
	if (!ci->nb_absolute.empty()) {
		nb_absolute = moduleName(ci, "absolute");
		dumpUnaryFunc(output, ci, ci->nb_absolute, nb_absolute, "__abs__");
	}
	if (!ci->nb_nonzero.empty()) {
		nb_nonzero = moduleName(ci, "nonzero");
		dumpInquiryFunc(output, ci, ci->nb_nonzero, nb_nonzero, "__nonzero__");
	}
	if (!ci->nb_invert.empty()) {
		nb_invert = moduleName(ci, "invert");
		dumpUnaryFunc(output, ci, ci->nb_invert, nb_invert, "__invert__");
	}
	if (!ci->nb_lshift.empty()) {
		nb_lshift = moduleName(ci, "lshift");
		dumpBinaryFunc(output, ci, ci->nb_lshift, nb_lshift, "__lshift__");
	}
	if (!ci->nb_rshift.empty()) {
		nb_rshift = moduleName(ci, "rshift");
		dumpBinaryFunc(output, ci, ci->nb_rshift, nb_rshift, "__rshift__");
	}
	if (!ci->nb_and.empty()) {
		nb_and = moduleName(ci, "and");
		dumpBinaryFunc(output, ci, ci->nb_and, nb_and, "__and__");
	}
	if (!ci->nb_xor.empty()) {
		nb_xor = moduleName(ci, "xor");
		dumpBinaryFunc(output, ci, ci->nb_xor, nb_xor, "__xor__");
	}
	if (!ci->nb_or.empty()) {
		nb_or = moduleName(ci, "or");
		dumpBinaryFunc(output, ci, ci->nb_or, nb_or, "__or__");
	}
#if 0
	// Obsolete and not wanted anyway because we want the comparison
	// code to decide to coerce, not the coerce method.
	if (ci->nb_coerce) {
		nb_coerce = moduleName(ci, "coerce");
		dumpCoercionFunc(output, nb_coerce);
	}
#endif
	if (!ci->nb_int.empty()) {
		nb_int = moduleName(ci, "int");
		dumpUnaryFunc(output, ci, ci->nb_int, nb_int, "__int__");
	}
	if (!ci->nb_long.empty()) {
		nb_long = moduleName(ci, "long");
		dumpUnaryFunc(output, ci, ci->nb_long, nb_long, "__long__");
	}
	if (!ci->nb_float.empty()) {
		nb_float = moduleName(ci, "float");
		dumpUnaryFunc(output, ci, ci->nb_float, nb_float, "__float__");
	}
	if (!ci->nb_oct.empty()) {
		nb_oct = moduleName(ci, "oct");
		dumpUnaryFunc(output, ci, ci->nb_oct, nb_oct, "__oct__");
	}
	if (!ci->nb_hex.empty()) {
		nb_hex = moduleName(ci, "hex");
		dumpUnaryFunc(output, ci, ci->nb_hex, nb_hex, "__hex__");
	}
	if (!ci->nb_inplace_add.empty()) {
		nb_inplace_add = moduleName(ci, "inplaceAdd");
		dumpBinaryFunc(output, ci, ci->nb_inplace_add, nb_inplace_add, "__iadd__");
	}
	if (!ci->nb_inplace_subtract.empty()) {
		nb_inplace_subtract = moduleName(ci, "inplaceSubtract");
		dumpBinaryFunc(output, ci, ci->nb_inplace_subtract, nb_inplace_subtract, "__isub__");
	}
	if (!ci->nb_inplace_multiply.empty()) {
		nb_inplace_multiply = moduleName(ci, "inplaceMultiply");
		dumpBinaryFunc(output, ci, ci->nb_inplace_multiply, nb_inplace_multiply, "__imul__");
	}
	if (!ci->nb_inplace_divide.empty()) {
		nb_inplace_divide = moduleName(ci, "inplaceDivide");
		dumpBinaryFunc(output, ci, ci->nb_inplace_divide, nb_inplace_divide, "__idiv__");
	}
	if (!ci->nb_inplace_remainder.empty()) {
		nb_inplace_remainder = moduleName(ci, "inplaceRemainder");
		dumpBinaryFunc(output, ci, ci->nb_inplace_remainder, nb_inplace_remainder, "__imod__");
	}
#if 0
	if (!ci->nb_inplace_power.empty()) {
		nb_inplace_power = moduleName(ci, "inplacePower");
		dumpTernaryFunc(output, ci, ci->nb_inplace_power, nb_inplace_power);
	}
#endif
	if (!ci->nb_inplace_lshift.empty()) {
		nb_inplace_lshift = moduleName(ci, "inplaceLShift");
		dumpBinaryFunc(output, ci, ci->nb_inplace_lshift, nb_inplace_lshift, "__ilshift__");
	}
	if (!ci->nb_inplace_rshift.empty()) {
		nb_inplace_rshift = moduleName(ci, "inplaceRShift");
		dumpBinaryFunc(output, ci, ci->nb_inplace_rshift, nb_inplace_rshift, "__irshift__");
	}
	if (!ci->nb_inplace_and.empty()) {
		nb_inplace_and = moduleName(ci, "inplaceAnd");
		dumpBinaryFunc(output, ci, ci->nb_inplace_and, nb_inplace_and, "__iand__");
	}
	if (!ci->nb_inplace_xor.empty()) {
		nb_inplace_xor = moduleName(ci, "inplaceXOr");
		dumpBinaryFunc(output, ci, ci->nb_inplace_xor, nb_inplace_xor, "__ixor__");
	}
	if (!ci->nb_inplace_or.empty()) {
		nb_inplace_or = moduleName(ci, "inplaceOr");
		dumpBinaryFunc(output, ci, ci->nb_inplace_or, nb_inplace_or, "__or__");
	}
	if (!ci->nb_floor_divide.empty()) {
		nb_floor_divide = moduleName(ci, "floorDivide");
		dumpBinaryFunc(output, ci, ci->nb_floor_divide, nb_floor_divide, "__floordiv__");
	}
	if (!ci->nb_true_divide.empty()) {
		nb_true_divide = moduleName(ci, "trueDivide");
		dumpBinaryFunc(output, ci, ci->nb_true_divide, nb_true_divide, "__itruediv__");
	}
	if (!ci->nb_inplace_floor_divide.empty()) {
		nb_inplace_floor_divide = moduleName(ci, "inplaceFloorDivide");
		dumpBinaryFunc(output, ci, ci->nb_inplace_floor_divide, nb_inplace_floor_divide, "__ifloordiv__");
	}
	if (!ci->nb_inplace_true_divide.empty()) {
		nb_inplace_true_divide = moduleName(ci, "inplaceTrueDivide");
		dumpBinaryFunc(output, ci, ci->nb_inplace_true_divide, nb_inplace_true_divide, "__truediv__");
	}

	output <<
		"\n"
		"static PyNumberMethods " << name << " = {\n"
		"\t" << nb_add << ", // nb_add\n"
		"\t" << nb_subtract << ", // nb_subtract\n"
		"\t" << nb_multiply << ", // nb_multiply\n"
		"\t" << nb_divide << ", // nb_divide\n"
		"\t" << nb_remainder << ", // nb_remainder\n"
		"\t" << nb_divmod << ", // nb_divmod\n"
		"\t" << nb_power << ", // nb_power\n"
		"\t" << nb_negative << ", // nb_negative\n"
		"\t" << nb_positive << ", // nb_positive\n"
		"\t" << nb_absolute << ", // nb_absolute\n"
		"\t" << nb_nonzero << ", // nb_nonzero\n"
		"\t" << nb_invert << ", // nb_invert\n"
		"\t" << nb_lshift << ", // nb_lshift\n"
		"\t" << nb_rshift << ", // nb_rshift\n"
		"\t" << nb_and << ", // nb_and\n"
		"\t" << nb_xor << ", // nb_xor\n"
		"\t" << nb_or << ", // nb_or\n"
		"\t" << nb_coerce << ", // nb_coerce\n"
		"\t" << nb_int << ", // nb_int\n"
		"\t" << nb_long << ", // nb_long\n"
		"\t" << nb_float << ", // nb_float\n"
		"\t" << nb_oct << ", // nb_oct\n"
		"\t" << nb_hex << ", // nb_hex\n"
		"\t" << nb_inplace_add << ", // nb_inplace_add\n"
		"\t" << nb_inplace_subtract << ", // nb_inplace_subtract\n"
		"\t" << nb_inplace_multiply << ", // nb_inplace_multiply\n"
		"\t" << nb_inplace_divide << ", // nb_inplace_divide\n"
		"\t" << nb_inplace_remainder << ", // nb_inplace_remainder\n"
		"\t" << nb_inplace_power << ", // nb_inplace_power\n"
		"\t" << nb_inplace_lshift << ", // nb_inplace_lshift\n"
		"\t" << nb_inplace_rshift << ", // nb_inplace_rshift\n"
		"\t" << nb_inplace_and << ", // nb_inplace_and\n"
		"\t" << nb_inplace_xor << ", // nb_inplace_xor\n"
		"\t" << nb_inplace_or << ", // nb_inplace_or\n"
		"\t" << nb_floor_divide << ", // nb_floor_divide\n"
		"\t" << nb_true_divide << ", // nb_true_divide\n"
		"\t" << nb_inplace_floor_divide << ", // nb_inplace_floor_divide\n"
		"\t" << nb_inplace_true_divide << ", // nb_inplace_true_divide\n"
		"#if PY_VERSION_HEX >= 0x02050000\n"
		"\t" << nb_index << ", // nb_index\n"
		"#endif\n"
		"};\n";
}

static void
dumpSequenceMethods(std::ostream &output, const ClassInfo *ci, const string &name)
{
	string sq_length("0");
	string sq_concat("0");
	string sq_repeat("0");
	string sq_item("0");
	string sq_slice("0");
	string sq_ass_item("0");
	string sq_ass_slice("0");
	string sq_contains("0");
	string sq_inplace_concat("0");
	string sq_inplace_repeat("0");

	if (!ci->sq_length.empty()) {
		sq_length = moduleName(ci, "size");
		dumpInquiryFunc(output, ci, ci->sq_length, sq_length, "__len__");
	}
	if (!ci->sq_concat.empty()) {
		sq_concat = moduleName(ci, "concat");
		dumpBinaryFunc(output, ci, ci->sq_concat, sq_concat, "__add__");
	}
	if (!ci->sq_repeat.empty()) {
		sq_repeat = moduleName(ci, "repeat");
		dumpSsizeArgFunc(output, ci, ci->sq_repeat, sq_repeat, "__mul__");
	}
	if (!ci->sq_item.empty()) {
		sq_item = moduleName(ci, "getitem");
		dumpSsizeArgFunc(output, ci, ci->sq_item, sq_item, "__getitem__");
	}
	if (!ci->sq_ass_item.empty()) {
		sq_ass_item = moduleName(ci, "setitem");
		dumpSsizeObjArgProc(output, ci, ci->sq_ass_item, sq_ass_item,
								"__setitem__");
	}
#if 0
	ssizessizeargfunc sq_slice;
	ssizessizeobjargproc sq_ass_slice;
#endif
	if (!ci->sq_contains.empty()) {
		sq_contains = moduleName(ci, "contains");
		dumpObjObjProc(output, ci, ci->sq_contains, sq_contains,
								"__contains__");
	}
	if (!ci->sq_inplace_concat.empty()) {
		sq_inplace_concat = moduleName(ci, "inplaceConcat");
		dumpBinaryFunc(output, ci, ci->sq_inplace_concat,
						sq_inplace_concat, "__iadd__");
	}
	if (!ci->sq_inplace_repeat.empty()) {
		sq_inplace_repeat = moduleName(ci, "repeat");
		dumpSsizeArgFunc(output, ci, ci->sq_inplace_repeat,
						sq_inplace_repeat, "__imul__");
	}
	output <<
		"\n"
		"static PySequenceMethods " << name << " = {\n"
		"\t" << sq_length << ", // sq_length\n"
		"\t" << sq_concat << ", // sq_concat\n"
		"\t" << sq_repeat << ", // sq_repeat\n"
		"\t" << sq_item << ", // sq_item\n"
		"\t" << sq_slice << ", // sq_slice\n"
		"\t" << sq_ass_item << ", // sq_ass_item\n"
		"\t" << sq_ass_slice << ", // sq_ass_slice\n"
		"\t" << sq_contains << ", // sq_contains\n"
		"\t" << sq_inplace_concat << ", // sq_inplace_concat\n"
		"\t" << sq_inplace_repeat << ", // sq_inplace_repeat\n"
		"};\n";
}

static void
dumpMappingMethods(std::ostream &output, const ClassInfo *ci, const string &name)
{
	string mp_length("0");
	string mp_subscript("0");
	string mp_ass_subscript("0");

	if (!ci->mp_length.empty()) {
		mp_length = moduleName(ci, "size");
		dumpInquiryFunc(output, ci, ci->mp_length, mp_length, "__len__");
	}
	if (!ci->mp_subscript.empty()) {
		mp_subscript = moduleName(ci, "subscript");
		dumpBinaryFunc(output, ci, ci->mp_subscript, mp_subscript, "__getitem__");
	}
#if 0
	objobjargproc mp_ass_subscript;
	if (!ci->mp_ass_subscript.empty()) {
		mp_ass_subscript = moduleName(ci, "ass_subscript");
		dumpObjObjArgProc(output, ci, ci->mp_ass_subscript,
					mp_ass_subscript, "__setitem__");
	}
#endif
	output <<
		"\n"
		"static PyMappingMethods " << name << " = {\n"
		"\t" << mp_length << ", // mp_length\n"
		"\t" << mp_subscript << ", // mp_subscript\n"
		"\t" << mp_ass_subscript << ", // mp_ass_subscript\n"
		"};\n";
}

static void
dumpAttrInit(std::ostream &output, const ClassInfo *ci)
{
	if (!ci->isBaseClass)
		return;

	bool first = true;
	for (AttrVec::const_iterator i = ci->attrs.begin();
					i != ci->attrs.end(); ++i) {
		const ClassAttrInfo &ai = *i;
		if (!ai.cache)
			continue;	// skip primitive types

		string init = moduleName(ci, "attrInit" + ai.name.str());
		string getter = moduleName(ci, "attrGet" + ai.name.str());
		if (first) {
			output <<
				"\n"
				"} // extern \"C\"\n";
			first = false;
		}
		output <<
			"\n"
			"PyObject*\n" <<
			init << "(PyObject* obj, void*)\n"
			"{\n"
			"\t// help subclasses initialize cached attribute\n"
			"\treturn " << getter << "(obj, NULL);\n"
			"}\n";
	}
	if (!first) {
		output <<
			"\n"
			"extern \"C\" {\n";
	}
}

bool
dumpTypeCode(const ClassInfo *ci)
{
	string tp_dealloc("0");
	string tp_print("0");
	string tp_compare("0");
	string tp_repr("0");
	string tp_as_number("0");
	string tp_as_sequence("0");
	string tp_as_mapping("0");
	string tp_hash("PyObject_HashNotImplemented");
	string tp_call("0");
	string tp_str("0");
	string tp_getattro("0");
	string tp_setattro("0");
	string tp_as_buffer("0");
	string tp_flags("Py_TPFLAGS_DEFAULT");
	string tp_doc("0");
	string tp_traverse("0");
	string tp_clear("0");
	string tp_richcompare("0");
	string tp_weaklistoffset("0");
	string tp_tp_iter("0");
	string tp_iternext("0");
	string tp_methods("0");
	string tp_members("0");
	string tp_getset("0");
	string tp_base("0");
	string tp_dict("0");
	string tp_descr_get("0");
	string tp_descr_set("0");
	string tp_dictoffset("0");
	string tp_init("0");
	string tp_alloc("0");
	string tp_new("0");
	string tp_free("0");
	string tp_is_gc("0");
	string tp_bases("0");
	string tp_mro("0");
	string tp_cache("0");
	string tp_subclasses("0");
	string tp_weaklist("0");
	string tp_del("0");

	string objectName(moduleName(ci, "object"));
	string cppName(qualify(ci->cd->scope, ci->name.str()));
	string filename(moduleName(ci, "Object.cpp"));
	std::auto_ptr<std::ostream> outs(outputStream(filename).release());
	std::ostream &output = *outs.get();

	output <<
		"#include \"" << module << ".h\"\n"
		"#include <structmember.h> /* for PyMemberDef*/\n";
	if (ci->print)
		output << "#include <sstream>\n";

	output << "\n"
		"#ifndef PY_STUPID\n"
		"// workaround for Python API missing const's.\n"
		"# define PY_STUPID (char*)\n"
		"#endif\n"
		"\n"
		"#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)\n"
		"typedef int Py_ssize_t;\n"
		"#define PY_SSIZE_T_MAX INT_MAX\n"
		"#define PY_SSIZE_T_MIN INT_MIN\n"
		"#endif\n";

	if (!nameSpace.empty())
		output <<
			"\n"
			"namespace " << nameSpace << " {\n";

	output <<
		"\n"
		<< cppName << "*\n"
		"getInst(" << objectName << "* self)\n"
		"{\n";
	if (ci->isEmbedded)
		output <<
			"\tif (!self->_initialized)\n"
			"\t\tthrow std::logic_error(\"underlying C++ "
				<< pythonName(ci) << " object is missing\");\n"
			"\treturn self->_inst();\n";
	else
		output <<
			"\t" << cppName << "* inst = self->_inst();\n"
			"\tif (inst == NULL)\n"
			"\t\tthrow std::logic_error(\"underlying C++ "
				<< pythonName(ci) << " object is missing\");\n"
			"\treturn inst;\n";
	output<< 
		"}\n";

	output <<
		"\n"
		"extern \"C\" {\n";

	// object definition
	if (!ci->isFinal)
		tp_dictoffset = "offsetof(" + objectName + ", _inst_dict)";

	if (ci->isWrappySubclass)
		tp_weaklistoffset = "offsetof(" + objectName + ", _weaklist)";

	//
	// find member functions that act like special Python type functions
	//

	// print to file
	if (ci->print || !ci->str.empty()) {
		tp_print = moduleName(ci, "print");
		dumpPrint(output, ci, tp_print);
	}

	if (!ci->compare.empty()) {
		tp_compare = moduleName(ci, "compare");
		dumpCompare(output, ci, tp_compare);
	}

	if (!ci->lt.empty() || !ci->eq.empty()) {
		tp_richcompare = moduleName(ci, "richcompare");
		dumpRichCompare(output, ci, tp_richcompare);
	}

	if (!ci->call.empty()) {
		tp_call = moduleName(ci, "call");
		dumpCall(output, ci, tp_call);
	}

	if (ci->genIdentityHash) {
		tp_hash = moduleName(ci, "hash");
		dumpPtrHash(output, ci, tp_hash);
	} else if (!ci->isEmbedded || !ci->hash.empty()) {
		tp_hash = moduleName(ci, "hash");
		if (ci->hash.empty())
			dumpPtrHash(output, ci, tp_hash);
		else
			dumpHash(output, ci, tp_hash);
	}

	// string
	if (!ci->str.empty() || ci->print) {
		tp_str = moduleName(ci, "str");
		dumpStr(output, ci, tp_str);
	}

	if (ci->hasNumberMethods) {
		tp_as_number = "&" + moduleName(ci, "asNumber");
		dumpNumberMethods(output, ci, tp_as_number.substr(1));
		tp_flags += " | Py_TPFLAGS_CHECKTYPES";
	}
	if (ci->hasSequenceMethods) {
		tp_as_sequence = "&" + moduleName(ci, "asSequence");
		dumpSequenceMethods(output, ci, tp_as_sequence.substr(1));
	}
	if (ci->hasMappingMethods) {
		tp_as_mapping = "&" + moduleName(ci, "asMap");
		dumpMappingMethods(output, ci, tp_as_mapping.substr(1));
	}

	// methods table
	dumpMethods(output, ci, &tp_methods);

	dumpAttributes(output, ci, &tp_members, &tp_getset);

	dumpAttrInit(output, ci);

	tp_doc = moduleName(ci, "doc");
	if (ci->isAbstractType || ci->constructors.empty()) {
		// abstract type or only non-public constructors
		tp_new = moduleName(ci, ci->name.str());
		dumpInaccessibleConstructor(output, ci, tp_new, tp_doc);
	} else {
		tp_init = moduleName(ci, ci->name.str());
		tp_alloc = "PyType_GenericAlloc";
		tp_new = "PyType_GenericNew";
		dumpConstructor(output, ci, tp_init, tp_doc);
	}

	tp_dealloc = moduleName(ci, "dealloc");
	dumpDeallocator(output, ci, tp_dealloc);

	if (ci->isWrappySubclass) {
		string prefix(qualify(ci->cd->scope, ci->name.str()));
		string objectType(moduleName(ci, "objectType"));
		output <<
			// create function
			"\n"
			<< exportTag << "PyObject*\n"
			<< prefix << "::wpyNew() const\n"
			"{\n"
			"\t" << objectName << "* self = static_cast<"
				<< objectName << "*>(" << objectName
				<< "Type.tp_alloc(&" << objectName
				<< "Type, 0));\n"
			"\tif (" << module << "Debug >= 6)\n"
			"\t\tstd::cerr << \"Allocate " << ci->name
						<< ": \" << self << '\\n';\n"
			"\tif (self == NULL)\n"
			"\t\tthrow wrappy::PythonError();\n";
		if (!ci->isFinal)
			output <<
				"\tself->_inst_dict = PyDict_New();\n"
				"\tif (self->_inst_dict == NULL) {\n"
				"\t\tPy_DECREF(self);\n"
				"\t\tthrow wrappy::PythonError();\n"
				"\t}\n";
		if (ci->isEmbedded)
			output <<
				"\towner = true;\n"
				"\ttry {\n"
				"\t\tnew (&self->_inst_data) " << prefix
								<< "(*this);\n"
				"\t\tself->_initialized = true;\n"
				"\t} catch (...) {\n"
				"\t\tPy_DECREF(self);\n"
				"\t\tthrow;\n"
				"\t}\n";
		else
			output <<
				"\towner = false;\n"
				"\tself->_inst_data = const_cast<" << prefix
						<< "*>(static_cast<" << prefix
						<< " const*>(this));\n";
		output <<
			"\tself->_inst_data->wpyAssociate(self);\n";
			dumpTypeAttrInit(output, 1, ci);
		output <<
			"\treturn static_cast<PyObject*>(self);\n"
			"}\n";
	}

	output <<
		"\n"
		"} // extern \"C\"\n";

	if (ci->isBaseClass) {
		tp_flags += " | Py_TPFLAGS_BASETYPE";
	}

	// Python 2.6 type definition -- must be last
	string objectType(moduleName(ci, "objectType"));
	output <<
		"\n" <<
		exportTag << "PyTypeObject " << objectType << " = {\n"
		"\tPyVarObject_HEAD_INIT(NULL, 0)\n"
		"\t\"" << module << '.' << pythonName(ci) << "\", // tp_name\n"
		"\tsizeof (" << objectName << "), // tp_basicsize\n"
		"\t0, // tp_itemsize\n"
		"\t" << tp_dealloc << ", // tp_dealloc\n"
		"\t" << tp_print << ", // tp_print\n"
		"\t0, // tp_getattr\n"
		"\t0, // tp_setattr\n"
		"\t" << tp_compare << ", // tp_compare\n"
		"\t" << tp_repr << ", // tp_repr\n"
		"\t" << tp_as_number << ", // tp_as_number\n"
		"\t" << tp_as_sequence << ", // tp_as_sequence\n"
		"\t" << tp_as_mapping << ", // tp_as_mapping\n"
		"\t" << tp_hash << ", // tp_hash\n"
		"\t" << tp_call << ", // tp_call\n"
		"\t" << tp_str << ", // tp_str\n"
		"\t" << tp_getattro << ", // tp_getattro\n"
		"\t" << tp_setattro << ", // tp_setattro\n"
		"\t" << tp_as_buffer << ", // tp_as_buffer\n"
		"\t" << tp_flags << ", // tp_flags\n"
		"\t" << tp_doc << ", // tp_doc\n"
		"\t" << tp_traverse << ", // tp_traverse\n"
		"\t" << tp_clear << ", // tp_clear\n"
		"\t" << tp_richcompare << ", // tp_richcompare\n"
		"\t" << tp_weaklistoffset << ", // tp_weaklistoffset\n"
		"\t" << tp_tp_iter << ", // tp_tp_iter\n"
		"\t" << tp_iternext << ", // tp_iternext\n"
		"\t" << tp_methods << ", // tp_methods\n"
		"\t" << tp_members << ", // tp_members\n"
		"\t" << tp_getset << ", // tp_getset\n"
		"\t" << tp_base << ", // tp_base\n"
		"\t" << tp_dict << ", // tp_dict\n"
		"\t" << tp_descr_get << ", // tp_descr_get\n"
		"\t" << tp_descr_set << ", // tp_descr_set\n"
		"\t" << tp_dictoffset << ", // tp_dictoffset\n"
		"\t" << tp_init << ", // tp_init\n"
		"\t" << tp_alloc << ", // tp_alloc\n"
		"\t" << tp_new << ", // tp_new\n"
		"\t" << tp_free << ", // tp_free\n"
		"\t" << tp_is_gc << ", // tp_is_gc\n"
		"\t" << tp_bases << ", // tp_bases\n"
		"\t" << tp_mro << ", // tp_mro\n"
		"\t" << tp_cache << ", // tp_cache\n"
		"\t" << tp_subclasses << ", // tp_subclasses\n"
		"\t" << tp_weaklist << ", // tp_weaklist\n"
		"\t" << tp_del << ", // tp_del\n"
		"\t0, // tp_version_tag\n"
		"};\n";

	if (!nameSpace.empty())
		output <<
			"\n"
			"} // namespace " << nameSpace << "\n";

	dumpWrapTypeOutline(output, ci);

	// flush output and check return status
	output.flush();
	if (output.good())
		return true;
	std::cerr << programName << ": error writing " << filename << '\n';
	return false;
}

typedef std::map<Symbol, const ClassInfo*> AllAttrs;

void
findAllAttrs(const ClassInfo *ci, AllAttrs *all)
{
	for (AttrVec::const_iterator i = ci->attrs.begin();
						i != ci->attrs.end(); ++i) {
		const ClassAttrInfo &ai = *i;
		if (!ai.cache)
			continue;	// skip primitive types
		AllAttrs::iterator j = all->find(ai.name);
		if (j != all->end())
			continue;
		(*all)[ai.name] = ci;
	}

	// recurse into base classes
	for (CIList::const_iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
		const ClassInfo *base = *i;
		findAllAttrs(base, all);
	}
}


void
dumpTypeAttrInit(std::ostream &output, int indent, const ClassInfo *ci)
{
	string bi(tab(indent));		// base indent
	bool first = true;
	AllAttrs all;
	findAllAttrs(ci, &all);
	for (AllAttrs::iterator i = all.begin(); i != all.end(); ++i) {
		if (first) {
			first = false;
			output << bi << "// initialize cached attributes\n"
				<< bi << "PyObject* attrTmp;\n";
		}
		Symbol attrName = i->first;
		const ClassInfo *subci = i->second;
		string fname;
		if (subci == ci)
			fname = moduleName(subci, "attrGet" + attrName.str());
		else
			fname = moduleName(subci, "attrInit" + attrName.str());
		output << bi << "attrTmp = " << fname << "(self, NULL);\n";
		output << bi << "Py_XDECREF(attrTmp);\n";
	}
	if (!first)
		output << bi << "PyErr_Clear();\n";
}
