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

// $Id: module.cpp 37484 2012-09-19 00:09:48Z chimera $

#include <iostream>
#include <algorithm>
#include <list>
#include "Reg.h"
#include "ParseHeader.h"
#include "CvtType.h"
#include "common.h"
#include "compute.h"
#include "method.h"

using std::string;

bool dumpModuleHeader();

void
dumpInitTypes(std::ostream &output)
{
	string nsPrefix;
	if (!nameSpace.empty())
		nsPrefix = nameSpace + "::";

	typedef std::list<const ClassInfo*> CIDList;
	CIDList cidl;
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		const ClassInfo *ci = (*i).second;
		if (ci->skipClass)
			continue;
		if (ci->baseClasses.empty())
			cidl.push_front(ci);
		else
			cidl.push_back(ci);
	}
	if (cidl.empty())
		return;

	typedef std::set<const ClassInfo*> CISet;
	CISet visited;
	while (!cidl.empty()) {
		const ClassInfo *ci = cidl.front();
		cidl.pop_front();
		for (CIList::const_iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
			if ((*i)->skipClass)
				continue;
			if (visited.find(*i) == visited.end()) {
				// One of the base classes hasn't been
				// visited yet, save for later.
				cidl.push_back(ci);
				goto try_again;
			}
		}
		// Have to put the following within {}s so the 'goto try_again'
		// will work (since we can't contine 2 levels out).
		{
			// Check if enclosing scope is a class, and if
			// so that we have already processed it.
			Decl *scope = ci->cd->scope;
			ClassDecl *cd = dynamic_cast<ClassDecl *>(scope);
			CDCIMap::iterator j;
			if (cd == NULL)
				j = classInfoMap.end();
			else  {
				j = classInfoMap.find(cd);
				if (j != classInfoMap.end()
				&& visited.find(j->second) == visited.end()) {
					// enclosing class hasn't been
					// visited yet, save for later.
					cidl.push_back(ci);
					continue;
				}
			}

			visited.insert(ci);
			output << '\n';
			string objectType = moduleName(ci, "objectType", true); 
			if (ci->isWrappySubclass) {
				output << "\t" << objectType
				<< ".tp_base = &wrappy::Object_Type;\n";
			}
			if (!ci->baseClasses.empty()) {
				string tp_bases = objectType + ".tp_bases";
				output <<
					"\t" << tp_bases << " = PyTuple_Pack("
						<< ci->baseClasses.size();
				for (CIList::const_iterator i
						= ci->baseClasses.begin();
						i != ci->baseClasses.end();
						++i) {
					ClassInfo *base = *i;
					string baseName = moduleName(base,
							"objectType", true);
					output << ",\n\t\treinterpret_cast<PyObject*>(&"
						<< baseName << ")";
				}
				output << ");\n";
			}
			const char *ready = ci->isFinal ?
				"PyType_Ready" : "wrappy::MutableType_Ready";
			output <<
				"\tif (" << ready << "(&" << objectType
								<< ") < 0)\n"
				"\t\treturn;\n"
				"\tPy_INCREF(&" << objectType << ");\n";
			if (j == classInfoMap.end())
				output <<
					"\tPyModule_AddObject(module, \""
					<< ci->name
					<< "\", reinterpret_cast<PyObject*>(&"
					<< objectType << "));\n";
			else {
				string enclosing = moduleName(j->second,
							"objectType", true);
				output <<
					"\twrappy::PyType_AddObject(&"
					<< enclosing << ", \"" << ci->name
					<< "\", reinterpret_cast<PyObject*>(&"
					<< objectType << "));\n";
			}
			for (AttrVec::const_iterator i = ci->constants.begin();
						i != ci->constants.end(); ++i) {
				const ClassAttrInfo &ai = *i;
				output <<
					"\twrappy::PyType_AddObject(&"
					<< objectType << ", \"" << ai.name
					<< "\", "
					<< qualify(ci->cd->scope, ai.get, true)
					<< ");\n";
			}
		}
	try_again:;
	}
}

bool
dumpModuleCode()
{
	string filename(module);
	filename += ".cpp";
	std::auto_ptr<std::ostream> outs(outputStream(filename).release());
	std::ostream &output = *outs.get();
	string nsPrefix;
	if (!nameSpace.empty())
		nsPrefix = nameSpace + "::";

	output << "#include \"" << module << ".h\"\n";

	const DeclList *modDecls;
	if (moduleScope != NULL)
		modDecls = &moduleScope->decls;
	else
		modDecls = &globalDecls;

	// find all include files for C++ global (namespace) functions
	SymbolSet includes;
	for (DeclList::RAconst_iterator i = modDecls->rAbegin();
					i != modDecls->rAend(); ++i) {
		const Decl *d = i->second;
		if (d->tag.empty() || d->dtype != Decl::FUNCTION
		|| d->tag.str().find(':') != string::npos
		|| d->tag.str().compare(0, 8, "operator") == 0)
			// skip non-functions, class function declarations,
			// and operators
			continue;
		if (d->filename != "standard input")
			includes.insert(d->filename);
	}
	for (SymbolSet::iterator i = includes.begin(); i != includes.end();
									++i) {
		string tmp = i->str();
		if (tmp[0] == '<' || tmp[0] == '"')
			output << "#include " << tmp << '\n';
		else
			output << "#include \"" << tmp << "\"\n";
	}

	output <<
		"#include <stdlib.h>	/* for getenv() and atoi() */\n"
		"#include <typeinfo>\n"
		"\n"
		"#ifndef PY_STUPID\n"
		"// workaround for Python API missing const's.\n"
		"# define PY_STUPID (char *)\n"
		"#endif\n";

	if (!derivedClasses.empty()) {
		// generate pyObject function for base classes
		output <<
			"\n"
			"namespace wrappy {\n";
		for (DerivedClasses::iterator i = derivedClasses.begin();
						i != derivedClasses.end(); ) {
			DerivedClasses::iterator end
					= derivedClasses.upper_bound(i->first);
			if (i->first == "wrappy::Obj") {
				i = end;
				continue;
			}
			const ClassDecl *bcd = classDeclsMap[i->first];
			if (bcd == NULL) {
				i = end;
				continue;
			}
			CDCIMap::iterator j = classInfoMap.find(bcd);
			if (j == classInfoMap.end()) {
				i = end;
				continue;
			}
#if 0
			const ClassInfo *base = j->second;
			if (base->isPythonClass) {
				i = end;
				continue;
			}
			string prefix(qualify(moduleScope,
						i->first.str(), true));
			output <<
				"\n"
				"template <> PyObject*\n"
				"pyObject(" << prefix << "* o)\n"
				"{\n"
				"\twrappy::Obj* wo = dynamic_cast<wrappy::Obj*>(o);\n"
				"\tif (wo == NULL) {\n"
				"\t\tPy_INCREF(Py_None);\n"
				"\t\treturn Py_None;\n"
				"\t}\n"
				"\treturn wo->wpyGetObject();\n"
				"}\n";
#endif
			i = end;
		}
		output <<
			"\n"
			"} // namespace wrappy\n";
	}

	if (!nameSpace.empty())
		output <<
			"\n"
			"namespace " << nameSpace << " {\n";
	string errObj(module + "ErrorObj");
	output <<
		"\n"
		"PyObject* " << errObj << ";\n"
		"int " << module << "Debug;\n";

	// TODO: use exception specification knowledge
	output <<
		"\n"
		"void\n"
		<< module << "Error()\n"
		"{\n"
		"\t// generic exception handler\n"
		"\ttry {\n"
		"\t\t// rethrow exception to look at it\n"
		"\t\tthrow;\n"
		"\t} catch (wrappy::PythonError&) {\n"
		"\t\t// nothing to do, already set\n"
		"\t} catch (wrappy::Obj& wo) {\n"
		"\t\tPyObject* obj = wo.wpyGetObject();\n"
		"\t\tPyErr_SetObject(" << errObj << ", obj);\n"
		"\t\tPy_DECREF(obj);\n"
		"\t} catch (std::bad_alloc&) {\n"
		"\t\tPyErr_SetString(PyExc_MemoryError, \"not enough memory\");\n"
#if 0
		"\t} catch (std::domain_error& e) {\n"
		"\t\tPyErr_Format(" << errObj << ", \"domain_error: %s\", e.what());\n"
#else
		// use domain_error base class logic_error for error message
#endif
		"\t} catch (std::invalid_argument& e) {\n"
		"\t\tPyErr_SetString(PyExc_TypeError, e.what());\n"
		"\t} catch (std::length_error& e) {\n"
		"\t\tPyErr_SetString(PyExc_MemoryError, e.what());\n"
		"\t} catch (std::out_of_range& e) {\n"
		"\t\tPyErr_SetString(PyExc_IndexError, e.what());\n"
		"\t} catch (std::overflow_error& e) {\n"
		"\t\tPyErr_SetString(PyExc_OverflowError, e.what());\n"
		"\t} catch (std::range_error& e) {\n"
		"\t\tPyErr_Format(" << errObj << ", \"range_error: %s\", e.what());\n"
		"\t} catch (std::underflow_error& e) {\n"
		"\t\tPyErr_Format(" << errObj << ", \"underflow_error: %s\", e.what());\n"
		"\t} catch (std::logic_error& e) {\n"
		"\t\tPyErr_SetString(PyExc_ValueError, e.what());\n"
		"\t} catch (std::ios_base::failure& e) {\n"
		"\t\tPyErr_SetString(PyExc_IOError, e.what());\n"
		"\t} catch (std::runtime_error& e) {\n"
		"\t\tPyErr_SetString(" << errObj << ", e.what());\n"
		"#ifdef _WIN32\n"
		"\t} catch (wrappy::SE_Exception& e) {\n"
		"\t\tif (" << module << "Debug)\n"
		"\t\t\tthrow; // fatal exception\n"
		"\t\tPyErr_SetString(" << errObj << ", e.what());\n"
		"#endif\n"
		"\t} catch (std::exception& e) {\n"
		"\t\tPyErr_Format(" << errObj << ", \"unknown error (%s): %s\", typeid(e).name(), e.what());\n"
		"\t} catch (...) {\n"
		"\t\tif (" << module << "Debug)\n"
		"\t\t\tthrow; // fatal exception\n"
		"\t\tPyErr_SetString(" << errObj << ", \"unknown C++ exception\");\n"
		"\t}\n"
		"}\n";

	typedef std::vector<std::pair<string, string> > MMVec;
	MMVec modMethods;

	// C++ global (namespace) functions are Python module methods
	DeclList::RAconst_iterator next;
	for (DeclList::RAconst_iterator i = modDecls->rAbegin(); i != modDecls->rAend(); i = next) {
		next = modDecls->rAupper_bound(i->first);
		const Decl *d = i->second;
		if (d->dtype == Decl::VARIABLE) {
			std::cerr << programName
				<< ": unable to wrap global variable: "
				<< d->tag << '\n';
			continue;
		}
		if (d->tag.empty() || d->dtype != Decl::FUNCTION
		|| d->tag.str().find(':') != string::npos
		|| d->tag.str().compare(0, 8, "operator") == 0)
			// skip non-functions, class function declarations,
			// and operators
			continue;
		MethodMap funcs;
		for (DeclList::RAconst_iterator j = i; j != next; ++j) {
			const FuncDecl *fd
				= dynamic_cast<const FuncDecl *>(j->second);
			if (fd == NULL)
				continue;
			funcs.insert(MethodMap::value_type(fd->tag, fd));
		}
		MethodInfo mi;
		dumpMethod(output, NULL, funcs.begin(), funcs.end(), &mi);
		string proto = mi.doc;
		output <<
			"\n"
			"static const char " << module << d->tag
				<< "_doc[] = \"" << stringize(proto) << "\";\n";
		modMethods.push_back(std::make_pair(d->tag.str(), mi.ml_flags));
	}
	// fill in method table
	output <<
		"\n"
		"PyMethodDef " << module << "Methods[] = {\n";
	for (MMVec::iterator i = modMethods.begin(); i != modMethods.end();
									++i) {
		std::pair<string, string> &mm = *i;
		output <<
			"\t{\n"
			"\t\t\"" << mm.first << "\", (PyCFunction) "
					<< module << '_' << mm.first << ",\n"
			<< "\t\t" << mm.second << ", " << module
						<< mm.first << "_doc\n"
			<< "\t},\n";
	}
	output <<
		"\t{ NULL, NULL, 0, NULL }\n"
		"};\n";

	// generate module documentation string
	output <<
		"\n"
		"static const char " << module << "_doc[] = \"wrappy2 generated module.\\n\\\nFunction signature documentation only.\\n\\\nSee C++ headers for more documentation.\";\n";
	if (!nameSpace.empty())
		output <<
			"\n"
			"} // namespace " << nameSpace << "\n";

	output <<
		"\n"
		"PyMODINIT_FUNC\n"
		"init" << module << "()\n"
		"{\n";
	output <<
		"\tPyObject* module = Py_InitModule4(\"" << module
				<< "\", " << nsPrefix << module << "Methods,\n"
		"\t\t\t" << nsPrefix << module << "_doc, NULL, PYTHON_API_VERSION);\n"
		"\tif (module == NULL)\n"
		"\t\treturn;\n"
		"\n"
		"\tconst char* debug = getenv(\"" << module << "Debug\");\n"
		"\tif (debug != NULL)\n"
		"\t\t" << nsPrefix << module << "Debug = atoi(debug);\n"
		"\n"
		"\t" << nsPrefix << module
				<< "ErrorObj = PyErr_NewException(PY_STUPID \"" 
				<< module << ".error\", NULL, NULL);\n"
		"\tif (" << nsPrefix << module << "ErrorObj == NULL)\n"
		"\t\treturn;\n"
		"\tPy_INCREF(" << nsPrefix << module << "ErrorObj);\n"
		"\tPyModule_AddObject(module, \"error\", " << nsPrefix
						<< module << "ErrorObj);\n"
		"\tinitlibwrappy2();\n";

	// module constants
	output <<
		"\n";
	bool first = true;
	for (DeclList::const_iterator i = modDecls->begin();
						i != modDecls->end(); ++i) {
		if ((*i)->dtype != Decl::CONSTANT)
			continue;
		static bool warned = false;
		if (!warned) {
			std::cerr << programName
			<< ": warning: module constants can be overriden\n";
			warned = true;
		}
		if (first) {
			output << "\tPyObject* o;\n";
			first = false;
		}
		const VarDecl *vd = static_cast<VarDecl *>(*i);
		CvtType cvt(moduleScope, vd->type);
		output <<
			"\to = " << make_buildvalue(cvt.bvFormat(),
				qualify(moduleScope, cvt.bvArg(vd->tag.str()),
									true))
				<< ";\n"
			"\tPyModule_AddObject(module, \"" << (*i)->tag
				<< "\", o);\n";
	}
	
	// module Python classes
	// make sure we initialize base classes before derived ones
	dumpInitTypes(output);

	output <<
		"}\n";

	// flush output and check return status
	output.flush();
	if (output.good())
		return dumpModuleHeader();
	std::cerr << programName << ": error writing " << filename << '\n';
	return false;
}

bool
dumpModuleHeader()
{
	// output common includes and shared conversion functions

	string filename(module);
	filename += ".h";
	std::auto_ptr<std::ostream> outs(outputStream(filename).release());
	std::ostream &output = *outs.get();

	output <<
		"#ifndef " << module << "_h\n"
		"# define " << module << "_h\n"
		"# if defined(_MSC_VER) && (_MSC_VER >= 1020)\n"
		"#  pragma once\n"
		"# endif\n"
		"\n"
		"// include Python.h first so standard defines are the same\n"
		"# define PY_SSIZE_T_CLEAN 1\n"
		"# include <Python.h>\n"
		"# include <new>\n"
		"# include <WrapPy3.h>\n";
	for (StrVec::iterator i = extraHeaders.begin(); i != extraHeaders.end();
									++i) {
		const string &header = *i;
		if (!header.empty() && (header[0] == '"' || header[0] == '<'))
			output << "# include " << header << "\n";
		else
			output << "# include \"" << header << "\"\n";
	}
	typedef std::set<string> StrSet;
	StrSet includes;
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		const ClassInfo *ci = (*i).second;
		if (ci->skipClass) {
			includes.insert(ci->includeFile);
			continue;
		}
		output << "# include \"" << moduleName(ci, "Object.h")
								<< "\"\n";
	}
#ifdef TODO
	for (StrSet::iterator i = includes.begin(); i != includes.end(); ++i) {
		output << *i << '\n';
	}
#endif

	if (!nameSpace.empty())
		output <<
			"\n"
			"namespace " << nameSpace << " {\n";
	output <<
		"\n"
		<< exportTag << "extern void " << module << "Error();\n"
		"extern int " << module << "Debug;\n";
	if (!nameSpace.empty())
		output <<
			"\n"
			"} // namespace " << nameSpace << "\n";

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
