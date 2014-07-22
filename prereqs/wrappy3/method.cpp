// Copyright (c) 1998-2000 The Regents of the University of California.  // All rights reserved.
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

// $Id: method.cpp 29912 2010-01-28 01:31:29Z gregc $

#include <sstream>
#include <algorithm>
#include <assert.h>
#include <stdlib.h>
#include "common.h"
#include "method.h"
#include "compute.h"
#include "CvtType.h"
#include "type.h"

using std::string;

#undef ATTR_CACHE	/* caching is now done in the attr[GS]et functions */

struct ParseInfo {
	string	returnFormat;
	string	returnParams;
	string	funcParams;
	string	toCppArgs;
	string	cleanup;
#ifdef ATTR_CACHE
	Symbol	cacheReturn;
#endif
	string	ml_flags;
	string	doc;
};

void
dumpParseArgs(std::ostream &output, int indent, bool single, const FuncDecl *fd,
	const string &pyname, MethodType mtype, const ClassInfo *ci,
	int skipTypeCheck, const string &error_return, /*OUT*/ ParseInfo *pi)
{
	bool isStatic = fd->returnType.compare(0, 7, "static ") == 0;
	string bi(tab(indent));		// base indent
	pi->doc = pyname + '(';
	string returnProto;
	// declare local variables for each argument
	std::ostringstream parseFormat, parseParams, kwParams;
	std::ostringstream toCppArgs;
	bool hasDefault = false;
#ifdef ATTR_CACHE
	bool cacheAttribute = false;
	if (mtype == Attribute) {
		AttrVec::const_iterator i = std::find_if(ci->attrs.begin(),
			ci->attrs.end(), AttrHasName(fd->attrName));
		if (i != ci->attrs.end())
			cacheAttribute = i->cache;
	}
#endif
	if (!fd->returnType.empty() && fd->returnType != "void"
	&& fd->returnType != "static void") {
		CvtType rt(fd->scope, fd->returnType);
		pi->returnFormat = rt.bvFormat();
		pi->returnParams = ", "
				+ qualify(fd->scope, rt.bvArg("_result"));
#ifdef ATTR_CACHE
		if (cacheAttribute && rt.cache())
			pi->cacheReturn = fd->attrName;
#endif
		returnProto = rt.pythonType();
	}

	// want to know if there are zero, one, or "many" input arguments
	int numInArgs = 0;
	const Argument *first_in_arg = NULL;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {
		if (!i->in)
			continue;
		if (numInArgs < 0) {
			--numInArgs;
			continue;
		}
		++numInArgs;
		if (numInArgs == 1)
			first_in_arg = &*i;
		if (!i->defValue.empty())
			// if default arguments are present, it is always "many"
			numInArgs = -numInArgs;
	}
	if (numInArgs == 1) {
		CvtType arg(fd->scope, first_in_arg->type, false);
		if (arg.requireAPT())
			numInArgs = -1;
	}

	int argCount = 0;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {
		argCount += 1;
		string type = i->type;
		if (i->out && *type.rbegin() == '*') {
			// TODO: handle output arrays too
			// strip trailing *
			type.erase(type.size() - 1);
		}
		string ptArg;
		if (i->in) {
			ptArg = "_ptArg";
			if (numInArgs != 1 || !single)
				ptArg += itostr(argCount);
		}
		bool defaultPyObject = false;
		CvtType arg(fd->scope, type, i->in && i->null_ok);
		if (i->in && (numInArgs != 1 || !single)) {
			output << bi << qualify(fd->scope, arg.aptType())
								<< ' ' << ptArg;
			if (!i->defValue.empty()) {
				if (arg.aptType() == "PyObject*") {
					output << " = NULL";
					defaultPyObject = true;
				} else
					output << " = " << qualify(fd->scope,
						arg.cppToApt(i->defValue));
			}
			output << ";\n";
		}
		if (i->in && mtype != Attribute) {
			if (*pi->doc.rbegin() != '(')
				pi->doc += ", ";
			pi->doc += i->name.str() + ": " + arg.pythonType();
			if (!i->defValue.empty()) {
				pi->doc += " = ";
				if (*i->type.rbegin() == '*'
				&& (i->defValue == "NULL"
							|| i->defValue == "0"))
					pi->doc += "None";
				else
					pi->doc += i->defValue;
			}
		}
		if (!pi->funcParams.empty())
			pi->funcParams += ", ";
		if (i->out && !arg.hasImplicitAddress())
			pi->funcParams += '&';	// out params need leading &
		pi->funcParams += arg.cppToActual("cppArg" + itostr(argCount));
		string cleanup = arg.aptCleanup(ptArg);
		if (!cleanup.empty())
			pi->cleanup += bi + '\t' + cleanup + ";\n"; 
		if (i->out) {
			pi->returnFormat += arg.bvFormat();
			pi->returnParams += ", "
				+ qualify(fd->scope,
					arg.bvArg("cppArg" + itostr(argCount)));
#ifdef ATTR_CACHE
			if (cacheAttribute && arg.cache()
			&& pi->returnFormat.size() == 1)
				pi->cacheReturn = fd->attrName;
#endif
			if (mtype != Attribute) {
				if (!returnProto.empty())
					returnProto += ", ";
				returnProto += arg.pythonType();
			}
		}
		string decl = qualify(fd->scope, arg.cppTypeDecl(
					"cppArg" + itostr(argCount),
					i->out || defaultPyObject));
		if (defaultPyObject) {
			toCppArgs
				<< bi << '\t' << decl << ";\n"
				<< bi << "\tif (" << ptArg << " == NULL)\n"
				<< bi << "\t\t" << "cppArg" << itostr(argCount)
					<< " = " << i->defValue << ";\n"
				<< bi << "\telse {\n";
			bi += '\t';
		}
		if (i->in && (skipTypeCheck != argCount)
		&& ((numInArgs == 1 && single) || arg.needTypeCheck())) {
			toCppArgs
				<< bi << "\tif (!"
					<< arg.typeCheck(ptArg) << ")\n"
				<< bi << "\t\tthrow std::invalid_argument(\"argument "
					<< argCount << " should be a "
					<< arg.pythonType() << "\");\n";
		}
#ifdef ATTR_CACHE
		if (cacheAttribute && !i->out && arg.cache()
		&& !fd->attrName.empty()) {
			// Note: all arguments that need to be cached
			// are PyObject*'s.
			// TODO: cache other arguments
			toCppArgs << bi << "\t// cache attribute\n"
				<< bi << "\tstatic PyObject* ca" << argCount
									<< ";\n"
				<< bi << "\tif (ca" << argCount << " == NULL)\n"
				<< bi << "\t\tca" << argCount
					<< " = PyUnicode_InternFromString(\"__cached_"
					<< fd->attrName << "__\");\n"
				<< bi << "\tPyDict_SetItem(self->_inst_dict, ca"
					<< argCount << ", " << ptArg << ");\n";
		}
#endif
		if (defaultPyObject)
			toCppArgs << bi << "\tcppArg" << itostr(argCount);
		else
			toCppArgs << bi << '\t' << decl;
		if (i->in) {
			string init;
			if (numInArgs == 1 && single)
				init = qualify(fd->scope, arg.pyToCpp(ptArg));
			else
				init = qualify(fd->scope, arg.aptToCpp(ptArg));
			// Elide copy constructor if possible
			// for types that aren't copy constructable.
			string type = decl.substr(0, decl.find(' '));
			if (init.compare(0, type.size(), type) == 0) {
				toCppArgs << init.substr(type.size());
			} else {
				toCppArgs << " = " << init;
			}
		}
		toCppArgs << ";\n";
		if (defaultPyObject) {
			bi = bi.substr(0, bi.size() - 1);
			toCppArgs << bi << "\t}\n";
		}
		if (!i->in)
			continue;
		if (!i->defValue.empty() && !hasDefault) {
			hasDefault = true;
			parseFormat << '|';
		}
		parseFormat << arg.aptFormat();
		parseParams << ", &" << ptArg;
		if (hasDefault)
			kwParams << " PY_STUPID \"" << i->name << "\",";
		else
			kwParams << " PY_STUPID \"\",";
	}
	parseFormat << ':' << pyname;
	if (!single) {
		if (mtype == Attribute)
			pi->ml_flags = "METH_VARARGS";
		else
			pi->ml_flags = "METH_VARARGS | METH_KEYWORDS";
	} else if (numInArgs == 0)
		pi->ml_flags = "METH_NOARGS";
	else if (numInArgs == 1 /* implicitly known that && !hasDefault */)
		pi->ml_flags = "METH_O";
	else if (!hasDefault)
		pi->ml_flags = "METH_VARARGS";
	else
		pi->ml_flags = "METH_VARARGS | METH_KEYWORDS";

	if (pi->ml_flags == "METH_VARARGS") {
		output 
			<< bi << "if (!PyArg_ParseTuple(_args, \""
				<< parseFormat.str() << "\""
				<< parseParams.str() << "))\n"
			<< bi << "\treturn " << error_return << ";\n";
	} else if (pi->ml_flags == "METH_VARARGS | METH_KEYWORDS") {
		if (hasDefault) {
			output << bi << "static char* kwlist[] = {"
				<< kwParams.str() << " NULL };\n"
				<< bi << "if (!PyArg_ParseTupleAndKeywords(_args, _keywds, \""
				<< parseFormat.str() << "\", kwlist"
				<< parseParams.str() << "))\n"
				<< bi << "\treturn " << error_return << ";\n";
		} else if (numInArgs == 0) {
			output << bi << "if (PyTuple_GET_SIZE(_args) != 0\n"
				<< bi << "|| (_keywds != NULL && PyDict_Size(_keywds) != 0)) {\n"
				<< bi << "\tPyErr_SetString(PyExc_TypeError, \""
					<< pyname
					<< "() expected no arguments\");\n"
				<< bi << "\treturn " << error_return << ";\n"
				<< bi << "}\n";
		} else {
			if (numInArgs < 0)
				numInArgs = -numInArgs;
			output << bi << "if (_keywds != NULL && PyDict_Size(_keywds) != 0) {\n"
				<< bi << "\tPyErr_SetString(PyExc_TypeError, \""
					<< pyname <<
					"() expected no keyword arguments\");\n"
				<< bi << "\treturn " << error_return << ";\n"
				<< bi << "}\n"
				<< bi << "if (!PyArg_ParseTuple(_args, \""
					<< parseFormat.str() << "\""
					<< parseParams.str() << "))\n"
				<< bi << "\treturn " << error_return << ";\n";
		}
	}
	if (ci != NULL && isStatic)
		pi->ml_flags += " | METH_STATIC";

	argCount = 0;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {
		// handle default arguments that are PyObject's
		argCount += 1;
		if (!i->in || i->defValue.empty())
			continue;
		string ptArg = "_ptArg";
		if (numInArgs != 1 || !single)
			ptArg += itostr(argCount);
		string type = i->type;
		if (i->out && *type.rbegin() == '*') {
			// strip trailing *
			type.erase(type.size() - 1);
		}
		CvtType arg(fd->scope, type, i->null_ok);
		if (!(arg.aptType() == "O"))
			continue;
		output << bi << "if (" << ptArg << " == NULL)\n"
			<< bi << "\t" << ptArg << " =  "
				<< make_buildvalue(arg.bvFormat(), 
					qualify(fd->scope,
							arg.bvArg(i->defValue)))
				<< ";\n";
	}
	pi->toCppArgs = toCppArgs.str();

	if (mtype == Attribute)
		return;
	pi->doc += ')';
	if (!returnProto.empty())
		pi->doc += " -> " + returnProto;
	else if (mtype == Constructor)
		pi->doc += " -> " + fd->tag.str();
}

void
dumpConstructorBody(std::ostream &output, int indent, bool single,
		const ClassInfo *ci, const FuncDecl *fd, string pyname,
		int skipTypeCheck, const string &error_return,
		/*OUT*/ MethodInfo *mi)
{
	string bi(tab(indent));		// base indent
	ParseInfo pi;
	dumpParseArgs(output, indent, single, fd, pyname, Constructor, ci,
						skipTypeCheck, "-1", &pi);
	mi->ml_flags = pi.ml_flags;
	mi->doc = pi.doc;
	string qualName = qualify(ci->cd->scope, ci->name.str());
	
	output << bi << "try {\n";
	output << pi.toCppArgs;
	bi = tab(indent + 1);
	if (fd->allowThreads)
		output << bi << "Py_BEGIN_ALLOW_THREADS\n";
	if (ci->isEmbedded)
		output
			<< bi << "new (&self->_inst_data) " << qualName
					<< "(" << pi.funcParams << ");\n"
			<< bi << "self->_initialized = true;\n";
	else {
		output 
			<< bi << "self->_inst_data = new " << qualName << '('
					<< pi.funcParams << ");\n";
	}
	output
		<< bi << "if (" << module << "Debug >= 6)\n"
		<< bi << "\tstd::cerr << \"Allocate " << ci->name
					<< ": \" << self << '\\n';\n";
	if (ci->isWrappySubclass)
		output
			<< bi << "self->_inst_data->wpyAssociate(self);\n"
			<< bi << "self->_inst_data->setPyOwned();\n";
	output << pi.cleanup;
	bi = tab(indent);
	output << bi << "} catch (...) {\n";
	if (fd->allowThreads)
		output	<< bi << "\tPy_BLOCK_THREADS\n";
	output << pi.cleanup;
	output	<< bi << "\t" << module << "Error();\n"
		<< bi << "\treturn " << error_return << ";\n"
		<< bi << "}\n";
	if (fd->allowThreads)
		output << bi << "Py_END_ALLOW_THREADS\n";
}

void
dumpMethodBody(std::ostream &output, int indent, bool single,
		const ClassInfo *ci, const FuncDecl *fd, string pyname,
		MethodType mtype, int skipTypeCheck, string error_return,
		/*OUT*/ MethodInfo *mi)
{
	string bi(tab(indent));		// base indent
	ParseInfo pi;
	dumpParseArgs(output, indent, single, fd, pyname, mtype, ci,
					skipTypeCheck, error_return, &pi);
	mi->ml_flags = pi.ml_flags;
	mi->doc = pi.doc;
	string returnType;
	bool isStatic = fd->returnType.compare(0, 7, "static ") == 0;
	if (isStatic)
		returnType = fd->returnType.substr(7);
	else
		returnType = fd->returnType;
	output << bi << "try {\n";
	output << pi.toCppArgs;
	bi = tab(indent + 1);
	if (fd->allowThreads)
		output << bi << "Py_BEGIN_ALLOW_THREADS\n";
	if (ci != NULL && !isStatic) {
		string qualName
			= qualify(ci->cd->scope, ci->name.str());
		output <<
			bi << qualName << "* _inst = getInst(self);\n";
	}
	output << bi;
	if (!(returnType == "void"))
		output << qualify(fd->scope, returnType) << " _result = ";
	if (ci != NULL && !isStatic)
		// class function
		output << "_inst->" << fd->tag;
	else if (ci != NULL)
		// class static function
		output << qualify(ci->cd->scope, ci->name.str()) << "::"
								<< fd->tag;
	else
		// module function
		output << qualify(fd->scope, fd->tag.str());
	output << '(' << pi.funcParams << ");\n";
	if (fd->allowThreads)
		output << bi << "Py_BLOCK_THREADS\n";
	output << pi.cleanup;
#ifdef ATTR_CACHE
	if (!pi.cacheReturn.empty())
		output << bi << "// cache attribute\n"
			<< bi << "static PyObject* ca;\n"
			<< bi << "if (ca == NULL)\n"
			<< bi << "\tca = PyUnicode_InternFromString(\"__cached_"
						<< pi.cacheReturn << "__\");\n"
			<< bi << "PyObject* pyResult = ";
	else
#endif
		output << bi << "return ";
	switch (mtype) {
	  default:
		output << make_buildvalue(pi.returnFormat, pi.returnParams) << ";\n";
		break;
	  case Inquiry:
		// TODO: if returnType is void then return output parameter
		output << "_result;\n";
		break;
	}
#ifdef ATTR_CACHE
	if (!pi.cacheReturn.empty()) {
		output << bi << "PyDict_SetItem(self->_inst_dict, ca, pyResult);\n"
			<< bi << "return pyResult;\n";
	}
#endif
	bi = tab(indent);
	output << bi << "} catch (...) {\n";
	if (fd->allowThreads)
		output	<< bi << "\tPy_BLOCK_THREADS\n";
	output << pi.cleanup;
	output	<< bi << "\t" << module << "Error();\n"
		<< bi << "}\n";
	if (fd->allowThreads)
		output << bi << "Py_END_ALLOW_THREADS\n";
}

typedef std::multimap<int, const FuncDecl *> FMethods;

bool
sortMethods(FMethods *methods, MethodMap::const_iterator start,
						MethodMap::const_iterator stop)
{
	// return true if all methods have the same number of arguments.
	int count = -1;
	// sort methods by number of arguments
	for (MethodMap::const_iterator i = start; i != stop; ++i) {
		const FuncDecl *fd = i->second;
		int numInputArgs = 0;
		for (ArgList::const_iterator j = fd->args.begin();
						j != fd->args.end(); ++j) {
			if (!j->defValue.empty())
				break;
			if (j->in)
				++numInputArgs;
		}
		if (count == -1)
			count = numInputArgs;
		else if (count > -1 && count != numInputArgs)
			count = -2;
		methods->insert(FMethods::value_type(numInputArgs, fd));
	}
	return count != -2;
}

typedef std::map<const FuncDecl *, ArgList::size_type> UniArgs;

void
uniqueArgs(UniArgs *uniargs, FMethods::iterator first, FMethods::iterator last)
{
	for (FMethods::iterator i = first; i != last; ++i) {
		const FuncDecl *fd = i->second;
		ArgList::size_type diffArgPos = ~0;
		for (ArgList::size_type j = 0; j != fd->args.size(); ++j) {
			// TODO: only consider input arguments
			// (this means arguments in different functions
			// could match different positions!)
			const Argument &a = fd->args[j];
			string type(a.type);
			if (a.out && *type.rbegin() == '*') {
				// TODO: handle output arrays
				// strip trailing *
				type.erase(type.size() - 1);
			}
			CvtType arg(fd->scope, type, a.null_ok);
			// is this arg different from all the other args?
			for (FMethods::iterator k = first; k != last; ++k) {
				if (k == i)
					continue;
				const FuncDecl *fd2 = k->second;
				if (j >= fd2->args.size())
					goto try_again;
				const Argument &a2 = fd2->args[j];
				string type2(a2.type);
				if (a2.out && *type2.rbegin() == '*') {
					// TODO: handle output arrays
					// strip trailing *
					type2.erase(type2.size() - 1);
				}
				CvtType arg2(fd2->scope, type2, a2.null_ok);
				if (arg.typeCheck("?") == arg2.typeCheck("?"))
					goto try_again;
			}
			diffArgPos = j;
			goto found;
		try_again:;
		}
		std::cerr << "unable to discern different methods: "
				<< qualify(fd->scope, fd->tag.str()) << '\n';
	found:
		(*uniargs)[fd] = diffArgPos;
	}
}

void
dumpMethod(std::ostream &output, const ClassInfo *ci,
	MethodMap::const_iterator start, MethodMap::const_iterator stop,
	/*OUT*/ MethodInfo *mi, string fname, string pyname, MethodType mtype)
{
	bool isConstructor = ci != NULL && ci->name == start->first;
	if (isConstructor)
		mtype = Constructor;
	bool isStatic = start->second->returnType.compare(0, 7, "static ") == 0;
	if (fname.empty()) {
		if (ci == NULL)
			fname = module + '_' + start->first.str();
		else
			fname = moduleName(ci, start->first.str());
	}
	if (pyname.empty())
		pyname = start->first.str();
	string self;
	if (ci != NULL && !isStatic)
		self = " _obj";
	string error_return;
	switch (mtype) {
	  default:
		error_return = "NULL";
		break;
	  case Constructor:
	  case Inquiry:
		error_return = "-1";
		break;
	}

	std::ostringstream body;
	int maxInputArgs = 0;
	for (MethodMap::const_iterator i = start; i != stop; ++i) {
		const FuncDecl &fd = *i->second;
		int numInputArgs = 0;
		for (ArgList::const_iterator j = fd.args.begin();
							j != fd.args.end(); ++j)
			if (j->in)
				++numInputArgs;
		if (numInputArgs > maxInputArgs)
			maxInputArgs = numInputArgs;
	}
	if (std::distance(start, stop) == 1) {
		switch (mtype) {
		  default:
			break;
		  case Constructor:
			dumpConstructorBody(body, 1, false, ci, start->second,
						pyname, 0, error_return, mi);
			break;
		  case Normal:
		  case Attribute:
		  case Inquiry:
			dumpMethodBody(body, 1, true, ci, start->second, pyname,
						mtype, 0, error_return, mi);
			break;
		  case TernaryFunc:
			dumpMethodBody(body, 1, false, ci, start->second,
					pyname, mtype, 0, error_return, mi);
		}
	} else {
		bool single = false;
		if (mtype == Attribute) {
			mi->ml_flags = "METH_O";
			single = true;	// pretend for Attributes
		} else {
			mi->ml_flags = "METH_VARARGS | METH_KEYWORDS";
			if (isStatic)
				mi->ml_flags += " | METH_STATIC";
		}
		// sort by number of arguments and handle multiple
		// versions with same number of arguments
		FMethods methods;
		bool needSwitch = !sortMethods(&methods, start, stop);
		if (needSwitch) {
			if (mtype == Attribute) {
				single = false;
				body << "\tPyObject *_args;\n"
					"\tif (PyTuple_Check(_ptArg)) {\n"
					"\t\t_args = _ptArg;\n"
					"\t} else {\n"
					"\t\t// LEAK: need C++ PyObject holder\n"
					"\t\tstatic PyObject* _myargs = NULL;\n"
					"\t\tif (_myargs == NULL)\n"
					"\t\t\t_myargs = PyTuple_New(1);\n"
					"\t\tPyTuple_SET_ITEM(_myargs, 0, _ptArg);\n"
					"\t\t_args = _myargs;\n"
					"\t}\n";
			}
			int curArgs = methods.begin()->first;
			body <<
				"\tswitch (PyTuple_Size(_args)) {\n"
				"\t  default:\n"
				"\t\tPyErr_SetString(PyExc_TypeError, \""
					<< pyname << "() expected " << curArgs;
			int maxArgs = methods.rbegin()->first;
			for (FMethods::iterator i = methods.begin();
						i != methods.end(); ++i) {
				int numArgs = i->first;
				if (numArgs == curArgs)
					continue;
				if (numArgs == maxArgs) {
					body << " or " << maxArgs;
					break;
				}
				body << ", " << numArgs;
				curArgs = numArgs;
			}
			body << " arguments\");\n"
				<< "\t\treturn " << error_return << ";\n";
		}
		string bi(tab(1));		// base indent
		if (needSwitch)
			bi += '\t';
		for (FMethods::iterator i = methods.begin();
							i != methods.end(); ) {
			size_t numArgs = i->first;
			FMethods::iterator next = methods.upper_bound(numArgs);
			if (needSwitch)
				body << "\t  case " << numArgs << ": {\n";
			FMethods::size_type count = std::distance(i, next);
			UniArgs uniargs;
			if (count > 1)
				uniqueArgs(&uniargs, i, next);
			for (FMethods::iterator j = i; j != next; ++j) {
				const FuncDecl *fd = j->second;
				int indent = needSwitch ? 2 : 1;
				int skipTypeCheck = 0;
				if (count > 1) {
					// discern this case
					ArgList::size_type argNum = uniargs[fd];
					if (argNum == ~0u)
						continue;
					skipTypeCheck = argNum + 1;
					const Argument &a = fd->args[argNum];
					string type(a.type);
					if (a.out && *type.rbegin() == '*') {
						// TODO: handle output arrays
						// strip trailing *
						type.erase(type.size() - 1);
					}
					CvtType arg(fd->scope, type, a.null_ok);
					body << bi;
					if (j != i)
						body << "} else ";
					body << "if (";
					if (single) {
						body << arg.typeCheck("_ptArg");
					} else {
						if (!needSwitch
						|| argNum >= numArgs)
							body << "PyTuple_Size(_args) > " << argNum << "\n" << bi << "&& ";
						body << '(' << arg.typeCheck("PyTuple_GetItem(_args, " + itostr(argNum) + ")") << ")";
					}
					body << ") {\n";
					indent = needSwitch ? 3 : 2;
				}
				MethodInfo miTmp;
				switch (mtype) {
				  default:
					break;
				  case Constructor:
					dumpConstructorBody(body, indent,
						single, ci, fd, pyname,
						skipTypeCheck, error_return,
						&miTmp);
					break;
				  case Normal:
				  case Attribute:
				  case TernaryFunc:
					dumpMethodBody(body, indent, single, ci,
						fd, pyname, mtype,
						skipTypeCheck, error_return,
						&miTmp);
					break;
				}
				if (!mi->doc.empty())
					mi->doc += '\n';
				mi->doc += miTmp.doc;
			}
			if (count > 1) {
				body << bi << "} else {\n"
					<< bi <<
					"\tPyErr_SetString(PyExc_TypeError, "
						"\"argument type mismatch\");\n"
					<< bi << "\treturn " << error_return << ";\n"
					<< bi << "}\n";
			}
			if (needSwitch) {
				body << bi << "break;\n"
					"\t  }\n";
			}
			i = next;
		}
		if (needSwitch)
			body <<
				"\t}\n";
	}

	switch (mtype) {
	  default:
		break;
	  case Normal:
	  case Attribute:
	  case TernaryFunc:
		output <<
			"\n"
			"static PyObject*\n" <<
			fname << "(PyObject*" << self;
		if (mi->ml_flags == "METH_NOARGS"
		|| mi->ml_flags == "METH_NOARGS | METH_STATIC")
			output << ", PyObject* /* = NULL */)\n";
		else if (mi->ml_flags == "METH_O"
		|| mi->ml_flags == "METH_O | METH_STATIC")
			output << ", PyObject* _ptArg)\n";
		else if (mi->ml_flags == "METH_VARARGS"
		|| mi->ml_flags == "METH_VARARGS | METH_STATIC")
			output << ", PyObject* _args)\n";
		else
			output << ", PyObject* _args, PyObject* _keywds)\n";
		break;
	  case Constructor:
		output <<
			"\n"
			"static int\n" <<
			fname << "(PyObject* _self, PyObject* _args, PyObject* _keywds)\n";
		break;
	  case Inquiry:
		output <<
			"\n"
			"static Py_ssize_t\n" <<
			fname << "(PyObject* _obj)\n";
		break;
	}
	output
		<< "{\n";
	if (ci != NULL) {
		string objectName(moduleName(ci, "object"));
		if (mtype == Constructor) {
			output << "\t" << objectName << "* self = static_cast<"
					<< objectName << "*>(_self);\n";
			if (ci->isEmbedded)
				output << "\tif (self->_initialized) {\n";
			else
				output << "\tif (self->_inst_data) {\n";
			output << "\t\tPyErr_SetString(PyExc_ValueError, "
				"\"underlying C++ " << pythonName(ci)
				<< " object is already initialized\");\n"
				<< "\t\treturn " << error_return << ";\n"
				<< "\t}\n"
				<< "\n";
		} else if (!isStatic) {
			output << "\t" << objectName
				<< "* self = static_cast<" << objectName
				<< "*>(_obj);\n"
				<< "\n";
		}
	}
	output << body.str();

	if (mtype == Constructor) {
		if (!ci->isFinal)
			output <<
				"\tself->_inst_dict = PyDict_New();\n"
				"\tif (self->_inst_dict == NULL) {\n"
				"\t\treturn -1;\n"
				"\t}\n";
		dumpTypeAttrInit(output, 1, ci);
		output <<
			"\treturn 0;\n";
	}

	output << "\treturn " << error_return << ";\n"
		"}\n";
}

struct OpInfo {
	CvtType args[3];	// up to 3 arguments
	unsigned argPos[3];
	const FuncDecl *fd;
	OpInfo(): fd(0) { argPos[0] = argPos[1] = argPos[2] = 0; }
	bool operator<(const OpInfo &r) const
	{
		return this->args[0].pythonType() < r.args[0].pythonType();
	}
};

typedef std::vector<OpInfo> Operators;

void
sortOperators(Operators *ops, const MethodMap &mm, unsigned numArgs)
{
	ops->reserve(mm.size());
	for (MethodMap::const_iterator i = mm.begin(); i != mm.end(); ++i) {
		const FuncDecl *fd = i->second;
		OpInfo opi;
		opi.fd = fd;
		size_t arg = 0;
		if (!fd->notMemberFunction) {
			assert(fd->scope->dtype == Decl::CLASS
					|| fd->scope->dtype == Decl::STRUCT);
			const ClassDecl *cd = static_cast<const ClassDecl *>
								(fd->scope);
			opi.args[0] = CvtType(cd->scope, cd->name.str() + '*');
			opi.argPos[0] = fd->args.size() + 10;
			++arg;
		}

		for (ArgList::size_type j = 0; j != fd->args.size(); ++j) {
			const Argument &a = fd->args[j];
			if (!a.in)
				continue;
			if (arg == numArgs) {
				std::cerr << "too many arguments to operator: "
					<< qualify(fd->scope, fd->tag.str())
					<< " (" << numArgs << ")\n";
				exit(1);
			}

			string type(a.type);
			if (a.out && *type.rbegin() == '*') {
				// TODO: handle output arrays
				// strip trailing *
				type.erase(type.size() - 1);
			}
			opi.args[arg] = CvtType(fd->scope, type, a.null_ok);
			if (opi.args[arg].pythonType() == "__unknown__")
				goto skip;
			opi.argPos[arg] = j;
			++arg;
		}
		if (arg < numArgs) {
			// assume return type is an arg
			string type(fd->returnType);
			// need to strip &, otherwise we'll often try to
			// create a reference to a rvalue, which is currently
			// not allowed.
			if (*type.rbegin() == '&') {
				// strip trailing &
				type.erase(type.size() - 1);
			}
			opi.args[arg] = CvtType(fd->scope, type);
			opi.argPos[arg] = fd->args.size();
			++arg;
			assert(arg == numArgs);
		}
		ops->push_back(opi);
skip:;
	}
	std::stable_sort(ops->begin(), ops->end());
}

static const char *ArgName[] = { "v", "w", "z" };
static const char *ArgPos[] = { "first", "second", "third" };

void
dumpOperatorBody(std::ostream &output, MethodType mtype, const OpInfo &opi,
	unsigned numArgs, int indent, unsigned skipCheck, const string &pyname,
	const string &error_return)
{
	string bi(tab(indent));		// base indent
	const FuncDecl *fd = opi.fd;
	if (numArgs > 1) {
		// if there is only one argument, then we know the type
		for (unsigned i = 0; i < numArgs; ++i) {
			if (i < skipCheck)
				continue;
			if ((mtype == SsizeArgFunc || mtype == SsizeObjArgProc)
			&& i == 1)
				continue;
			const CvtType &arg = opi.args[i];
			output
				<< bi << "if (!"
					<< arg.typeCheck(ArgName[i]) << ") {\n"
				<< bi << "\tPyErr_SetString(PyExc_TypeError, \""
					<< ArgPos[i] << " argument should be a "
					<< arg.pythonType() << "\");\n"
				<< bi << "return " << error_return << ";\n"
				<< bi << "}\n";
		}
	}
	for (unsigned i = 0; i < numArgs; ++i) {
		const CvtType &arg = opi.args[i];
		output <<
			bi << qualify(fd->scope, arg.cppTypeDecl(
					"cppArg" + itostr(opi.argPos[i])))
				<< " = ";
		if ((mtype == SsizeArgFunc || mtype == SsizeObjArgProc) && i == 1)
			output << ArgName[i] << ";\n";
		else
			output <<
				qualify(fd->scope, arg.pyToCpp(ArgName[i])) <<
				";\n";
	}

	string returnType;
	bool isStatic = fd->returnType.compare(0, 7, "static ") == 0;
	if (isStatic)
		returnType = fd->returnType.substr(7);
	else
		returnType = fd->returnType;

	string returnFormat;
	string returnParams;
	string funcParams;
	if (!returnType.empty()) {
		if (returnType != "void") {
			CvtType rt(fd->scope, returnType);
			returnFormat = rt.bvFormat();
			returnParams = ", "
				+ qualify(fd->scope, rt.bvArg("_result"));
		} else if (pyname.compare(0, 3, "__i") == 0
		&& pyname != "__int__") {
			returnFormat = 'O';
			returnParams = string(", ") + ArgName[0];
		}
	}
	for (ArgList::size_type j = 0; j != fd->args.size(); ++j) {
		const Argument &a = fd->args[j];
		if (!funcParams.empty())
			funcParams += ", ";
		if (a.out)
			funcParams += '&'; // out params need leading &
		funcParams += "cppArg" + itostr(j);
		if (!a.out)
			continue;
		string type = a.type;
		if (*type.rbegin() == '*') {
			// TODO: handle output arrays too
			// strip trailing *
			type.erase(type.size() - 1);
		}
		CvtType arg(fd->scope, type, a.null_ok);
		returnFormat += arg.bvFormat();
		returnParams += ", " + qualify(fd->scope,
				arg.bvArg("cppArg" + itostr(j)));
	}
	output << bi << "try {\n";
	bi = tab(indent + 1);
	if (fd->allowThreads)
		output << bi << "Py_BEGIN_ALLOW_THREADS\n";
	output << bi;
	if (mtype != SsizeObjArgProc && mtype != ObjObjArgProc
	&& !(returnType == "void"))
		output << qualify(fd->scope, returnType) << " _result = ";
	if (fd->notMemberFunction)
		output << qualify(fd->scope, fd->tag.str());
	else if (fd->reverseOp)
		output << "cppArg" << opi.argPos[1] << "->" << fd->tag;
	else
		output << "cppArg" << opi.argPos[0] << "->" << fd->tag;
	output << '(' << funcParams << ")";
	if (mtype == SsizeObjArgProc || mtype == ObjObjArgProc) {
		output << " = cppArg" << fd->args.size();
	}
	output << ";\n";
	if (fd->allowThreads)
		output << bi << "Py_BLOCK_THREADS\n";
	switch (mtype) {
	  default:
		output << bi << "return "
			<< make_buildvalue(returnFormat, returnParams) << ";\n";
		break;
	  case ObjObjProc:
		// TODO: if returnType is void then return output parameter
		output << bi << "return _result;\n";
		break;
	  case SsizeObjArgProc:
	  case ObjObjArgProc:
		output << bi << "return 0;\n";
		break;
	}
	bi = tab(indent);
	output << bi << "} catch (...) {\n";
	if (fd->allowThreads)
		output	<< bi << "\tPy_BLOCK_THREADS\n";
	output	<< bi << "\t" << module << "Error();\n"
		<< bi << "\treturn " << error_return << ";\n"
		<< bi << "}\n";
	if (fd->allowThreads)
		output << bi << "\tPy_END_ALLOW_THREADS\n";
}

bool
dumpOperator(std::ostream &output, const MethodMap &mm,
		const std::string &fname, std::string pyname, MethodType mtype)
{
	unsigned numArgs = 0;
	switch (mtype) {
	  default:
		std::cerr << "unknown operator type\n";
		abort();
	  case UnaryFunc: numArgs = 1; break;
	  case BinaryFunc: numArgs = 2; break;
	  case TernaryFunc: numArgs = 3; break;
	  case SsizeArgFunc: numArgs = 2; break;
	  case ObjObjProc: numArgs = 2; break;
	  case SsizeObjArgProc: numArgs = 3; break;
	  case ObjObjArgProc: numArgs = 3; break;
	}
	Operators ops;
	sortOperators(&ops, mm, numArgs);
	if (ops.empty())
		return false;

	string error_return;
	switch (mtype) {
	  default:
		break;
	  case UnaryFunc:
	  case BinaryFunc:
	  case TernaryFunc:
	  case SsizeArgFunc:
		error_return = "NULL";
		break;
	  case ObjObjProc:
	  case SsizeObjArgProc:
	  case ObjObjArgProc:
		error_return = "-1";
		break;
	}

	if (pyname.empty())
		pyname = fname;
	output <<
		"\n";
	switch (mtype) {
	  default:
		break;
	  case UnaryFunc:
		output <<
			"static PyObject*\n" <<
			fname << "(PyObject* v)\n";
		break;
	  case BinaryFunc:
		output <<
			"static PyObject*\n" <<
			fname << "(PyObject* v, PyObject* w)\n";
		break;
	  case TernaryFunc:
		output <<
			"static PyObject*\n" <<
			fname << "(PyObject* v, PyObject* w, PyObject* z)\n";
		break;
	  case SsizeArgFunc:
		output <<
			"static PyObject*\n" <<
			fname << "(PyObject* v, Py_ssize_t w)\n";
		break;
	  case ObjObjProc:
		output <<
			"static int\n" <<
			fname << "(PyObject* v, PyObject* w)\n";
		break;
	  case SsizeObjArgProc:
		output <<
			"static int\n" <<
			fname << "(PyObject* v, Py_ssize_t w, PyObject* z)\n";
		break;
	  case ObjObjArgProc:
		output <<
			"static int\n" <<
			fname << "(PyObject* v, PyObject* w, PyObject* z)\n";
		break;
	}
	output <<
		"{\n";
	switch (mtype) {
	  default:
		break;
	  case SsizeObjArgProc:
	  case SsizeSsizeObjArgProc:
	  case ObjObjArgProc:
		output <<
			"\tif (" << ArgName[numArgs - 1] << " == NULL) {\n"
			"\t\tPyErr_SetString(PyExc_ValueError, "
				"\"item deletion is not supported\");\n"
			"\t\treturn -1;\n"
			"\t}\n";
		break;
	}

	// all "methods" have the same number of arguments
	if (ops.size() == 1) {
		OpInfo &opi = ops[0];
		dumpOperatorBody(output, mtype, opi, numArgs, 1, 1, error_return, pyname);
	} else {
		// 2 or more functions that we have to differeniate between
		// (for now, assume BinaryFunc (can't be UnaryFunc))
		// TODO: TernaryFunc
		CvtType *save0 = NULL;
		CvtType *save1 = NULL;
		for (Operators::iterator i = ops.begin(); i != ops.end(); ++i) {
			OpInfo &opi = *i;
			if (save0 == NULL
			|| opi.args[0].pythonType() != save0->pythonType()) {
				output << "\t";
				if (save0 != NULL)
					output << "\t\t}\n"
						"\t} else ";
				save0 = &opi.args[0];
				save1 = NULL;
				output <<
					"if (" <<
						save0->typeCheck(ArgName[0])
						<< ") {\n";
			}
			if (save1 == NULL
			|| opi.args[1].pythonType() != save1->pythonType()) {
				output << "\t\t";
				if (save1 != NULL)
					output << "} else ";
				save1 = &opi.args[1];
				output <<
					"if (" <<
						save1->typeCheck(ArgName[1])
						<< ") {\n";
			}
			dumpOperatorBody(output, mtype, opi, numArgs, 3, 2, error_return, pyname);
		}
		output <<
			"\t\t}\n"
			"\t}\n"
			"\tPyErr_SetString(PyExc_TypeError, "
					"\"argument type mismatch\");\n";
	}

	output << "\treturn " << error_return << ";\n"
		"}\n";
	return true;
}
