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

// $Id: compute.cpp 37794 2012-11-26 23:32:55Z gregc $

#include <set>
#include <algorithm>
#include <stdarg.h>
#include "Reg.h"
#include "common.h"
#include "compute.h"
#include "CvtType.h"
#ifdef OTF_NO_LOCALE
# include <ctype.h>
#endif

using std::string;

DeclList	globalDecls;
NamespaceDecl	*moduleScope = NULL;

DerivedClasses	derivedClasses;
SymCDMap	classDeclsMap;
CDCIMap		classInfoMap;

static Reg	btStatic("static\\b[[:space:]]*", Reg::EXTENDED);

static void
computeDerivedClasses(const ClassDecl *cd)
{
	Symbol derived(qualify(cd->scope, cd->name.str()));
	for (DeclList::const_iterator i = cd->baseClasses.begin();
					i != cd->baseClasses.end(); ++i) {
		Symbol base(qualify(cd->scope, (*i)->tag.str()));
		derivedClasses.insert(std::make_pair(base, derived));
	}
}

static Reg containsClass("\\bWrapPyObj\\b");

static bool
isWrapPySubclass(const ClassDecl *cd)
{
	// A class is wrapped if it is subclassed from WrapPyObj
	// This only return true if it is directly subclassed.
	// The ClassDecl baseClasses includes classes that don't have
	// ClassInfo associated with them.
	for (DeclList::const_iterator i = cd->baseClasses.begin();
					i != cd->baseClasses.end(); ++i) {
		if (string::npos != containsClass.find((*i)->text))
			return true;
	}
	return false;
}

// A "get" attribute function is a function that has a non-void return
// value, possibily through output arguments, and has no input arguments.

inline const FuncDecl *
findGetFunc(const ClassDecl *cd, Symbol fname)
{
	if (cd == NULL)
		return NULL;
	const Decl *d = cd->members.unique(fname);
	if (d == NULL)
		return NULL;
	const FuncDecl *fd = dynamic_cast<const FuncDecl *>(d);
	if (fd == NULL
	|| (fd->args.size() == 0 && fd->returnType == "void"))
		return NULL;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i)
		if (i->in)
			return NULL;
	return fd;
}

static const FuncDecl *
findGetFunc(const ClassInfo *ci, Symbol fname)
{
	if (ci == NULL)
		return NULL;
	const FuncDecl *result = findGetFunc(ci->cd, fname);
	if (result != NULL)
		return result;
	// recurse on base classes
	for (CIList::const_iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
		const ClassInfo *bci = *i;
		if (ci->isWrappySubclass && bci->isWrappySubclass)
			continue;
		result = findGetFunc(bci, fname);
		if (result != NULL)
			return result;
	}
	return NULL;
}

// A "set" attribute function is a function that does not return a value,
// not even through output arguments, and has at least one input argument.

inline const FuncDecl *
findSetFunc(const DeclList::RAConstRange &range)
{
	for (DeclList::RAconst_iterator i = range.first; i != range.second;
									++i) {
		const FuncDecl *fd
				= dynamic_cast<const FuncDecl *>(i->second);
		if (fd == NULL || !(fd->returnType == "void")
		|| fd->args.size() == 0)
			continue;
		for (ArgList::const_iterator i = fd->args.begin();
						i != fd->args.end(); ++i)
			if (i->out)
				goto tryNext;
		return fd;
	tryNext:
		;
	}
	return NULL;
}

static const FuncDecl *
findSetFunc(const ClassInfo *ci, Symbol fname)
{
	if (ci == NULL)
		return NULL;
	DeclList::RAConstRange range = ci->cd->members.rAequal_range(fname);
	const FuncDecl *result = findSetFunc(range);
	if (result != NULL)
		return result;
	// recurse on base classes
	for (CIList::const_iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
		const ClassInfo *bci = *i;
		if (ci->isWrappySubclass && bci->isWrappySubclass)
			continue;
		result = findSetFunc(bci, fname);
		if (result != NULL)
			return result;
	}
	return NULL;
}

typedef std::set<const Decl *> DeclSet;

inline int
markVisited(const ClassDecl *cd, Symbol fname, DeclSet *visited, MethodMap *mm,
							Symbol attrName)
{
	int count = 0;
	DeclList::RAConstRange range = cd->members.rAequal_range(fname);
	for (DeclList::RAconst_iterator i = range.first; i != range.second;
									++i) {
		++count;
		(*visited).insert(i->second);
		if (mm == NULL)
			continue;
		const FuncDecl *fd = dynamic_cast<const FuncDecl *>(i->second);
		if (fd == NULL)
			continue;
		// assert(fd->tag == fname);
		fd->attrName = attrName;
		mm->insert(MethodMap::value_type(fd->tag, fd));
	}
	return count;
}

static int
markVisited(const ClassInfo *ci, Symbol fname, DeclSet *visited, MethodMap *mm,
							Symbol attrName)
{
	int count = 0;
	count += markVisited(ci->cd, fname, visited, mm, attrName);
	// recurse on base classes
	for (CIList::const_iterator i = ci->baseClasses.begin();
						i != ci->baseClasses.end(); ++i)
		count += markVisited(*i, fname, visited, mm, attrName);
	return count;
}

// all "count" extra arguments must be of type "const char*"

static bool
fmatch(const FuncDecl *fd, const char *rtype, string::size_type count, ...)
{
	if (fd == NULL)
		return false;
	if (rtype != NULL)
		if (!(fd->returnType == rtype)
		&& !(fd->returnType == string("inline ") + rtype))
			return false;
	if (fd->args.size() != count)
		return false;

	va_list argp;
	va_start(argp, count);
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {
		const char *type = va_arg(argp, const char *);
		if (type == NULL)
			continue;	// don't care arg
		if (!(i->type == type)) {
			va_end(argp);
			return false;
		}
	}
	va_end(argp);
	return true;
}

static void
intArgOnly(MethodMap *mm)
{
	// strip out all methods that don't have a single "int" argument
	for (MethodMap::iterator i = mm->begin(); i != mm->end();) {
		const FuncDecl *fd = i->second;
		size_t arg = fd->notMemberFunction ? 1 : 0;
		if (fd->args.size() > arg) {
			const Argument &a = fd->args[arg];
			if (a.in) {
				CvtType type(CvtType(fd->scope, a.type));
				const string& pt = type.pythonType();
				if (pt == "int"
				|| pt == "unsigned"
				|| pt == "unsigned int"
				|| pt == "long"
				|| pt == "unsigned long") {
					++i;
					continue;
				}
			}
		}
		mm->erase(i++);
	}
}

static string
pythonArgs(const FuncDecl *fd)
{
	string result;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {
		const Argument &a = *i;
		if (!a.in)
			continue;
		string type = a.type;
		if (a.out && *type.rbegin() == '*')
			type.erase(type.size() - 1);	// remove trailing *
		CvtType arg(fd->scope, type, a.null_ok);
		if (!result.empty())
			result += ", ";
		result += arg.pythonType();
	}
	return result;
}

static void
separateMethods(MethodMap *get, MethodMap *set)
{
	// Remove all but one method with equivalent Python arguments
	// (prefer the non-const version over the const version)
	typedef std::map<string, MethodMap::iterator> Info;
	Info info;
	for (MethodMap::iterator i = get->begin(); i != get->end(); ++i) {
		const FuncDecl *fd = i->second;
		string args = pythonArgs(fd); 
		Info::iterator j = info.find(args);
		if (j != info.end() && fd->isConst)
			continue;
		info.insert(Info::value_type(args, i));
	}
	if (get->size() > info.size()) {
		MethodMap g;
		for (Info::iterator i = info.begin(); i != info.end(); ++i) {
			MethodMap::iterator j = i->second;
			g.insert(MethodMap::value_type(j->first, j->second));
		}
		get->swap(g);
	}

	// copy all non-const methods from the get MethodMap into set
	for (MethodMap::iterator i = get->begin(); i != get->end(); ++i) {
		const FuncDecl *fd = i->second;
		if (!fd->isConst)
			set->insert(MethodMap::value_type(i->first, fd));
	}
}

static bool
computeStreamOperator(ClassDecl *cd, const char *opname, const char *stream0,
							const char *stream1)
{
	// destructively search for stream operator??(stream, type)
	Symbol op(string("operator") + opname);
	string pattern("\\b");
	pattern += cd->name.str() + "\\b";
	Reg containsClass(pattern.c_str());
	DeclList *scopeDecls;
	if (cd->scope == NULL)
		scopeDecls = &globalDecls;
	else if (cd->scope->dtype == Decl::NAMESPACE) {
		NamespaceDecl *nd
				= static_cast<NamespaceDecl *>(cd->scope);
		scopeDecls = &nd->decls;
	} else { // dtype == CLASS or STRUCT
		ClassDecl *nd = static_cast<ClassDecl *>(cd->scope);
		scopeDecls = &nd->members;
	}
	DeclList::RARange range = scopeDecls->rAequal_range(op);
	for (DeclList::RAiterator i = range.first; i != range.second;
									++i) {
		FuncDecl *fd = dynamic_cast<FuncDecl *>(i->second);
		if (fd == NULL)
			continue;
		if (!fmatch(fd, stream0, 2, NULL, NULL)
		&& !fmatch(fd, stream1, 2, NULL, NULL))
			continue;
		if (fd->args[0].type != stream0
		&& fd->args[0].type != stream1)
			continue;
		// see if second argument contains class name
		if (containsClass.find(fd->args[1].type) != string::npos) {
			scopeDecls->remove(fd);
			return true;
		}
	}
	return false;
}

static bool
computeOstreamOperator(ClassDecl *cd)
{
	// destructively search for ostream& operator<<(ostream&, type)
	return computeStreamOperator(cd, "<<", "ostream&", "std::ostream&");
}

static bool
computeIstreamOperator(ClassDecl *cd)
{
	// destructively search for istream& operator>>(istream&, type)
	return computeStreamOperator(cd, ">>", "istream&", "std::istream&");
}

static bool
unaryOp(const ClassDecl *cd, const char *unaryop, const char *returnType,
						MethodMap *mm, DeclSet *visited)
{
	// a class unary operator can either be a member function that
	// takes zero arguments or a function that takes one.
	MethodMap tmpMM;
	if (mm == NULL)
		mm = &tmpMM;
	Symbol op(unaryop);
	string pattern("\\b");
	pattern += cd->name.str() + "\\b";
	Reg containsClass(pattern.c_str());
	// first look in class
	DeclList::RAConstRange range = cd->members.rAequal_range(op);
	for (DeclList::RAconst_iterator i = range.first; i != range.second;
									++i) {
		const FuncDecl *fd
				= dynamic_cast<const FuncDecl *>(i->second);
		if (fd == NULL || !fmatch(fd, returnType, 0))
			continue;
		mm->insert(MethodMap::value_type(op, fd));
		if (visited)
			visited->insert(fd);
		return true;	// there can only be one, so stop looking
	}
	// next look at class' scope
	bool mustBeStatic = false;
	const DeclList *scopeDecls;
	if (cd->scope == NULL) {
		scopeDecls = &globalDecls;
	} else if (cd->scope->dtype == Decl::NAMESPACE) {
		const NamespaceDecl *nd
				= static_cast<const NamespaceDecl *>(cd->scope);
		scopeDecls = &nd->decls;
	} else { // dtype == CLASS or STRUCT
		const ClassDecl *nd = static_cast<const ClassDecl *>(cd->scope);
		scopeDecls = &nd->members;
		mustBeStatic = true;
	}
	DeclList::RAConstRange erange = scopeDecls->rAequal_range(op);
	for (DeclList::RAconst_iterator i = erange.first; i != erange.second;
									++i) {
		const FuncDecl *fd
				= dynamic_cast<const FuncDecl *>(i->second);
		if (fd == NULL || !fmatch(fd, returnType, 1, NULL))
			continue;
		if (mustBeStatic && btStatic.find(fd->returnType) != 0)
			continue;
		// see if first arguments contains class name
		if (containsClass.find(fd->args[0].type) == string::npos)
			continue;
		fd->notMemberFunction = true;
		mm->insert(MethodMap::value_type(op, fd));
		if (visited)
			visited->insert(fd);
		return true;	// there can only be one, so stop looking
	}
	return false;
}

static bool
binOp(const ClassDecl *cd, const char *binop, const char *returnType,
						MethodMap *mm, DeclSet *visited)
{
	// a class binary operator can either be a member function that
	// takes one argument or a function that takes two.
	MethodMap tmpMM;
	if (mm == NULL)
		mm = &tmpMM;
	Symbol op(binop);
	string pattern("\\b");
	pattern += cd->name.str() + "\\b";
	Reg containsClass(pattern.c_str());
	DeclList::RAConstRange range = cd->members.rAequal_range(op);
	for (DeclList::RAconst_iterator i = range.first; i != range.second;
									++i) {
		const FuncDecl *fd
				= dynamic_cast<const FuncDecl *>(i->second);
		if (fd && fmatch(fd, returnType, 1, NULL)) {
			mm->insert(MethodMap::value_type(op, fd));
			if (visited)
				visited->insert(fd);
		}
	}
	bool mustBeStatic = false;
	const DeclList *scopeDecls;
	if (cd->scope == NULL)
		scopeDecls = &globalDecls;
	else if (cd->scope->dtype == Decl::NAMESPACE) {
		const NamespaceDecl *nd
				= static_cast<const NamespaceDecl *>(cd->scope);
		scopeDecls = &nd->decls;
	} else { // dtype == CLASS or STRUCT
		const ClassDecl *nd = static_cast<const ClassDecl *>(cd->scope);
		scopeDecls = &nd->members;
		mustBeStatic = true;
	}
	DeclList::RAConstRange erange = scopeDecls->rAequal_range(op);
	for (DeclList::RAconst_iterator i = erange.first; i != erange.second;
									++i) {
		const FuncDecl *fd
				= dynamic_cast<const FuncDecl *>(i->second);
		if (fd == NULL || !fmatch(fd, returnType, 2, NULL, NULL))
			continue;
		if (mustBeStatic && btStatic.find(fd->returnType) != 0)
			continue;
		// see if either argument contains class name
		if (containsClass.find(fd->args[0].type) != string::npos) {
			fd->notMemberFunction = true;
			mm->insert(MethodMap::value_type(op, fd));
			if (visited)
				visited->insert(fd);
		}
		else if (containsClass.find(fd->args[1].type) != string::npos) {
			fd->notMemberFunction = true;
			fd->reverseOp = true;
			mm->insert(MethodMap::value_type(op, fd));
			if (visited)
				visited->insert(fd);
		}
	}
	return !mm->empty();
}

static void
computeBuiltins(ClassInfo *ci, DeclSet *visited)
{
	const FuncDecl *fd;

	// look for type methods
	// print method
	ci->print = computeOstreamOperator(ci->cd);
	// compare method
	binOp(ci->cd, "operator<", "bool", &ci->compare, visited);
	// TODO: repr method
	// hash method
	fd = findGetFunc(ci, Symbol("hash"));
	if (fd) {
		ci->hash.insert(MethodMap::value_type(Symbol("hash"), fd));
		visited->insert(fd);
	}
	// call method
	Symbol op("operator()");
	DeclList::RAConstRange range = ci->cd->members.rAequal_range(op);
	for (DeclList::RAconst_iterator i = range.first; i != range.second; ++i)
		if (i->second->dtype == Decl::FUNCTION) {
			fd = dynamic_cast<FuncDecl *>(i->second);
			ci->call.insert(MethodMap::value_type(op, fd));
			visited->insert(fd);
		}

	// str method
	fd = dynamic_cast<FuncDecl *>(ci->cd->members.unique(Symbol("str")));
	if (fd
	&& (fmatch(fd, "std::string", 0) || fmatch(fd, "const std::string&", 0)
	|| fmatch(fd, "string", 0) || fmatch(fd, "const string&", 0))) {
		ci->str.insert(MethodMap::value_type(Symbol("str"), fd));
		visited->insert(fd);
	} else {
		fd = dynamic_cast<FuncDecl *>
				(ci->cd->members.unique(Symbol("c_str")));
		if (fd
		&& (fmatch(fd, "char*", 0) || fmatch(fd, "const char*", 0))) {
			ci->str.insert(MethodMap::value_type(Symbol("c_str"),
									fd));
			visited->insert(fd);
		}
	}

	binOp(ci->cd, "operator<", 0, &ci->lt, visited);
	binOp(ci->cd, "operator<=", 0, &ci->le, visited);
	binOp(ci->cd, "operator==", 0, &ci->eq, visited);
	binOp(ci->cd, "operator!=", 0, &ci->ne, visited);
	binOp(ci->cd, "operator>", 0, &ci->gt, visited);
	binOp(ci->cd, "operator>=", 0, &ci->ge, visited);

	// look for number methods
	if (ci->hasNumberMethods) {
		binOp(ci->cd, "operator+", 0, &ci->nb_add, visited);
		binOp(ci->cd, "operator-", 0, &ci->nb_subtract, visited);
		binOp(ci->cd, "operator*", 0, &ci->nb_multiply, visited);
		binOp(ci->cd, "operator/", 0, &ci->nb_divide, visited);
		binOp(ci->cd, "operator%", 0, &ci->nb_remainder, visited);
		// TODO: divmod, rdivmod
		// TODO: power, rpower
		unaryOp(ci->cd, "operator-", 0, &ci->nb_negative, visited);
		unaryOp(ci->cd, "operator+", 0, &ci->nb_positive, visited);
		// TODO: absolute
		if (!unaryOp(ci->cd, "operator!", "bool", &ci->nb_nonzero, visited))
			unaryOp(ci->cd, "operator bool", "", &ci->nb_nonzero, visited);
		unaryOp(ci->cd, "operator~", 0, &ci->nb_invert, visited);
		binOp(ci->cd, "operator<<", 0, &ci->nb_lshift, visited);
		binOp(ci->cd, "operator>>", 0, &ci->nb_rshift, visited);
		binOp(ci->cd, "operator&", 0, &ci->nb_and, visited);
		binOp(ci->cd, "operator^", 0, &ci->nb_xor, visited);
		binOp(ci->cd, "operator|", 0, &ci->nb_or, visited);
		unaryOp(ci->cd, "operator int", "", &ci->nb_int, visited);
		unaryOp(ci->cd, "operator long", "", &ci->nb_long, visited);
		unaryOp(ci->cd, "operator float", "", &ci->nb_float, visited);
		// TODO: oct
		// TODO: hex

		// Python 2.0 iadd, etc.
		binOp(ci->cd, "operator+=", 0, &ci->nb_inplace_add, visited);
		binOp(ci->cd, "operator-=", 0, &ci->nb_inplace_subtract, visited);
		binOp(ci->cd, "operator*=", 0, &ci->nb_inplace_multiply, visited);
		binOp(ci->cd, "operator/=", 0, &ci->nb_inplace_divide, visited);
		binOp(ci->cd, "operator%=", 0, &ci->nb_inplace_remainder, visited);
		// TODO: inplace_power
		binOp(ci->cd, "operator<<=", 0, &ci->nb_inplace_lshift, visited);
		binOp(ci->cd, "operator>>=", 0, &ci->nb_inplace_rshift, visited);
		binOp(ci->cd, "operator&=", 0, &ci->nb_inplace_and, visited);
		binOp(ci->cd, "operator^=", 0, &ci->nb_inplace_xor, visited);
		binOp(ci->cd, "operator|=", 0, &ci->nb_inplace_or, visited);

#ifdef TODO
		// TODO: floor divide, true divide, and their inplace versions
		binOp(ci->cd, "?", 0, &nb_floor_divide, visited);
		binOp(ci->cd, "?", 0, &nb_true_divide, visited);
		binOp(ci->cd, "?=", 0, &nb_inplace_floor_divide, visited);
		binOp(ci->cd, "?=", 0, &nb_inplace_true_divide, visited);
#endif
	}

	// look for sequence methods
	if (ci->hasSequenceMethods) {
		// length (TODO? return type should be int/size_t/?::size_type)
		// TODO? size member function should be const
		unaryOp(ci->cd, "size", 0, &ci->sq_length, visited);
		if (!ci->hasNumberMethods) {
			// concat
			binOp(ci->cd, "operator+", 0, &ci->sq_concat, visited);
			binOp(ci->cd, "operator+=", 0, &ci->sq_inplace_concat,
								visited);
			// repeat
			binOp(ci->cd, "operator*", 0, &ci->sq_repeat, visited);
			intArgOnly(&ci->sq_repeat);
			binOp(ci->cd, "operator*=", 0, &ci->sq_inplace_repeat,
								visited);
			intArgOnly(&ci->sq_inplace_repeat);
		}
		// item (subscript should be int-compatible)
		if (!binOp(ci->cd, "at", 0, &ci->sq_item, visited))
			binOp(ci->cd, "operator[]", 0, &ci->sq_item, visited);
		intArgOnly(&ci->sq_item);
		separateMethods(&ci->sq_item, &ci->sq_ass_item);

		// TODO: slice
		// TODO: ass_slice	(assign slice)

		binOp(ci->cd, "find", 0, &ci->sq_contains, visited);
	}

	// look for mapping methods
	if (ci->hasMappingMethods) {
		// length (TODO: return type should be int/size_t/::size_type)
		// TODO? size member function should be const
		unaryOp(ci->cd, "size", 0, &ci->mp_length, visited);
		// subscript
		binOp(ci->cd, "operator[]", 0, &ci->mp_subscript, visited);
		separateMethods(&ci->mp_subscript, &ci->mp_ass_subscript);
		if (binOp(ci->cd, "find", 0, &ci->sq_contains, visited))
			ci->hasSequenceMethods = true;
	}

	// look for buffer methods
	if (ci->hasBufferProcs) {
		// TODO: getreadbuffer
		// TODO: getwritebuffer
		// TODO: getsegcount
		// TODO: getcharbuffer
	}
}

static std::pair<bool, string>
theGetType(const Decl *scope, const FuncDecl *fd)
{
	// Return the type for a accessor get function.
	// This code should agree with dumpParseArgs(....).

	bool cache = false;
	string gt;	// get type
	if (!fd->returnType.empty() && fd->returnType != "void"
	&& fd->returnType != "static void") {
		CvtType cvt(scope, fd->returnType);
		cache = cvt.cache();
		gt += cvt.pythonType();
	}
	bool tuple = false;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {

		if (!i->out)
			continue;
		string type = i->type;
		if (*type.rbegin() == '*')
			type.erase(type.size() - 1);	// remove trailing *
		CvtType arg(scope, type);
		if (!cache)
			cache = arg.cache();
		if (!gt.empty()) {
			gt += ", ";
			tuple = true;
		}
		gt += arg.pythonType();
	}
	if (!tuple)
		return std::make_pair(cache, gt);
	gt += ')';
	return std::make_pair(cache, '(' + gt);
}

static std::pair<bool, string>
theSetType(const Decl *scope, const FuncDecl *fd)
{
	// Return the type for a accessor set function.
	// This code should agree with dumpParseArgs(....).

	bool cache = false;
	string st;	// set type
	bool tuple = false;
	for (ArgList::const_iterator i = fd->args.begin(); i != fd->args.end();
									++i) {
		if (!i->in)
			continue;
		string type = i->type;
		if (i->out && *type.rbegin() == '*')
			type.erase(type.size() - 1);	// remove trailing *
		CvtType arg(scope, type);
		if (!cache)
			cache = arg.cache();
		if (!st.empty()) {
			st += ", ";
			tuple = true;
		}
		st += arg.pythonType();
	}
	if (!tuple)
		return std::make_pair(cache, st);
	st += ')';
	return std::make_pair(cache, '(' + st);
}

static void
addAttr(ClassInfo *ci, const ClassDecl *cd, DeclSet *visited, Symbol name,
				const FuncDecl *getF, const FuncDecl *setF)
{
	ClassAttrInfo ai;
	if (getF == NULL && setF == NULL)
		return;
	ai.name = name;
	if (getF != NULL) {
		// TODO: should expand method, so we don't have to
		// compile attrMethods at all
		int count = markVisited(cd, getF->tag, visited,
						&ci->attrMethods, ai.name);
		if (count != 1) {
			// getF wrappy not METH_NOARGS, not possible?
			std::cerr << "more than one attribute get function for "
				<< name << " in " << ci->name << "!?\n";
		}
		ai.get = moduleName(ci, getF->tag.str()) + "(self, NULL)";
	}
	if (setF != NULL) {
		// TODO: should expand method, so we don't have to
		// compile attrMethods at all
		int count = markVisited(cd, setF->tag, visited,
						&ci->attrMethods, ai.name);
		ai.set = "\tPyObject* _result = "
					+ moduleName(ci, setF->tag.str())
					+ "(self, value);\n"
			"\tif (_result == NULL)\n"
			"\t\treturn -1;\n"
			"\tPy_DECREF(_result)";
	}
	typedef std::pair<bool, string> info;
	if (getF != NULL && setF != NULL) {
		// determine attribute's type from its functions
		info gInfo = theGetType(getF->scope, getF);
		info sInfo = theSetType(getF->scope, setF);
		if (warnings && gInfo != sInfo)
			std::cerr << programName
				<< ": warning: attribute " << ai.name
				<< " type mismatch: '" << gInfo.second
				<< "' versus '" << sInfo.second << "'\n";
		ai.cache = gInfo.first || sInfo.first;
		ai.doc = gInfo.second;
	} else if (getF != NULL) {
		info gInfo = theGetType(getF->scope, getF);
		ai.cache = gInfo.first;
		ai.doc = gInfo.second;
	} else {
		info sInfo = theSetType(setF->scope, setF);
		ai.cache = sInfo.first;
		ai.doc = sInfo.second;
	}
	ci->attrs.push_back(ai);
}

static void
processComment(ClassInfo *ci, const ClassDecl *cd, DeclSet *visited, const Decl *d)
{
	static const char ro[] = "READONLY:";
	static const string::size_type roLen = sizeof ro - 1;
	if (d->text.size() > roLen && d->text.compare(0, roLen, ro) == 0) {
		// change attribute to be read-only
		string::size_type start = d->text.find_first_not_of(SPACES,
								roLen);
		Symbol name(d->text.substr(start));
		AttrVec::iterator i = std::find_if(ci->attrs.begin(),
					ci->attrs.end(), AttrHasName(name));
		if (i != ci->attrs.end()) {
			i->set = string();
			i = std::find_if(ci->variables.begin(),
					ci->variables.end(), AttrHasName(name));
			if (i != ci->variables.end())
				i->set = string();
		}
		return;
	}

	static const char wo[] = "WRITEONLY:";
	static const string::size_type woLen = sizeof wo - 1;
	if (d->text.size() > woLen && d->text.compare(0, woLen, wo) == 0) {
		// change attribute to be write-only
		string::size_type start = d->text.find_first_not_of(SPACES,
								woLen);
		Symbol name(d->text.substr(start));
		AttrVec::iterator i = std::find_if(ci->attrs.begin(),
					ci->attrs.end(), AttrHasName(name));
		if (i != ci->attrs.end()) {
			i->get = string();
			i = std::find_if(ci->variables.begin(),
					ci->variables.end(), AttrHasName(name));
			if (i != ci->variables.end())
				i->get = string();
		}
		return;
	}

	static const char wr[] = "WEAKREF:";
	static const string::size_type wrLen = sizeof wr - 1;
	if (d->text.size() > wrLen && d->text.compare(0, wrLen, wr) == 0) {
		// change attribute to be not be cached
		string::size_type start = d->text.find_first_not_of(SPACES,
								wrLen);
		Symbol name(d->text.substr(start));
		AttrVec::iterator i = std::find_if(ci->attrs.begin(),
					ci->attrs.end(), AttrHasName(name));
		if (i != ci->attrs.end()) {
			i->cache = false;
			i = std::find_if(ci->variables.begin(),
					ci->variables.end(), AttrHasName(name));
			if (i != ci->variables.end())
				i->cache = false;
		}
		return;
	}

	static const char Attr[] = "ATTRIBUTE:";
	static const string::size_type AttrLen = sizeof Attr - 1;
	if (d->text.size() > AttrLen
	&& d->text.compare(0, AttrLen, Attr) == 0) {
		// look for user defined attributes,
		// could be missing a get or set function.
		string::size_type start
				= d->text.find_first_not_of(SPACES, AttrLen);
		string name = d->text.substr(start);
		string Name = name;
#ifdef OTF_NO_LOCALE
		if (islower(Name[0]))
			Name[0] = toupper(Name[0]);
#else
		if (ct.is(ct.lower, Name[0]))
			Name[0] = ct.toupper(Name[0]);
#endif
		Symbol nameSym(name);

		// look for get functions
		const FuncDecl *getF = findGetFunc(ci, nameSym);
		if (getF == NULL) {
			getF = findGetFunc(ci, Symbol("get" + Name));
			if (getF == NULL)
				getF = findGetFunc(ci, Symbol("Get" + Name));
		}
		// now look for set functions
		const FuncDecl *setF = findSetFunc(ci, Symbol("set" + Name));
		if (setF == NULL)
			setF = findSetFunc(ci, Symbol("Set" + Name));
		addAttr(ci, cd, visited, nameSym, getF, setF);
		if (getF) {
			// remove from methods
			MethodRange r = ci->methods.equal_range(getF->tag);
			for (MethodMap::iterator i = r.first; i != r.second;
					++i) {
				if (i->second == getF) {
					ci->methods.erase(i);
					break;
				}
			}
		}
		if (setF) {
			// remove from methods
			MethodRange r = ci->methods.equal_range(setF->tag);
			for (MethodMap::iterator i = r.first; i != r.second;
					++i) {
				if (i->second == setF) {
					ci->methods.erase(i);
					break;
				}
			}
		}
		return;
	}
}

static void
computeClassInfoAttributes(ClassInfo *ci, const ClassDecl *cd, DeclSet *visited)
{
	if (cd == ci->cd)
		computeBuiltins(ci, visited);

	// recurse into base classes if appropriate
	const ClassInfo *curci = classInfoMap[cd];
	if (curci != NULL)
		for (CIList::const_iterator i = curci->baseClasses.begin();
					i != curci->baseClasses.end(); ++i) {
			const ClassInfo *bci = *i;
			if (curci->isWrappySubclass && bci->isWrappySubclass)
				continue;
			// this adds base class attributes to current class
			computeClassInfoAttributes(ci, bci->cd, visited);
		}

	typedef std::vector<const Decl *> CommentList;
	CommentList comments;
	// now do this class
	for (DeclList::const_iterator i = cd->members.begin();
						i != cd->members.end(); ++i) {
		const Decl *d = *i;
		if (visited->find(d) != visited->end())
			continue;
		visited->insert(d);
		if (d->access != PUBLIC
		|| (!d->tag.empty() && d->tag.str()[0] == '~'))
			continue;

		switch (d->dtype) {
		  default:
			// anything not covered, we ignore
			break;
		  case Decl::COMMENT:
			// process all comments at the end of scope
			comments.push_back(d);
			break;
		  case Decl::CONSTANT: {
			const VarDecl *vd = static_cast<const VarDecl *>(d);
			CvtType cvt(cd, vd->type);
			ClassAttrInfo ai;
			ai.name = d->tag;
			ai.get = make_buildvalue(cvt.bvFormat(), 
				qualify(cd, cvt.bvArg(vd->tag.str())));
			ai.set = "";
			ai.doc = cvt.pythonType();
			ci->constants.push_back(ai);
			break;
		  }
		  case Decl::VARIABLE: {
			const VarDecl *vd = static_cast<const VarDecl *>(d);
			CvtType cvt(cd, vd->type);
			ClassAttrInfo ai;
			ai.name = d->tag;
			ai.cache = cvt.cache();
			ai.doc = cvt.pythonType();
			// TODO? ideally we'd move this code generation to
			// class.cpp and type.cpp.
			string prefix(qualify(ci->cd->scope, ci->name.str()));
			string objectName = moduleName(ci, "object");
			string attr;
			if (!ci->isEmbedded)
				attr = "getInst(static_cast<" + objectName
								+ "*>(self))->";
			else if (!cvt.PyMemberDefType().empty()) {
				ai.type = cvt.PyMemberDefType();
				ai.offset = "offsetof(" + objectName
					+ ", _inst_data) + offsetof(" + prefix
					+ ", " + vd->tag.str() + ")";
				ai.get = "yes";
				ai.set = "yes";
			} else {
				attr = "getInst(static_cast<" + objectName
								+ "*>(self))->";
			}
			if (!attr.empty()) {
				attr += vd->tag.str();
				ai.get = make_buildvalue(cvt.bvFormat(),
							cvt.bvArg(attr));
				ai.set = "\tif (!"
					+ cvt.typeCheck("value") + ") {\n"
					"\t\tPyErr_SetString(PyExc_TypeError, \"attribute "
					+ vd->tag.str() + " should be a '"
					+ cvt.pythonType() + "'\");\n"
					"\t\treturn -1;\n"
					"\t}\n"
					"\t" + attr + " = "
					+ qualify(cd, cvt.pyToCpp("value"));
			}
			ci->variables.push_back(ai);
			// simple attribute function (inherit cache from above)
#if 0
			ai.get = ci->name.str() + "Get" + vd->tag.str()
								+ "(self)";
			ai.set = ci->name.str() + "Set" + vd->tag.str()
							+ "(self, value)";
#endif
			ci->attrs.push_back(ai);
			break;
		  }
		  case Decl::FUNCTION: {
			const FuncDecl *fd = static_cast<const FuncDecl *>(d);
			if (fd->tag.str().compare(0, 8, "operator") == 0)
				break;
			if (fd->tag == ci->name) {
				// save away constructors
				ci->constructors.insert(
					MethodMap::value_type(fd->tag, fd));
				break;
			}
			if (fd->tag == cd->name)
				// skip subclass constructors
				break;
			if (btStatic.find(fd->returnType) == 0) {
				// save away class methods
				ci->staticMethods.insert(
					MethodMap::value_type(fd->tag, fd));
				break;
			}

			string name = fd->tag.str();
			const FuncDecl *getF = NULL;
			const FuncDecl *setF = NULL;
			Symbol nameSym;
			if (name.compare(0, 3, "set") == 0
			|| name.compare(0, 3, "Set") == 0) {
				// if name is set* then look for * or get*
				setF = findSetFunc(ci, fd->tag);
				if (setF == NULL) {
					ci->methods.insert(
						MethodMap::value_type(fd->tag,
									fd));
					break;
				}
				string attr = name.substr(3);
#ifdef OTF_NO_LOCALE
				if (isupper(attr[0]))
					attr[0] = tolower(attr[0]);
#else
				if (ct.is(ct.upper, attr[0]))
					attr[0] = ct.tolower(attr[0]);
#endif
				nameSym = Symbol(attr);
				getF = findGetFunc(ci, nameSym);
				if (getF == NULL) {
					Symbol get = Symbol(
						(name[0] == 's' ? 'g' : 'G')
						+ name.substr(1));
					getF = findGetFunc(ci, get);
				}
			}
			else if (name.compare(0, 3, "get") == 0
			|| name.compare(0, 3, "Get") == 0) {
				// else if name is get* then look for set*
				getF = findGetFunc(ci, fd->tag);
				if (getF == NULL) {
					ci->methods.insert(
						MethodMap::value_type(fd->tag,
									fd));
					break;
				}
				string attr = name.substr(3);
#ifdef OTF_NO_LOCALE
				if (isupper(attr[0]))
					attr[0] = tolower(attr[0]);
#else
				if (ct.is(ct.upper, attr[0]))
					attr[0] = ct.tolower(attr[0]);
#endif
				nameSym = Symbol(attr);
				Symbol set = Symbol((name[0] == 'g' ? 's' : 'S')
								+ name.substr(1));
				setF = findSetFunc(ci, set);
			}
			else if ((getF = findGetFunc(ci, fd->tag)) != NULL) {
				// if name is * then look for set*
				nameSym = getF->tag;
#ifdef OTF_NO_LOCALE
				if (islower(name[0]))
					name[0] = toupper(name[0]);
#else
				if (ct.is(ct.lower, name[0]))
					name[0] = ct.toupper(name[0]);
#endif
				setF = findSetFunc(ci, Symbol("set" + name));
				if (setF == NULL)
					setF = findSetFunc(ci,
							Symbol("Set" + name));
			}
			if (getF == NULL || setF == NULL) {
				// attributes have both set() and get()
				ci->methods.insert(
					MethodMap::value_type(fd->tag, fd));
				break;
			}
			addAttr(ci, cd, visited, nameSym, getF, setF);
			break;
		  }
		}
	}
	for (CommentList::iterator i = comments.begin(); i != comments.end(); ++i) {
		const Decl *d = *i;
		processComment(ci, cd, visited, d);
	}
	// TODO: remove following bandaid
	if (cd == ci->cd) {
		// remove duplicates
		std::stable_sort(ci->attrs.begin(), ci->attrs.end(),
							AttrNameLt());
		AttrVec::iterator dups = std::unique(ci->attrs.begin(),
					ci->attrs.end(), AttrNameEq());
		ci->attrs.erase(dups, ci->attrs.end());
	}
}

static ClassInfo *
fillInClassInfo(ClassInfo *ci)
{
	ClassDecl *cd = const_cast<ClassDecl *>(ci->cd);

	// figure out if we should default to embedded instances
	if (cd->hasPublicCopyConstructor && !ci->isWrappySubclass
	&& !ci->isBaseClass && ci->baseClasses.empty())
		ci->isEmbedded = true;

	// scan members (esp. comments) for interesting stuff
	for (DeclList::iterator i = cd->members.begin();
						i != cd->members.end(); ++i) {
		Decl *d = *i;
		if (d->dtype == Decl::FUNCTION) {
			FuncDecl *fd = static_cast<FuncDecl *>(d);
			if (fd->isPure) {
				ci->isAbstractType = true;
				ci->isEmbedded = false;
			}
			continue;
		}
		if (d->dtype != Decl::COMMENT)
			continue;
		if (d->text == "WRAP CLASS")
			ci->skipClass = false;
		else if (d->text == "BASE CLASS")
			ci->isBaseClass = true;
		else if (d->text == "FINAL")
			ci->isFinal = true;
#if 0
		// if any of the above conditions are violated, then
		// we can't embed properly
		else if (d->text == "EMBED")
			ci->isEmbedded = true;
#endif
		else if (d->text == "SEPARATE")
			ci->isEmbedded = false;
		else if (d->text == "STRUCT SEQUENCE") {
			std::cerr << "warning: only attribute access provided as this time\n";
			ci->isStruct = true;
			ci->isFinal = true;
		}
		else if (d->text == "ABSTRACT") {
			ci->isAbstractType = true;
			ci->isEmbedded = false;
		} else if (d->text == "NUMBER METHODS")
			ci->hasNumberMethods = true;
		else if (d->text == "SEQUENCE METHODS")
			ci->hasSequenceMethods = true;
		else if (d->text == "MAPPING METHODS")
			ci->hasMappingMethods = true;
		else if (d->text == "BUFFER PROCS") {
			std::cerr << "warning: buffer procs are not supported\n";
			ci->hasBufferProcs = true;
			ci->hasSequenceMethods = true;
		}
		else if (d->text == "DON'T CACHE")
			ci->dontCache = true;
		else if (d->text == "IDENTITY HASH")
			ci->genIdentityHash = true;
	}
	if (ci->isEmbedded)
		ci->dontCache = true;

	// if includefile isn't set, then set it
	if (ci->includeFile.empty()) {
		ci->includeFile = "# include ";
		if (cd->filename == "standard input")
			ci->includeFile += '"' + cd->name.str() + ".h\"";
		else if (cd->filename.c_str()[0] == '<'
		|| cd->filename.c_str()[0] == '"')
			ci->includeFile += cd->filename.str();
		else
			ci->includeFile += '"' + cd->filename.str() + '"';
	}

	// remove all declarations we're not going to look at
	typedef std::vector<Decl *> DeclVec;
	DeclVec removeDecls;
	for (DeclList::iterator i = cd->members.begin();
						i != cd->members.end(); ++i) {
		Decl *d = *i;
		if (d->dtype == Decl::COMMENT)
			// save comments
			continue;
		if (!d->tag.empty()
		&& (d->tag == cd->name || d->tag.str()[0] == '~'))
			// keep private constructors and destructors
			continue;
		if (d->access != PUBLIC)
			// eliminate private and protected regions of class
			removeDecls.push_back(d);
		else if (d->tag.empty())
			removeDecls.push_back(d);
	}
	for (DeclVec::iterator i = removeDecls.begin(); i != removeDecls.end();
									++i)
		cd->members.remove(*i);
	return ci;
}

static void
fillInClassInfo()
{
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		ClassInfo *ci = const_cast<ClassInfo*>(i->second);
		fillInClassInfo(ci);
	}
}

static void
fillInClassDecl(ClassDecl *cd)
{
	// Try to mimic C++ behavior of creating a default constructor,
	// a default destructor, a copy constructor, and an assignment
	// operator if they don't already exist.

	if (isWrapPySubclass(cd))
		cd->isWrapPySubclass = true;

	bool isAbstractType = false;
	bool implicitCopyConstructor = false;
	for (DeclList::iterator i = cd->members.begin();
						i != cd->members.end(); ++i) {
		Decl *d = *i;
		if (d->dtype == Decl::COMMENT) {
			if (d->text == "ABSTRACT") {
				isAbstractType = true;
				break;
			} else if (d->text == "IMPLICIT COPY CONSTRUCTOR") {
				implicitCopyConstructor = true;
				break;
			}
		}
		if (d->dtype == Decl::FUNCTION) {
			FuncDecl *fd = static_cast<FuncDecl *>(d);
			if (fd->isPure) {
				isAbstractType = true;
				break;
			}
		}
	}

	// look for constructors
	bool hasConstructor = false;
	bool hasCopyConstructor = false;
	DeclList::RARange range = cd->members.rAequal_range(cd->name);
	for (DeclList::RAiterator i = range.first; i != range.second; ++i) {
		FuncDecl *fd = dynamic_cast<FuncDecl *>(i->second);
		if (fd == NULL)
			continue;
		hasConstructor = true;
		if (fd->args.size() == 1) {
			// check for copy constructor
			const string &type = fd->args[0].type;
			if (type == cd->name.str() + '&'
			|| type == cd->name.str() + " const&"
			|| type == "const " + cd->name.str() + '&') {
				hasCopyConstructor = true;
				if (fd->access == PUBLIC)
					cd->hasPublicCopyConstructor = true;
			}
		}
	}
	if (!hasConstructor) {
		// no constructor found, fake one
		FuncDecl *fd = new FuncDecl(PUBLIC, cd, cd->filename);
		fd->tag = cd->name;
		fd->returnType = "";
		cd->members.append(fd);
	}
	if (isAbstractType) {
		if (/*!hasCopyConstructor ||*/ cd->hasPublicCopyConstructor)
			std::cerr << "error: public copy constructor found in abstract class " << cd->name << '\n';
	} else if (!hasCopyConstructor && implicitCopyConstructor) {
		// Can only copy if base classes are copyable and all
		// members variables are copyable.
		// no copy constructor found, fake one
		FuncDecl *fd = new FuncDecl(PUBLIC, cd, cd->filename);
		fd->tag = cd->name;
		fd->returnType = "";
		Argument arg;
		arg.name = Symbol("_x");
		arg.type = "const " + cd->name.str() + '&';
		fd->args.push_back(arg);
		cd->members.append(fd);
		cd->hasPublicCopyConstructor = true;
	}
	// check for destructor
	Symbol dtag = Symbol('~' + cd->name.str());
	if (!cd->members.has(dtag, ANY)) {
		// no destructor found, fake one
		FuncDecl *fd = new FuncDecl(PUBLIC, cd, cd->filename);
		fd->tag = dtag;
		fd->returnType = "";
		cd->members.append(fd);
	}
	MethodMap mm;
	bool hasAssign = binOp(cd, "operator=", NULL, &mm, NULL);
	if (isAbstractType) {
		bool hasPublicAssign = false;
		for (MethodMap::iterator i = mm.begin(); i != mm.end(); ++i) {
			const FuncDecl *fd = i->second;
			if (fd->access == PUBLIC) {
				hasPublicAssign = true;
				break;
			}
		}
		if (hasPublicAssign) {
			std::cerr << "error: public operator= found for " << cd->name << '\n';
		}
	} else if (!hasAssign) {
		// no operator=, fake one
		FuncDecl *fd = new FuncDecl(PUBLIC, cd, cd->filename);
		fd->tag = Symbol("operator=");
		fd->returnType = "void";
		Argument arg;
		arg.name = Symbol("_x");
		arg.type = "const " + cd->name.str() + '&';
		fd->args.push_back(arg);
		cd->members.append(fd);
	}
}

static void
computeClassDeclMap(DeclList *dl)
{
	for (DeclList::iterator i = dl->begin(); i != dl->end(); ++i) {
		switch ((*i)->dtype) {
		  default:
			break;
		  case Decl::CLASS:
		  case Decl::STRUCT: {
			ClassDecl *cd = static_cast<ClassDecl *>(*i);
			fillInClassDecl(cd);
			string qualName = qualify(cd->scope, cd->name.str());
			classDeclsMap[Symbol(qualName)] = cd;
			computeClassDeclMap(&cd->members);
			break;
		  }
		  case Decl::NAMESPACE: {
			NamespaceDecl *nd = static_cast<NamespaceDecl *>(*i);
			computeClassDeclMap(&nd->decls);
			break;
		  }
		  case Decl::COMMENT: {
			// #include is handled by in ParseHeader.cpp
			Decl *d = *i;
			if (d->text.substr(0, 14) == "ALLOW THREADS:") {
				string tmp = removeBlanks(d->text.substr(15));
				Symbol tag(tmp);
				DeclList::RAConstRange range
						= dl->rAequal_range(tag);
				bool found = false;
				for (DeclList::RAconst_iterator i = range.first;
						i != range.second; ++i) {
					FuncDecl *fd = dynamic_cast<FuncDecl *>
								(i->second);
					if (fd == NULL)
						continue;
					fd->allowThreads = true;
					found = true;
				}
				if (!found)
					std::cerr << "warning: no " << tmp
							<< " functions found\n";
			}
			break;
		  }
		}
	}
}

static void
computeClassInfoMap(DeclList *dl, bool genScope)
{
	for (DeclList::iterator i = dl->begin(); i != dl->end(); ++i) {
		switch ((*i)->dtype) {
		  default:
			break;
		  case Decl::CLASS:
		  case Decl::STRUCT: {
			ClassDecl *cd = static_cast<ClassDecl *>(*i);
			computeDerivedClasses(cd);
			computeClassInfoMap(&cd->members, genScope);
			ClassInfo *ci = new ClassInfo(cd);
			classInfoMap[cd] = ci;
			if (genScope
			&& (cd->access == PUBLIC || cd->access == GLOBAL)) {
				string tmp = qualify(cd->scope, cd->name.str(),
									true);
				if (tmp.substr(0, nameSpace.size() + 1)
							== nameSpace + ':')
					ci->skipClass = false;
				else {
					// TODO: never happens because genScope
					// is always false for classes in other
					// name spaces.`:w`
					// assume it's found in other wrappy
					// generated code
					tmp = tmp.substr(0, tmp.find("::"));
					ci->includeFile = "# include <" + tmp
						+ '/' + tmp + ">";
				}
			}
			break;
		  }
		  case Decl::NAMESPACE: {
			NamespaceDecl *nd = static_cast<NamespaceDecl *>(*i);
			bool saveGenScope = genScope;
			if (!genScope && nameSpace == nd->tag) {
				genScope = true;
				moduleScope = nd;
			}
			computeClassInfoMap(&nd->decls, genScope);
			genScope = saveGenScope;
			break;
		  }
		}
	}
}

static bool
isWrapPySubclass(ClassInfo *ci)
{
	if (ci->cd->isWrapPySubclass)
		return true;
	// double check subclasses for indirect inheritance
	for (CIList::iterator i = ci->baseClasses.begin();
				i != ci->baseClasses.end(); ++i) {
		ClassInfo *base = *i;
		if (isWrapPySubclass(base))
			return true;
	}
	return false;
}

void
computeBaseClasses()
{
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		ClassInfo *ci = const_cast<ClassInfo*>(i->second);
		ClassDecl *cd = const_cast<ClassDecl*>(ci->cd);	// or i->first
		for (DeclList::const_iterator i = cd->baseClasses.begin();
					i != cd->baseClasses.end(); ++i) {
			string qualName = qualify(cd->scope, (*i)->tag.str());
			const ClassDecl *bcd = classDeclsMap[Symbol(qualName)];
			if (bcd == NULL)
				continue;
			CDCIMap::iterator j = classInfoMap.find(bcd);
			if (j == classInfoMap.end())
				continue;
			ClassInfo *base = const_cast<ClassInfo*>(j->second);
			ci->baseClasses.push_back(base);
		}
	}

	// Recurse into base classes to assign whether or not class
	// subclasses directly or indirectly from WrapPyObj.
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		ClassInfo *ci = const_cast<ClassInfo*>(i->second);

		ci->isWrappySubclass = isWrapPySubclass(ci);
		if (ci->isWrappySubclass && ci->isFinal) {
			ci->isFinal = false;
			std::cerr << "warning: ignoring FINAL for " << ci->name 
									<< '\n';
		}
	}
}

void
computeUsing(DeclList *dl)
{
	typedef std::vector<Decl *> DeclVec;
	DeclVec removeDecls;
	DeclVec addDecls;
	for (DeclList::iterator i = dl->begin(); i != dl->end(); ++i) {
		Decl *d = *i;
		switch (d->dtype) {
		  default:
			break;
		  case Decl::USING: {
			static const int uSz = sizeof "using";
			string name = d->text.substr(uSz, d->text.size() - uSz);
			if (name.compare(0, 9, "namespace") == 0) {
				std::cerr << programName
				    << ": 'using namespace' is not supported\n";
				removeDecls.push_back(d);
				continue;
			}
			removeDecls.push_back(d);
			if (*name.rbegin() == ';')
				// remove trailing ;
				name.erase(name.size() - 1);
			DeclList::RAConstRange ads
					= funcVarTypeDecls(d->scope, name);
			for (DeclList::RAconst_iterator j = ads.first;
							j != ads.second; ++j)
				addDecls.push_back(j->second);
			break;
		  }
		  case Decl::CLASS:
		  case Decl::STRUCT: {
			ClassDecl *cd = static_cast<ClassDecl *>(*i);
			computeUsing(&cd->members);
			break;
		  }
		  case Decl::NAMESPACE: {
			NamespaceDecl *nd = static_cast<NamespaceDecl *>(*i);
			computeUsing(&nd->decls);
			break;
		  }
		}
	}
	for (DeclVec::iterator i = removeDecls.begin(); i != removeDecls.end();
									++i)
		dl->remove(*i);
	for (DeclVec::iterator i = addDecls.begin(); i != addDecls.end(); ++i) {
		Decl *d = *i;

		// need to avoid duplicates (esp. of types)
		switch (d->dtype) {
		  default:
			break;
		  case Decl::TYPEDEF:
		  case Decl::STRUCT:
		  case Decl::CLASS: {
			DeclList::RAConstRange range
						= dl->rAequal_range(d->tag);
			if (range.first != range.second)
				continue;
			break;
		  }
		  case Decl::FUNCTION: {
			FuncDecl *fd = dynamic_cast<FuncDecl *>(d);
			DeclList::RAConstRange range
						= dl->rAequal_range(d->tag);
			for (DeclList::RAconst_iterator i = range.first;
						i != range.second; ++i) {
				FuncDecl *cfd
					= dynamic_cast<FuncDecl *>(i->second);
				if (cfd == NULL)
					continue;
				if (*fd == *cfd)
					goto skip_it;
			}
			break;
		skip_it:
			continue;
		  }
		}
		// TODO? Change scope of declaration
		dl->append(d->copy());
	}
}

static void
fixupBaseClasses(ClassInfo *ci)
{
	for (CIList::iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
		ClassInfo *base = *i;
		if (base->isBaseClass)
			continue;
		base->isBaseClass = true;
		if (!base->baseClasses.empty())
			fixupBaseClasses(base);
	}
}

static void
confirmSubclassConsistency()
{
	if (derivedClasses.empty())
		return;

	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		ClassInfo *ci = const_cast<ClassInfo*>(i->second);
		if (ci->baseClasses.empty())
			continue;
		fixupBaseClasses(ci);
	}

	// make sure all abstract types are base classes
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		ClassInfo *ci = const_cast<ClassInfo*>(i->second);
		if (ci->isAbstractType)
			ci->isBaseClass = true;
	}
}

static void
computeAttributes()
{
	// find attributes
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		ClassInfo *ci = const_cast<ClassInfo*>(i->second);
		if (ci->skipClass)
			continue;
		DeclSet visited;
		computeClassInfoAttributes(ci, ci->cd, &visited);
	}
}

void
computeWrapInformation()
{
	computeClassDeclMap(&globalDecls);
	computeClassInfoMap(&globalDecls, nameSpace.empty());
	computeBaseClasses();
	confirmSubclassConsistency();
	fillInClassInfo();
	computeUsing(&globalDecls);
	computeAttributes();
}
