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

// $Id: compute.h 26655 2009-01-07 22:02:30Z gregc $

#ifndef compute_h
# define compute_h

# include "ParseHeader.h"

struct ClassInfo;

typedef std::vector<ClassInfo *> CIList;
typedef std::set<const ClassDecl *> CDSet;

struct ClassAttrInfo {
	Symbol		name;
	std::string	get;
	std::string	set;
	std::string	doc;
	std::string	offset;		// for PyMemberDef
	std::string	type;		// for PyMemberDef
	bool		cache;
	ClassAttrInfo(): cache(false) {}
};
typedef std::vector<ClassAttrInfo> AttrVec;

typedef std::multimap<Symbol, const FuncDecl *> MethodMap;
typedef std::pair<MethodMap::iterator, MethodMap::iterator> MethodRange;

struct ClassInfo {
	// This structure describes how a C++ class is converted to
	// a Python class or type.
	ClassDecl	*cd;
	CIList		baseClasses;
	Symbol		name;
	std::string	includeFile;
	bool		skipClass;		// class in header, no code 
	bool		isWrappySubclass;	// subclass of WrapPy<>
	bool		isAbstractType;		// an implicit Python type
	bool		isBaseClass;
	bool		isEmbedded;
	bool		isFinal;		// can use __slots__
	bool		isStruct;		// use StructSequence
	bool		hasNumberMethods;
	bool		hasSequenceMethods;
	bool		hasMappingMethods;
	bool		hasBufferProcs;
	bool		dontCache;
	bool		genIdentityHash;
	AttrVec		attrs;
	AttrVec		constants;
	AttrVec		variables;	// a subset of attrs that are variables
	MethodMap	attrMethods;	// attributes that are C++ member funcs
	MethodMap	methods;	// public member funcs
	MethodMap	staticMethods;	// static public member funcs
	MethodMap	constructors;
	// Python type methods
	bool		print;
	MethodMap	compare, hash;
	MethodMap	call, repr, str;
	// Use copy constructor to support copy module.
	MethodMap	copy;
			// Python 2.1
	MethodMap	lt, le, eq, ne, gt, ge;
	// Python number methods
	MethodMap	nb_add, nb_subtract, nb_multiply, nb_divide,
			nb_remainder, nb_divmod, nb_power, nb_negative,
			nb_positive, nb_absolute, nb_nonzero, nb_invert,
			nb_lshift, nb_rshift, nb_and, nb_xor, nb_or;
	bool		nb_coerce;
	MethodMap	nb_int, nb_long, nb_float, nb_oct, nb_hex;
			// added in Python 2.0
	MethodMap	nb_inplace_add, nb_inplace_subtract,
			nb_inplace_multiply, nb_inplace_divide,
			nb_inplace_remainder, nb_inplace_power,
			nb_inplace_lshift, nb_inplace_rshift, nb_inplace_and,
			nb_inplace_xor, nb_inplace_or;
			// added in Python 2.2
	MethodMap	nb_floor_divide, nb_true_divide,
			nb_inplace_floor_divide, nb_inplace_true_divide;
#if 1
	// reverse number methods for Python classic classes
	MethodMap	nb_radd, nb_rsubtract, nb_rmultiply, nb_rdivide,
			nb_rremainder, nb_rdivmod, nb_rpower,
			nb_rlshift, nb_rrshift, nb_rand, nb_rxor, nb_ror;
#endif
	// Python sequence methods
	MethodMap	sq_length, sq_concat, sq_repeat, sq_item,
			sq_slice, sq_ass_item, sq_ass_slice;
			// added in Python 1.6
	MethodMap	sq_contains;
			// added in Python 2.0
	MethodMap	sq_inplace_concat, sq_inplace_repeat;
	// Python mapping methods
	MethodMap	mp_length, mp_subscript, mp_ass_subscript;
	// Python buffer procedures
	MethodMap	bf_getreadbuffer, bf_getwritebuffer, bf_getsegcount,
			bf_getcharbuffer;

	ClassInfo(ClassDecl *d): cd(d), name(d->name),
		skipClass(true), isWrappySubclass(false),
		isAbstractType(false), isBaseClass(false), isEmbedded(false),
		isFinal(false), isStruct(false),
		hasNumberMethods(false), hasSequenceMethods(false),
		hasMappingMethods(false), hasBufferProcs(false),
		dontCache(false), genIdentityHash(false),
		print(0), nb_coerce(false)
	{
	}
};

struct CDCIOrder:
	public std::binary_function<const ClassDecl*, const ClassDecl*, bool>
{
	bool operator()(const ClassDecl *a, const ClassDecl *b) const {
		return a->name < b->name;
	}
};

typedef std::map<const ClassDecl *, const ClassInfo *, CDCIOrder> CDCIMap;

class AttrNameLt {
public:
	bool operator()(const ClassAttrInfo &a, const ClassAttrInfo &b) const
	{
		return a.name < b.name;
	}
};

class AttrNameEq {
public:
	bool operator()(const ClassAttrInfo &a, const ClassAttrInfo &b) const
	{
		return a.name == b.name;
	}
};

class AttrHasName {
	Symbol name;
public:
	AttrHasName(Symbol n): name(n) {}
	bool operator()(const ClassAttrInfo &a) const
	{
		return a.name == name;
	}
};

typedef std::map<Symbol, const ClassDecl *> SymCDMap;

// All global declarations -- input to this module -- the parse result
extern DeclList		globalDecls;

//
// The following global variables are the output of this module
//

extern NamespaceDecl	*moduleScope;	// module scope, NULL if same as global

extern SymCDMap	classDeclsMap;		// key is C++ class name
extern CDCIMap	classInfoMap;		// key is ClassDecl*

// key is base class, values are derived classes
typedef std::multimap<Symbol, Symbol> DerivedClasses;
extern DerivedClasses	derivedClasses;

extern void	computeWrapInformation();

#endif
