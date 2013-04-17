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

// $Id: method.h 26655 2009-01-07 22:02:30Z gregc $

#ifndef method_h
# define method_h

# include <ostream>
# include <string>
# include "compute.h"

struct MethodInfo {
	std::string doc;
	std::string ml_flags;
};

enum MethodType {
	// these constants correspond to the various type of methods in the
	// C API: PyMemberDef, PyGetSetDef, PyNumberMethods, PySequenceMethods,
	// PyMappingMethods, and PyBufferProcs (and potentially others in the
	// _typeobject structure).
	Normal, Constructor, Attribute,
	UnaryFunc, BinaryFunc, TernaryFunc,
	Inquiry, Coercion, SsizeArgFunc, SsizeSsizeArgFunc,
	SsizeObjArgProc, SsizeSsizeObjArgProc, ObjObjArgProc,
	GetReadBufferProc, GetWriteBufferProc, GetSegCountProc,
	GetCharBufferProc, ObjObjProc
};

extern void
dumpMethod(std::ostream &output, const ClassInfo *ci,
		MethodMap::const_iterator start, MethodMap::const_iterator stop,
		/*OUT*/ MethodInfo *mi, std::string fname = std::string(),
		std::string pyname = std::string(), MethodType mtype = Normal);

inline void
dumpMethod(std::ostream &output, const ClassInfo *ci, const MethodMap &mm,
		/*OUT*/ MethodInfo *mi, std::string fname = std::string(),
		std::string pyname = std::string(), MethodType mtype = Normal)
{
	dumpMethod(output, ci, mm.begin(), mm.end(), mi, fname, pyname, mtype);
}

extern bool
dumpOperator(std::ostream &output, const MethodMap &mm,
		const std::string &fname, std::string pyname = std::string(),
		MethodType mtype = UnaryFunc);

#endif
