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

// $Id: CvtType.h 26655 2009-01-07 22:02:30Z gregc $

#ifndef CvtType_h
# define CvtType_h

# include <set>
# include <map>
# include "Symbol.h"
# include "ParseHeader.h"

class CvtType {
	// C++ parameter support (cpp)
	//
	// Given a C++ type (cppType in a given scope), describe how to
	// convert it to and from PyObject* and its PyArg_ParseTuple type.
public:
	CvtType(const Decl *scope, const std::string &cppType, bool noneOk=false);
	CvtType() {}

	// cache() returns true if attributes of type should be cached
	// -- usually true if C++ type is a C++ class.
	bool		cache() const;

	const std::string &cppType() const;	// C++ argument type
	const std::string &baseCppType() const;	// base type (followed
						// typedef's, remove reference)
	std::string cppTypeDecl(const std::string &argName,
						bool noReference = false) const;
				//  return declaration with given argument name
	bool hasImplicitAddress() const;
				// true if an array or reference

	// PyArg_ParseTuple support (apt)
	// 	PyArg_ParseTuple is used to convert PyObject*'s to C++ types
	// 	
	// 	aptType is the C type corresponding to the aptFormat.  If
	// 	aptType is PyObject* (i.e, no conversion), then we
	// 	typically need to type check (needTypeCheck, typeCheck) the
	// 	parsed PyObject* and then convert it with pyToCpp.
	//
	// 	aptToCpp returns code that converts the aptType variable
	// 	to the C++ type.
	//
	// 	cppToActual returns code that converts the C++ type variable
	// 	to the actual C++ type (usually a noop because the the C++
	// 	type is the same as the actual C++ type).
	// 	
	// 	cppToApt returns code that converts the actual C++ type
	// 	to the aptType (used for converting default argument values).
	//
	// 	aptCleanup returns code needed to undo any aptToCpp or
	// 	cppToActual conversion side effects.
	//
	// 	requireAPT returns true if PyArg_ParseTuple should always
	// 	be used.
	const std::string &aptType() const;
	const std::string &aptFormat() const;
	std::string	aptToCpp(const std::string &arg) const;
	std::string	cppToActual(const std::string &arg) const;
	std::string	cppToApt(const std::string &arg) const;
	bool		needTypeCheck() const;
	std::string	aptCleanup(const std::string &arg) const;
	bool		requireAPT() const;

	// Knowledge about PyObject*'s:
	//
	//	typeCheck returns code that checks if a PyObject* can be
	//	mapped to a given C++ type.
	//
	//	pyToCpp return code that converts a PyObject* to the C++ type
	//	(you need to call PyErr_Occured afterwards).
	std::string	typeCheck(const std::string &arg) const;
	std::string	pyToCpp(const std::string &arg) const;

	// Py_BuildValue support (bv)
	// 	Py_BuildValue is used to convert C++ types to PyObject*'s
	//
	// 	bvFormat returns what's needed for Py_BuildValue format
	// 	and bvArg
	const std::string &bvFormat() const;
	std::string	bvArg(const std::string &arg) const;

	// PyMemberDef_type() returns the PyMemberDef type that corresponds
	// to cpp type (or an empty string if there isn't any).
	const std::string &PyMemberDefType() const;

	// pythonTYpe gives a "Python" type useful for documentation
	const std::string &pythonType() const;
private:
	std::string	cppType_;
	std::string	type_;		// baseCppType
	std::string	declPattern;
	bool		implicitAddress;
	bool		cacheMe;
	std::string	pyToCppPattern;
	std::string	aptType_;
	std::string	aptFormat_;
	bool		aptNeedCheck;
	bool		aptRequired;
	std::string	aptCleanupPattern;
	std::string	aptToCppPattern;
	std::string	cppToActualPattern;
	std::string	cppToAptPattern;
	std::string	typeCheckPattern;
	std::string	bvFormat_;
	std::string	bvPattern;
	std::string 	pymemberdef_type;
	std::string 	python_type;
};

inline bool
CvtType::cache() const
{
	return cacheMe;
}

inline const std::string &
CvtType::cppType() const
{
	return cppType_;
}

inline const std::string &
CvtType::baseCppType() const
{
	return type_;
}

inline bool
CvtType::hasImplicitAddress() const
{
	return implicitAddress;
}

inline const std::string &
CvtType::aptFormat() const
{
	return aptFormat_;
}

inline const std::string &
CvtType::aptType() const
{
	return aptType_;
}

inline bool
CvtType::needTypeCheck() const
{
	return aptNeedCheck;
}

inline const std::string &
CvtType::bvFormat() const
{
	return bvFormat_;
}

inline const std::string &
CvtType::PyMemberDefType() const
{
	return pymemberdef_type;
}

inline const std::string &
CvtType::pythonType() const
{
	return python_type;
}

inline bool
CvtType::requireAPT() const
{
	return aptRequired;
}

extern DeclList::RAConstRange
		funcVarTypeDecls(const Decl *scope, std::string name);
extern const Decl *
		typeDecl(const Decl *scope, Symbol id);
extern std::string
		qualify(const Decl *scope, const std::string &str,
							bool keepNS = false);
extern std::string
		baseType(const Decl *scope, const std::string &name);
extern std::string
		scopeName(const Decl *scope, const std::string &sname);

#endif
