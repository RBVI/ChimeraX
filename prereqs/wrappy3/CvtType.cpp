// Copyright (c) 1998-1999 The Regents of the University of California.
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

// $Id: CvtType.cpp 35469 2012-02-03 00:06:06Z gregc $

#include "common.h"
#include "compute.h"
#include "CvtType.h"
#include "Reg.h"
#include <stdlib.h>
#include <assert.h>

using std::string;

static string
getSubType(const string &ttype, /*INOUT*/ string::size_type *start)
{
	// ttype is a template type,
	// start is the offset to start looking for a subtype
	string::size_type s = *start;
	if (s == 0) {
		s = ttype.find("<");
		if (s == string::npos) {
			*start = ttype.size();
			return ttype;
		}
	}
	switch (ttype[s]) {
	  case '<':
	  case ',':
		++s;
		break;
	}
	std::vector<char> stack;
	string::size_type i = s;
	for (; i < ttype.size(); ++i) {
		switch (ttype[i]) {
		  case '<':
			stack.push_back(ttype[i]);
			break;
		  case ')':
			if (stack.empty() || stack.back() != '(')
				goto done;
			stack.pop_back();
			break;
		  case ']':
			if (stack.empty() || stack.back() != '[')
				goto done;
			stack.pop_back();
			break;
		  case '>':
			if (stack.empty() || stack.back() != '<')
				goto done;
			stack.pop_back();
			break;
		  case ',':
			if (stack.empty())
				goto done;
			break;
		}
	}
done:
	*start = i;
	return removeBlanks(ttype.substr(s, i - s));
}

static Reg identifier("[[:alpha:]_][[:alpha:]_:[:digit:]]*");
static Reg cvtPrefix(
	"^(inline|static|extern|explicit|struct|class)\\b[[:space:]]*",
	Reg::EXTENDED
);
static Reg constPrefix("^const[[:space:]]*", Reg::EXTENDED);

CvtType::CvtType(const Decl *scope, const string &cpp_type, bool noneOk):
	implicitAddress(false), cacheMe(false), aptNeedCheck(false),
	aptRequired(false)
	// pyToCppPattern is needed iff aptFormat_ is not 'O'
{
	// cpp_type must have extra blanks removed already
	cppType_ = cpp_type;

	string fullType(qualify(scope, cppType_));
	// strip off leading explicit/inline/static/struct/class
	for (;;) {
		string::size_type len;
		if (cvtPrefix.find(fullType, &len) != 0)
			break;
		fullType.replace(0, len, "");
	}

	bool isReference = (*fullType.rbegin() == '&');
	if (isReference) {
		implicitAddress = true;
		fullType.erase(fullType.size() - 1);
		for (;;) {
			string::size_type len;
			if (constPrefix.find(fullType, &len) != 0)
				break;
			fullType.replace(0, len, "");
		}
	}

	// get base type (follow typedef's, const's, remove reference)
	type_ = baseType(scope, cpp_type);
	bool notAPointer = *type_.rbegin() != '*';

	if (*type_.rbegin() == ']') {
		implicitAddress = true;
		string::size_type s = type_.find("[");
		string value = type_.substr(0, s);
		string arrayLen = type_.substr(s + 1, type_.size() - 2 - s);
		declPattern = value + " @" + type_.substr(s);
		CvtType valueArg(scope, value);
		cacheMe = valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PySequence_Check(@)";
		if (!arrayLen.empty())
			typeCheckPattern = "(" + typeCheckPattern
				+ " && PySequence_Size(@) == " + arrayLen + ")";
		bvFormat_ = 'N';
		bvPattern = "wrappy::cvtSequenceToPyTuple(&@[0], &@" + type_.substr(s) + ")";
		python_type = "tuple of " + valueArg.pythonType();
		if (!arrayLen.empty())
			python_type = arrayLen + "-" + python_type;
		return;
	}

	// first, special case pointer types
	if (type_ == "char*") {
		// TODO: need to differeniate between char* and const char*
		if (noneOk) {
			pyToCppPattern = "@ == Py_None ? NULL : ";
			pyToCppPattern += "PyUnicode_AsUTF8(@)";
			aptType_ = "const char*";
			aptFormat_ = 'z';
			aptToCppPattern = '@';
			// cppToAptPattern = bvPattern;
			typeCheckPattern += "PyUnicode_Check(@)";
			typeCheckPattern = "(@ == Py_None || "
						+ typeCheckPattern + ')';
			bvFormat_ = 'z';
			bvPattern = '@';
			//pymemberdef_type = "T_STRING"; can only get, not set
			python_type = "(str|None)";
		} else {
			cppType_ = "std::string";
			pyToCppPattern = "wrappy::PythonUnicode_AsCppString(@)";
			aptType_ = "PyObject*";
			aptFormat_ = 'O';
			aptNeedCheck = true;
			// aptToCppPattern = pyToCppPattern;
			cppToActualPattern = "@.c_str()";
			// cppToAptPattern = bvPattern;
			typeCheckPattern = "PyUnicode_Check(@)";
			bvFormat_ = 'z';
			bvPattern = "@";
			python_type = "str";
		}
		return;
	}
	if (type_ == "signed char*" || type_ == "unsigned char*") {
		// TODO: treat as a bytes
		pyToCppPattern = "?TODO?";
		aptRequired = true;
		aptType_ = "Py_buffer";
		if (noneOk)
			aptFormat_ = "z*";
		else
			aptFormat_ = "s*";
		aptNeedCheck = false;
		aptCleanupPattern = "PyBuffer_Release(&@)";
		aptToCppPattern = "reinterpret_cast<" + type_ + ">(@.buf)";
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "?TODO?";
		bvFormat_ = '?';
		bvPattern = "?TODO?";
		python_type = "bytes";
		return;
	}
	if (type_ == "PyObject*" || type_ == "_object*") {
		// bypass wrappy argument conversions
		pyToCppPattern = '@';
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "true";
		bvFormat_ = 'N';
		bvPattern = '@';
		pymemberdef_type = "T_OBJECT_EX"; // NULL not allowed
		python_type = "object";
		return;
	}
	if (type_ == "PyArrayObject*") {
		// bypass wrappy argument conversions
		pyToCppPattern = '@';
		aptType_ = "PyArrayObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PyArrayObject_Check(@)";
		bvFormat_ = 'N';
		bvPattern = '@';
		python_type = "Numeric.array";
		return;
	}

	if (type_ == "void*") {
		pyToCppPattern = "PyCapsule_AsVoidPtr(@)";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PyCapsule_Check(@)";
		bvFormat_ = 'N';
		bvPattern = "PyCapsule_New(@, NULL, NULL)";
		python_type = "PyCObject";
		return;
	}
	// TODO: if still a pointer or an array
	//	strip pointer-ness, recurse, and reassemble

	if (type_ == "signed char" || type_ == "unsigned char") {
		// TODO: limit 0 <= unsigned <= 255
		pyToCppPattern = "static_cast<" + type_ + ">(PyLong_AsLong(@))";
		aptType_ = "char";
		aptFormat_ = 'b';
		aptToCppPattern = '@';
		cppToAptPattern = "static_cast<" + aptType_ + ">(@)";
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'b';
		bvPattern = '@';
		if (type_ == "signed char") {
			pymemberdef_type = "T_BYTE";
			python_type = "byte";
		} else {
			pymemberdef_type = "T_UBYTE";
			python_type = "unsigned byte";
		}
		return;
	}
	if (type_ == "char") {
		// To get C/C++ char semantics of a null character being
		// false, we use an empty string for the null character.
		pyToCppPattern = "wrappy::PythonUnicode_AsCChar(@)"; // has trailing null
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PyUnicode_Check(@)";
		bvFormat_ = 'N';
		bvPattern = "wrappy::pyObject<char>(@)";
		pymemberdef_type = "T_CHAR";
		python_type = "str";
		return;
	}
	if (type_ == "int") {
		pyToCppPattern = "static_cast<int>(PyLong_AsLong(@))";
		aptType_ = "int";
		aptFormat_ = 'i';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'i';
		bvPattern = '@';
		pymemberdef_type = "T_INT";
		python_type = "int";
		return;
	}
	if (type_ == "unsigned" || type_ == "unsigned int") {
		// TODO: limit int >= 0
		pyToCppPattern = "static_cast<unsigned>(PyLong_AsUnsignedLongMask(@))";
		aptType_ = "unsigned";
		aptFormat_ = 'I';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'H';
		bvPattern = '@';
		pymemberdef_type = "T_UINT";
		python_type = "unsigned int";
		return;
	}
	if (type_ == "short" || type_ == "short int") {
		pyToCppPattern = "static_cast<short>(PyLong_AsLong(@))";
		aptType_ = "short int";
		aptFormat_ = 'h';
		aptToCppPattern = '@';
		// aptToCppPattern = pyToCppPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'h';
		bvPattern = '@';
		pymemberdef_type = "T_SHORT";
		python_type = "int";
		return;
	}
	if (type_ == "unsigned short" || type_ == "unsigned short int") {
		// TODO: limit int >= 0
		pyToCppPattern = "static_cast<unsigned short>(PyLong_AsLong(@))";
		aptType_ = "unsigned short";
		aptFormat_ = 'h';
		aptToCppPattern = '@';
		// aptToCppPattern = pyToCppPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'H';
		bvPattern = '@';
		pymemberdef_type = "T_USHORT";
		python_type = "unsigned int";
		return;
	}
	if (type_ == "long" || type_ == "long int") {
		pyToCppPattern = "PyLong_AsLong(@)";
		aptType_ = "long";
		aptFormat_ = 'l';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'l';
		bvPattern = '@';
		pymemberdef_type = "T_LONG";
		python_type = "int";
		return;
	}
	if (type_ == "unsigned long" || type_ == "unsigned long int") {
		pyToCppPattern = "PyLong_AsUnsignedLongMask(@)";
		aptType_ = "unsigned long";
		aptFormat_ = 'k';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'k';
		bvPattern = '@';
		pymemberdef_type = "T_ULONG";
		python_type = "unsigned int";
		return;
	}
	if (type_ == "long long") {
		pyToCppPattern = "PyLong_AsLongLong(@)";
		aptType_ = "long long";
		aptFormat_ = 'L';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'L';
		bvPattern = '@';
		python_type = "long";
		return;
	}
	if (type_ == "unsigned long long") {
		pyToCppPattern = "PyLong_AsUnsignedLongLongMask(@)";
		aptType_ = "unsigned long long";
		aptFormat_ = 'K';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'K';
		bvPattern = '@';
		python_type = "unsigned long";
		return;
	}
	if (type_ == "bool") {
		pyToCppPattern = "bool(PyLong_AsLong(@))";
		aptType_ = "int";
		aptFormat_ = 'i';
		aptToCppPattern = "bool(@)";
		cppToAptPattern = "int(@)";
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'N';
		bvPattern = "wrappy::pyObject(bool(@))";
		python_type = "bool";
		return;
	}
	if (type_ == "float") {
		pyToCppPattern = "float(PyFloat_AsDouble(@))";
		aptType_ = "float";
		aptFormat_ = 'f';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Float_Check(@)";
		bvFormat_ = 'f';
		bvPattern = '@';
		pymemberdef_type = "T_FLOAT";
		python_type = "float";
		return;
	}
	if (type_ == "double") {
		pyToCppPattern = "PyFloat_AsDouble(@)";
		aptType_ = "double";
		aptFormat_ = 'd';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "wrappy::Float_Check(@)";
		bvFormat_ = 'd';
		bvPattern = '@';
		pymemberdef_type = "T_DOUBLE";
		python_type = "float";
		return;
	}
	if (type_ == "ptrdiff_t" || type_ == "ssize_t" || type_ == "Py_ssize_t") {
		pyToCppPattern = "static_cast<" + type_ + ">(PyLong_AsSsize_t(@))";
		aptType_ = type_;
		aptFormat_ = 'n';
		aptToCppPattern = '@';
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PyIndex_Check(@)";
		bvFormat_ = 'n';
		bvPattern = '@';
		python_type = "ssize_t";
		return;
	}
	if (type_.compare(0, 5, "enum ") == 0) {
		cppType_ = type_.substr(5);
		pyToCppPattern = "static_cast<" + cppType_ + ">(PyLong_AsLong(@))";
		aptType_ = "int";
		aptFormat_ = 'i';
		aptToCppPattern = "static_cast<" + cppType_ + ">(@)";
		cppToAptPattern = '@';
		typeCheckPattern = "wrappy::Long_Check(@)";
		bvFormat_ = 'i';
		bvPattern = '@';
		python_type = "int";	// TODO: may have enum types
		return;
	}
	if (type_ == "std::string" || type_ == "string") {
		// C++ strings preserve length in the presense of null bytes
		pyToCppPattern = "wrappy::PythonUnicode_AsCppString(@)";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PyUnicode_Check(@)";
		bvFormat_ = 'N';
		bvPattern = "wrappy::pyObject(@)";
		python_type = "str";
		return;
	}
#if 0
	if (type_ == "otf::Symbol" || type_ == "Symbol") {
		pyToCppPattern = "otf::Symbol(wrappy::PythonUnicode_AsCppString(@))";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "PyUnicode_Check(@)";
		bvFormat_ = 'N';
		bvPattern = "PyUnicode_DecodeUTF8(@.str().data(), @.size(), \"replace\")";
		python_type = "str";
		return;
	}
#endif
	if (type_ == "std::istream" || type_ == "istream") {
		cppType_ = "wrappy::pyistream";
		pyToCppPattern = "wrappy::pyistream(@)";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		typeCheckPattern = "wrappy::PythonFileLike_Check(@, std::ios::in)";
		// TODO: support returning C++ streams
		// bvFormat = ?
		// bvPattern = ?
		pymemberdef_type = "T_OBJECT_EX"; // NULL not allowed
		python_type = "readable file-like";
		return;
	}
	if (type_ == "std::ostream" || type_ == "ostream") {
		cppType_ = "wrappy::pyostream";
		pyToCppPattern = "wrappy::pyostream(@)";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		typeCheckPattern = "wrappy::PythonFileLike_Check(@, std::ios::out)";
		// TODO: support returning C++ streams
		// bvFormat = ?
		// bvPattern = ?
		pymemberdef_type = "T_OBJECT_EX"; // NULL not allowed
		python_type = "writable file-like";
		return;
	}
	if (type_ == "std::iostream" || type_ == "iostream") {
		cppType_ = "wrappy::pyiostream";
		pyToCppPattern = "wrappy::pyiostream(@)";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		typeCheckPattern = "wrappy::PythonFileLike_Check(@, std::ios::in|std::ios::out)";
		// TODO: support returning C++ streams
		// bvFormat = ?
		// bvPattern = ?
		pymemberdef_type = "T_OBJECT_EX"; // NULL not allowed
		python_type = "file-like";
		return;
	}

	if (type_ == "std::complex<float>" || type_ == "complex<float>"
	|| type_ == "std::complex<double>" || type_ == "std::complex<double>") {
		pyToCppPattern = type_ + "(PyComplex_RealAsDouble(@), PyComplex_ImagAsDouble(@))";
		aptType_ = "Py_complex";
		aptFormat_ = 'D';
		aptToCppPattern = type_ + "(@.real, @.imag)";
		cppToAptPattern = "wrappy::makePy_complex(@.real(), @.imag())";
		typeCheckPattern = "PyComplex_Check(@)";
		bvFormat_ = 'N';
		bvPattern = "PyComplex_FromDoubles(@.real(), @.imag())";
		python_type = "complex";
		return;
	}
	if (type_.compare(0, 10, "std::pair<") == 0
	|| type_.compare(0, 5, "pair<") == 0) {
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "?TODO?";
		// bvFormat_ and bvPatteren are figured out below
		string::size_type s = 0;
		string first = getSubType(type_, &s);
		string second = getSubType(type_, &s);
		string::size_type len = first.rfind("::");
		if (first != second || first.size() < 11 || len == string::npos
		|| !(first.substr(first.size() - 8) == "iterator")) {
			CvtType firstArg(scope, first);
			CvtType secondArg(scope, second);
			cacheMe = firstArg.cache() || secondArg.cache();
			typeCheckPattern
				= "PyTuple_Check(@) && PyTuple_Size(@) == 2"
				" && "
				+ firstArg.typeCheck("PyTuple_GetItem(@, 0)");
				+ " && "
				+ secondArg.typeCheck("PyTuple_GetItem(@, 1)");
			// TODO: pyToCppPattern
			bvFormat_ = "(" + firstArg.bvFormat()
						+ secondArg.bvFormat() + ")";
			bvPattern = firstArg.bvArg("@.first") + ", "
						+ secondArg.bvArg("@.second");
			python_type = "tuple(" + firstArg.pythonType()
					+ ", " + secondArg.pythonType() + ")";
			return;
		}
		// ::.*iterator
		bvFormat_ = 'N';
		CvtType subType(scope, first.substr(0, len));
		cacheMe = subType.cache();
		const string &tmp = subType.baseCppType();
		if (tmp.compare(0, 9, "std::map<") == 0
		|| tmp.compare(0, 4, "map<") == 0
		|| tmp.compare(0, 14, "std::unordered_map<") == 0
		|| tmp.compare(0, 9, "unordered_map<") == 0) {
			string::size_type s = 0;
			string key = getSubType(tmp, &s);
			CvtType keyArg(scope, key);
			string value = getSubType(tmp, &s);
			CvtType valueArg(scope, value);
			cacheMe = keyArg.cache() || valueArg.cache();
			bvPattern = "wrappy::cvtMapToPyDict(@.first, @.second)";

			python_type = "mapping of (" + keyArg.pythonType()
					+ ", " + valueArg.pythonType() + ")";
			return;
		}
		if (tmp.compare(0, 14, "std::multimap<") == 0
		|| tmp.compare(0, 9, "multimap<") == 0
		|| tmp.compare(0, 19, "std::unordered_multimap<") == 0
		|| tmp.compare(0, 14, "unordered_multimap<") == 0) {
			string::size_type s = 0;
			string key = getSubType(tmp, &s);
			CvtType keyArg(scope, key);
			string value = getSubType(tmp, &s);
			CvtType valueArg(scope, value);
			cacheMe = keyArg.cache() || valueArg.cache();
			bvPattern = "wrappy::cvtMultiMapToPyDict(@.first, @.second)";
			python_type = "dict of (" + keyArg.pythonType()
				+ ", list of " + valueArg.pythonType() + "))";
			return;
		}
		/*string::size_type*/ s = 0; string value = getSubType(tmp, &s);
		CvtType valueArg(scope, value);
		cacheMe = valueArg.cache();
		bvPattern = "wrappy::cvtSequenceToPyList(@.first, @.second)";
		python_type = "sequence of " + valueArg.pythonType();
		return;
	}

	if (type_.compare(0, 9, "std::map<") == 0
	|| type_.compare(0, 4, "map<") == 0
	|| type_.compare(0, 14, "std::unordered_map<") == 0
	|| type_.compare(0, 9, "unordered_map<") == 0) {
		string::size_type s = 0;
		string key = getSubType(type_, &s);
		CvtType keyArg(scope, key);
		string value = getSubType(type_, &s);
		CvtType valueArg(scope, value);
		cacheMe = keyArg.cache() || valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "?TODO?";
		bvFormat_ = 'N';
		bvPattern = "wrappy::cvtMapToPyDict(@.begin(), @.end())";
		python_type = "mapping of (" + keyArg.pythonType() + ", "
						+ valueArg.pythonType() + ")";
		return;
	}

	if (type_.compare(0, 14, "std::multimap<") == 0
	|| type_.compare(0, 9, "multimap<") == 0
	|| type_.compare(0, 19, "std::unordered_multimap<") == 0
	|| type_.compare(0, 14, "unordered_multimap<") == 0) {
		string::size_type s = 0;
		string key = getSubType(type_, &s);
		CvtType keyArg(scope, key);
		string value = getSubType(type_, &s);
		CvtType valueArg(scope, value);
		cacheMe = keyArg.cache() || valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		typeCheckPattern = "?TODO?";
		bvFormat_ = 'N';
		bvPattern = "wrappy::cvtMultiMapToPyDict(@)";
		python_type = "dict of (" + keyArg.pythonType()
				+ ", list of " + valueArg.pythonType() + "))";
		return;
	}

	if (type_.compare(0, 9, "std::set<") == 0
	|| type_.compare(0, 4, "set<") == 0
	|| type_.compare(0, 19, "std::unordered_set<") == 0
	|| type_.compare(0, 14, "unordered_set<") == 0) {
		// TODO? convert to dictionary keys?
		string::size_type s = 0;
		string value = getSubType(type_, &s);
		CvtType valueArg(scope, value);
		cacheMe = valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "?TODO?";
		bvFormat_ = 'N';
		// TODO: use a special type instead of a Python list
		bvPattern = "wrappy::cvtSetToPySet(@.begin(), @.end())";
		python_type = "set of " + valueArg.pythonType();
		return;
	}

	if (type_.compare(0, 9, "std::multiset<") == 0
	|| type_.compare(0, 4, "multiset<") == 0
	|| type_.compare(0, 19, "std::unordered_multiset<") == 0
	|| type_.compare(0, 14, "unordered_multiset<") == 0) {
		// TODO? convert to dictionary keys?
		string::size_type s = 0;
		string value = getSubType(type_, &s);
		CvtType valueArg(scope, value);
		cacheMe = valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "?TODO?";
		bvFormat_ = 'N';
		// TODO: use a special type instead of a Python list
		bvPattern = "wrappy::cvtSequenceToPyList(@.begin(), @.end())";
		python_type = "list of " + valueArg.pythonType();
		return;
	}

	if (type_.compare(0, 10, "std::list<") == 0
	|| type_.compare(0, 5, "list<") == 0
	|| type_.compare(0, 11, "std::slist<") == 0
	|| type_.compare(0, 6, "slist<") == 0
	|| type_.compare(0, 11, "std::deque<") == 0
	|| type_.compare(0, 6, "deque<") == 0
	|| type_.compare(0, 12, "std::vector<") == 0
	|| type_.compare(0, 7, "vector<") == 0) {
		string::size_type s = 0;
		string value = getSubType(type_, &s);
		CvtType valueArg(scope, value);
		cacheMe = valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ = 'O';
		aptNeedCheck = true;
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		typeCheckPattern = "?TODO?";
		bvFormat_ = 'N';
		bvPattern = "wrappy::cvtSequenceToPyList(@.begin(), @.end())";
		python_type = "list of " + valueArg.pythonType();
		return;
	}

#if 0
	if (type_.compare(0, 12, "otf::AArray<") == 0
	|| type_.compare(0, 7, "AArray<") == 0
	|| type_.compare(0, 11, "otf::Array<") == 0
	|| type_.compare(0, 6, "Array<") == 0) {
		// TODO: convert to/from numeric arrays
		string::size_type s = 0;
		string value = getSubType(type_, &s);
		CvtType valueArg(scope, value);
		string arrayLen = getSubType(type_, &s);
		cacheMe = valueArg.cache();
		pyToCppPattern = "?TODO?";
		aptType_ = "PyObject*";
		aptFormat_ ='O';
		// aptToCppPattern = pyToCppPattern;
		// cppToAptPattern = bvPattern;
		aptNeedCheck = true;
		typeCheckPattern = "(PySequence_Check(@) && PySequence_Size(@) == " + arrayLen + ")";
		bvFormat_ = 'N';
		bvPattern = "otf::cvtSequenceToPyTuple(@.begin(), @.end())";
		python_type = arrayLen + "-tuple of " + valueArg.pythonType();
		return;
	}
#endif

	// assumed wrapped class
	pyToCppPattern = "?TODO?";
	aptType_ = "PyObject*";
	aptFormat_ = 'O';
	aptNeedCheck = true;
	// aptToCppPattern = pyToCppPattern;
	// cppToAptPattern = bvPattern;
	typeCheckPattern = "?TODO?";
	bvFormat_ = 'N';
	bvPattern = "?TODO?";
	string::size_type len;
	string::size_type start = identifier.find(type_, &len);
	if (start == string::npos)
		return;
	Symbol id(type_.substr(start, len));
	bvPattern = "wrappy::pyObject<" + fullType + ">(@)";
	typeCheckPattern = "wrappy::Type<" + id.str() + ">::check(@";
	if (noneOk)
		typeCheckPattern += ", true";
	typeCheckPattern += ")";
	const ClassInfo *ci = NULL;
	SymCDMap::iterator temp = classDeclsMap.find(id);
	if (temp != classDeclsMap.end()) {
		CDCIMap::iterator x = classInfoMap.find(temp->second);
		if (x != classInfoMap.end())
			ci = x->second;
	}
	string objectName;
	if (ci) {
		objectName = moduleName(ci, "object");
		python_type = pythonName(ci);
		if (noneOk) {
			python_type = "(" + python_type + "|None)";
		}
		if (!ci->dontCache)
			cacheMe = true;
	} else {
		// TODO: what should be really be doing with unknown types?
		// std::ostream can be safely ignored, but what about others?
		string::size_type i = id.str().rfind("::");
		if (i == string::npos)
			objectName = id.str();
		else
			objectName = id.str().substr(i + 2);
		python_type = "__unknown__";
		objectName += "_object";
		if (type_ != "std::ostream")
			std::cerr << programName << ": unable to wrap: \""
							<< type_ << "\"\n";
		return;
	}
	if (!ci->isBaseClass) {
		if (notAPointer)
			pyToCppPattern = "*";
		else if (noneOk)
			pyToCppPattern = "@ == Py_None ? NULL : ";
		else
			pyToCppPattern = "";
		string getInst = moduleScopeName(ci, "getInst");
		pyToCppPattern += getInst + "(static_cast<" + objectName + "*>(@))";
		return;
	}
	if (ci->isWrappySubclass) {
		// && ci->isBaseClass
		if (notAPointer)
			pyToCppPattern = "*";
		else if (noneOk)
			pyToCppPattern = "@ == Py_None ? NULL : ";
		else
			pyToCppPattern = "";
		pyToCppPattern += "dynamic_cast<" + id.str()
			+ "*>(static_cast<wrappy::Object*>(@)->_inst_data)";
		return;
	}
	if (derivedClasses.find(id) != derivedClasses.end()) {
		// id is a base class and we're generating a Python type
		std::pair<DerivedClasses::iterator, DerivedClasses::iterator>
					bounds = derivedClasses.equal_range(id);
		aptNeedCheck = true;
		typeCheckPattern = "(";
		if (notAPointer)
			pyToCppPattern = "*(";
		else if (noneOk)
			pyToCppPattern = "@ == Py_None ? NULL : ";
		else
			pyToCppPattern = "";
		for (DerivedClasses::iterator i = bounds.first;
						i != bounds.second; ++i) {
			if (i != bounds.first) {
				typeCheckPattern += " || ";
				pyToCppPattern += " : ";
			}
			Symbol klass = i->second;
			typeCheckPattern += "wrappy::Type<" + klass.str()
								+ ">::check(@)";
			string getInst = /*moduleScopeName(klass,*/ "getInst"/*)*/;
			pyToCppPattern += "wrappy::Type<" + klass.str()
				+ ">::check(@) ? static_cast<" + type_
				+ ">(" + getInst + "(static_cast<" + klass.str()
				+ "_object*>(@)))";
		}
		typeCheckPattern += ")";
		pyToCppPattern += " : NULL";
		if (notAPointer)
			pyToCppPattern += ")";
		return;
	} else {
		std::cerr << "missing derived type information for base class: " << id << '\n';
		return;
	}
}

string
CvtType::cppTypeDecl(const std::string &arg, bool noReference) const
{
	string tmp(declPattern);
	if (tmp.empty()) {
		// tmp = cppType_ + " @";
		if (!noReference || *cppType_.rbegin() != '&') {
			return cppType_ + ' ' + arg;
		}
		tmp = cppType_.substr(0, cppType_.size() - 1);
		if (tmp.substr(0, 6) == "const ")
			tmp = tmp.substr(6);
		return tmp + ' ' + arg;
	}
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::aptToCpp(const string &arg) const
{
	string tmp(aptToCppPattern);
	if (tmp.empty()) {
		assert(aptType_ == "PyObject*");
		tmp = pyToCppPattern;
	}
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::cppToActual(const string &arg) const
{
	string tmp(cppToActualPattern);
	if (tmp.empty())
		tmp = '@';
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::aptCleanup(const std::string &arg) const
{
	string tmp(aptCleanupPattern);
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::cppToApt(const string &arg) const
{
	string tmp(cppToAptPattern);
	if (tmp.empty())
		tmp = bvPattern;
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::pyToCpp(const string &arg) const
{
	string tmp(pyToCppPattern);
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::bvArg(const string &arg) const
{
	string tmp(bvPattern);
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

string
CvtType::typeCheck(const string &arg) const
{
	string tmp(typeCheckPattern);
	string::size_type x = 0;
	for (;;) {
		x = tmp.find('@', x);
		if (x == string::npos)
			break;
		tmp.replace(x, 1, arg);
		x += arg.size() - 1;
	}
	return tmp;
}

// generic type support

static Decl *
checkBaseClassesForType(const ClassInfo *ci, Symbol id)
{
	for (CIList::const_iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
		Decl *idDecl = (*i)->cd->members.unique(id);
		if (idDecl != NULL
		&& idDecl->dtype != Decl::FUNCTION)
			// constructors are not types
			return idDecl;
		idDecl = checkBaseClassesForType(*i, id);
		if (idDecl != NULL)
			return idDecl;
	}
	return NULL;
}

const Decl *
typeDecl(const Decl *scope, Symbol id)
{
	for (;;) {
		const ClassDecl *cd = NULL;
		const ClassInfo *ci = NULL;
		const NamespaceDecl *nd = NULL;
		const DeclList *decls;
		if (scope == NULL)
			decls = &globalDecls;
		else if ((nd = dynamic_cast<const NamespaceDecl *>(scope)))
			decls = &nd->decls;
		else if ((cd = dynamic_cast<const ClassDecl *>(scope))) {
			decls = &cd->members;
			CDCIMap::iterator x = classInfoMap.find(cd);
			if (x != classInfoMap.end())
				ci = x->second;
		} else
			return NULL;
		const Decl *idDecl = decls->unique(id);
		if (idDecl != NULL && idDecl->dtype != Decl::FUNCTION)
			// constructors are not types
			return idDecl;
		if (ci != NULL) {
			Decl *idDecl = checkBaseClassesForType(ci, id);
			if (idDecl != NULL)
				return idDecl;
		}
		if (nd) {
			// This doesn't find types, but it does make
			// qualify() of namespace functions work.
			DeclList::RAconst_iterator i = decls->rAfind(id);
			if (i != decls->rAend())
				return i->second;
		}
		if (scope == NULL)
			return NULL;
		scope = scope->scope;
	}
}

static DeclList::RAConstRange
checkBaseClassesForFuncVarType(const ClassInfo *ci, Symbol id)
{
	for (CIList::const_iterator i = ci->baseClasses.begin();
					i != ci->baseClasses.end(); ++i) {
		DeclList::RAConstRange idDecls
					= (*i)->cd->members.rAequal_range(id);
		if (idDecls.first != idDecls.second)
			return idDecls;
		idDecls = checkBaseClassesForFuncVarType(*i, id);
		if (idDecls.first != idDecls.second)
			return idDecls;
	}
	DeclList::RAconst_iterator dummy = globalDecls.rAend();
	return DeclList::RAConstRange(dummy, dummy);
}

DeclList::RAConstRange
funcVarTypeDecls(const Decl *scope, string name)
{
	for (;;) {
		string::size_type delim = name.find("::");
		if (delim == string::npos)
			break;
		if (delim == 0)
			scope = NULL;
		else {
			const Decl *symDecl = typeDecl(scope,
						Symbol(name.substr(0, delim)));
			if (symDecl == NULL) {
				DeclList::RAconst_iterator dummy = globalDecls.rAend();
				return DeclList::RAConstRange(dummy, dummy);
			}
			scope = symDecl;
		}
		name = name.substr(delim + 2);
	}
	Symbol id(name);

	for (;;) {
		const ClassDecl *cd = NULL;
		const ClassInfo *ci = NULL;
		const NamespaceDecl *nd = NULL;
		const DeclList *decls;
		if (scope == NULL)
			decls = &globalDecls;
		else if ((nd = dynamic_cast<const NamespaceDecl *>(scope)))
			decls = &nd->decls;
		else if ((cd = dynamic_cast<const ClassDecl *>(scope))) {
			decls = &cd->members;
			CDCIMap::iterator x = classInfoMap.find(cd);
			if (x != classInfoMap.end())
				ci = x->second;
		} else {
			DeclList::RAconst_iterator dummy = globalDecls.rAend();
			return DeclList::RAConstRange(dummy, dummy);
		}
		const DeclList::RAConstRange idDecls = decls->rAequal_range(id);
		if (idDecls.first != idDecls.second) {
			// constructors are not types
			return idDecls;
		}
		if (ci != NULL) {
			DeclList::RAConstRange idDecls
				= checkBaseClassesForFuncVarType(ci, id);
			if (idDecls.first != idDecls.second)
				return idDecls;
		}
		if (scope == NULL) {
			DeclList::RAconst_iterator dummy = globalDecls.rAend();
			return DeclList::RAConstRange(dummy, dummy);
		}
		scope = scope->scope;
	}
}

string
qualify(const Decl *scope, const string &str, bool keepNamespace)
{
	// look through string for identifiers that need to be qualified
	// with class and/or namespace names when used outside of the
	// class/namespace they were defined in.
	string tmp(str);
	string result;
	string::size_type idLen;
	for (;;) {
		string::size_type start = identifier.find(tmp, &idLen);
		if (start == string::npos) {
			result += tmp;
			break;
		}
		string id = tmp.substr(start, idLen);
		result += tmp.substr(0, start);
		tmp.replace(0, start + idLen, "");
		string::size_type delim = id.find("::");
		const Decl *symDecl = typeDecl(scope,
						Symbol(id.substr(0, delim)));
		if (symDecl != NULL)
			for (; symDecl->scope != NULL; symDecl = symDecl->scope)
				id = symDecl->scope->tag.str() + "::" + id;
		if (!keepNamespace && !nameSpace.empty()
		&& id.size() > nameSpace.size()
		&& id.compare(0, nameSpace.size(), nameSpace) == 0
		&& id.compare(nameSpace.size(), 2, "::") == 0)
			id = id.substr(nameSpace.size() + 2);
		result += id;
	}
	return result;
}

string
scopeName(const Decl *scope, const string &sname)
{
	// return full name of scope
	string name;
	for (; scope != NULL; scope = scope->scope)
		name = scope->tag.str() + "::" + name;
	name += sname;

	bool sameNamespace = false;
	if (!nameSpace.empty() && name.size() > nameSpace.size()
	&& name.compare(0, nameSpace.size(), nameSpace) == 0
	&& name.compare(nameSpace.size(), 2, "::") == 0) {
		name = name.substr(nameSpace.size() + 2);
		sameNamespace = true;
	}
	return name;
}

static Reg btPrefix(
	"^(inline|static|extern|explicit|struct|class|const)\\b[[:space:]]*",
	Reg::EXTENDED
);
static Reg btConst(
	"[[:space:]]*\\bconst\\b",
	Reg::EXTENDED
);

string
baseType(const Decl *scope, const string &name)
{
	string tmp(name);
startOver:
	// strip off leading const/explicit/inline/static/struct/class
	for (;;) {
		string::size_type len;
		if (btPrefix.find(tmp, &len) != 0)
			break;
		tmp.replace(0, len, "");
	}
	// strip off embedded const
	for (;;) {
		string::size_type len, where;
		where = btConst.find(tmp, &len);
		if (where == string::npos)
			break;
		tmp.replace(where, len, "");
	}
	// reference doesn't change base type
	if (*tmp.rbegin() == '&')
		tmp.erase(tmp.size() - 1);
	// check if name is a typedef, if so, expand it and start over
	const Decl *curScope = scope;
	string::size_type idLen;
	string::size_type start = identifier.find(tmp, &idLen);
	if (start == string::npos)
		return "??";
	string type(tmp.substr(start, idLen));
	for (;;) {
		if (type.empty())
			return "???";
		string::size_type delim = type.find("::");
		// lookup name in current scope, then enclosing scope
		Symbol sym(type.substr(0, delim));
		const Decl *symDecl = typeDecl(curScope, sym);
		if (delim == string::npos) {
			if (symDecl == NULL)
				break;
			if (symDecl->dtype == Decl::ENUM) {
				tmp.replace(start, 0, "enum ");
				break;
			}
			if (symDecl->dtype != Decl::TYPEDEF)
				break;
			// it's a typedef, so recurse
			tmp.replace(start, idLen, symDecl->text);
			scope = symDecl->scope;
			goto startOver;
		}
		// we have a :: delimitor, so the symbol we just found must
		// be a namespace or class and thus the scope we want to
		// look up the next part in.
		curScope = symDecl;
		type = type.substr(delim + 2);
	}
	return qualify(scope, tmp);
}
