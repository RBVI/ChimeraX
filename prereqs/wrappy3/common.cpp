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

// $Id: common.cpp 29912 2010-01-28 01:31:29Z gregc $

#include <fstream>
#include <sstream>
#include <limits.h>
#include "common.h"

using std::string;

#ifndef OTF_NO_LOCALE
const std::ctype<std::string::value_type> &ct
	= std::use_facet<std::ctype<string::value_type> >(std::locale::classic());
#endif

const char	SPACES[] = " \f\n\r\t\v";

const char	*programName;
Symbol		singleClass;
string		module("module");
string		nameSpace;
unsigned int	gperfMinimum = UINT_MAX;
bool		noOutput = false;
bool		warnings = false;
bool		wrappyDoc = false;
string		exportTag;
string		alignmentType("double");
StrVec		extraHeaders;
StrVec		ignoreTokens;

string
make_buildvalue(const string &format, const string &args)
{
	// Optimize out Py_BuildValue layer if possible
	if (format == "")
		return "(Py_INCREF(Py_None), Py_None)";
	string tmp = args;
	if (tmp[0] == ',')
		tmp = args.substr(2);
	if (format.size() == 1) {
		switch (format[0]) {
		  case 'N':
			return tmp; // tmp is a PyObject* already
		  case 'b':
		  case 'h':
		  case 'i':
		  case 'l':
			return "PyLong_FromLong(" + tmp + ')';
		  case 'L':
			return "PyLong_FromLongLong(" + tmp + ')';
		  case 'B':
		  case 'H':
		  case 'I':
		  case 'k':
			return "PyLong_FromUnsignedLong(" + tmp + ')';
		  case 'K':
			return "PyLong_FromUnsignedLongLong(" + tmp + ')';
		  case 'f':
		  case 'd':
			return "PyFloat_FromDouble(" + tmp + ')';
		  case 'y':
			return "PyBytes_FromString(" + tmp + ')';
		  case 's':
		  case 'z':
			return "PyUnicode_FromString(" + tmp + ')';
		  case 'n':
			return "PyLong_FromSsize_t(" + tmp + ')';
		}
	}
	return "Py_BuildValue(\"" + format + "\", " + tmp + ')';
}

string
stringize(const string &input)
{
	string output;
	for (string::const_iterator i = input.begin(); i != input.end(); ++i) {
		switch (*i) {
		  default:
			output += *i;
			break;
		  case '\a':
			output += "\\a";
			break;
		  case '\b':
			output += "\\b";
			break;
		  case '\f':
			output += "\\f";
			break;
		  case '\n':
			output += "\\n\\\n";
			break;
		  case '\r':
			output += "\\r";
			break;
		  case '\t':
			output += "\\t";
			break;
		  case '\v':
			output += "\\v";
			break;
		  case '\\':
			output += "\\\\";
			break;
		  case '"':
			output += "\\\"";
			break;
		  case '\0':
			output += "\\000";
			break;
		}
	}
	return output;
}

string
prefixLines(const string &prefix, const string &lines)
{
	string result;
	bool addPrefix = true;
	for (string::const_iterator i = lines.begin(); i != lines.end(); ++i) {
		if (addPrefix)
			result += prefix;
		result += *i;
		addPrefix = (*i == '\n');
	}
	return result;
}

std::auto_ptr<std::ostream>
outputStream(const string &filename)
{
	typedef std::auto_ptr<std::ostream> OSptr;
	if (noOutput)
		// TODO: use "sinkstream" instead of ostringstream below
		return OSptr(new std::ostringstream());
	else
		return OSptr(new std::ofstream(filename.c_str()));
}

string
itostr(int i)
{
	std::ostringstream ost;
	ost << i;
	return ost.str();
}

string
tab(int count)
{
	return string(count, '\t');
}
