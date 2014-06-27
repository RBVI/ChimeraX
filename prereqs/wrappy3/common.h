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

// $Id: common.h 26655 2009-01-07 22:02:30Z gregc $

#ifndef common_h
# define common_h

# include <string>
# ifndef OTF_NO_LOCALE
#  include <locale>
# endif
# include <ostream>
# include <memory>
# include <vector>
# include "Symbol.h"

extern const char	SPACES[];

extern const char	*programName;
extern Symbol		singleClass;
extern std::string	module;
extern std::string	nameSpace;
extern bool		noOutput;
extern bool		warnings;
extern bool		wrappyDoc;
extern std::string	exportTag;
extern std::string	alignmentType;

typedef std::vector<std::string> StrVec;
extern StrVec		extraHeaders;
extern StrVec		ignoreTokens;

extern std::string	itostr(int i);
extern std::string	tab(int count);

# ifndef OTF_NO_LOCALE
extern const std::ctype<std::string::value_type> &ct;
#endif

extern std::string	make_buildvalue(const std::string &format,
						const std::string &args);
extern std::string	stringize(const std::string &input);
extern std::string	prefixLines(const std::string &prefix,
						const std::string &lines);
extern std::unique_ptr<std::ostream>
			outputStream(const std::string &filename);

struct ClassInfo;
extern std::string	moduleName(const ClassInfo *ci,
				const std::string &suffix, bool external=false);
extern std::string	pythonName(const ClassInfo *ci,
						const std::string &suffix="");
extern std::string	moduleScopeName(const ClassInfo *ci,
				const std::string &suffix, bool external=false);

#endif
