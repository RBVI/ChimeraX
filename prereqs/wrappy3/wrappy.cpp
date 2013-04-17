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

// $Id: wrappy.cpp 34315 2011-08-24 18:25:25Z gregc $

#include "GetOpt.h"
#include "Symbol.h"
#include <string>
#include <iostream>
#include <algorithm>
#include "common.h"
#include "compute.h"
#include "type.h"
#include "module.h"
#include <stdlib.h>
#include <stdio.h>	// for remove
#include <string.h>	// for strrchr

using std::string;

int processInput();

int sizetIsLong = 0;	// 0 == int, 1 == long, 2 == long long

int
main(int argc, char **argv)
{
	std::ios::sync_with_stdio(false); // we don't need no stinking C
	const char *cp;
	if (argc == 0)
		programName = "unknown";
	else if ((cp = ::strrchr(argv[0], '/')) != NULL)
		programName = cp + 1;
	else
		programName = argv[0];

	int opt;
	GetOpt getopt(argc, argv, "a:de:h:i:Lm:N:ns:w");
	while ((opt = getopt()) != -1) switch (opt) {
	  case 'a':
		alignmentType = getopt.arg();
		break;
	  case 'd':
		wrappyDoc = true;
		break;
	  case 'e':
		exportTag = string(getopt.arg()) + ' ';
		ignoreTokens.push_back(getopt.arg());
		break;
	  case 'h':
		extraHeaders.push_back(getopt.arg());
		break;
	  case 'i':
		ignoreTokens.push_back(getopt.arg());
		break;
	  case 'L':
		++sizetIsLong;
		break;
	  case 'm':
		module = string(getopt.arg());
		break;
	  case 'N':
		nameSpace = string(getopt.arg());
		break;
	  case 'n':
		noOutput = true;
		break;
	  case 's':
		singleClass = Symbol(getopt.arg());
		break;
	  case 'w':
		warnings = true;
		break;
	  case '?':
		goto usage;
	}

	if (getopt.index() == argc)
		parseHeader("", &globalDecls);
	else for (int i = getopt.index(); i < argc; i++)
		parseHeader(argv[i], &globalDecls);
	return processInput();
usage:
	std::cerr << "usage: " << argv[0] << " [ options ] [ input-file ]\n"
		"-d\t-- generate wrappy_doc dictionary\n"
		"-e tag\t-- import/export tag for Windows DLL's\n"
		"-h file\t-- add extra header file to module.h\n"
		"-i token\t-- ignore token\n"
		"-L\tsize_t is unsigned long (not int)\n"
		"-LL\tsize_t is unsigned long long\n"
		"-m name\t-- module name\n"
		"-n\t-- no output (just error check input)\n"
		"-N name\t-- namespace name\n"
		"-s name\t-- single class\n"
		"-w\t-- turn on (possibily incorrect) warnings\n";
	return 1;
}

const char intTypes[] = "\
typedef unsigned size_t;\n\
typedef int ssize_t;\n\
typedef int ptrdiff_t;";

const char longTypes[] = "\
typedef unsigned long size_t;\n\
typedef long ssize_t;\n\
typedef long ptrdiff_t;";

const char longlongTypes[] = "\
typedef unsigned long long size_t;\n\
typedef long long ssize_t;\n\
typedef long long ptrdiff_t;";

const char predefines[] = "\
namespace std {\n\
using ::size_t;\n\
using ::ptrdiff_t;\n\
}\n\
\n\
namespace wrappy {\n\
\n\
struct Obj {\n\
};\n\
\n\
}";

int
processInput()
{
	switch (sizetIsLong) {
	  default:
	  case 0:
		parseHeader(Symbol("--size_t--"), intTypes, &globalDecls);
		break;
	  case 1:
		parseHeader(Symbol("--size_t--"), longTypes, &globalDecls);
		break;
	  case 2:
		parseHeader(Symbol("--size_t--"), longlongTypes,
								&globalDecls);
		break;
	}
	parseHeader(Symbol("--predefines--"), predefines, &globalDecls);

	// figure out set of wrapped classes
	computeWrapInformation();

	// if a single class
	if (singleClass == "__module__") {
		if (!dumpModuleCode())
			return 1; return 0;
	}
	if (!singleClass.empty()) {
		const ClassDecl *cd = classDeclsMap[Symbol(singleClass)];
		if (cd == NULL && !nameSpace.empty()) {
			string tmp = nameSpace + "::" + singleClass.str();
			cd = classDeclsMap[Symbol(tmp)];
		}
		if (cd == NULL) {
			std::cerr << programName << ": unable to find class "
							<< singleClass << '\n';
			return 1;
		}
		const ClassInfo *ci = classInfoMap[cd];
		if (ci == NULL || ci->skipClass)
			return 1;
		std::cout << "on class " << cd->name << std::endl;
		if (!dumpTypeHeader(ci) || !dumpTypeCode(ci))
			return 1;
		std::cout << "on module " << module << std::endl;
		if (!dumpModuleCode())
			return 1;
		return 0;
	}
	// foreach wrapped class dump header and code
	for (CDCIMap::iterator i = classInfoMap.begin();
						i != classInfoMap.end(); ++i) {
		const ClassInfo *ci = (*i).second;
		if (ci->skipClass)
			continue;
		std::cout << "on class " << ci->name << std::endl;
		if (!dumpTypeHeader(ci) || !dumpTypeCode(ci))
			return 1;
	}
	std::cout << "on module " << module << std::endl;
	if (!dumpModuleCode())
		return 1;
	return 0;
}
