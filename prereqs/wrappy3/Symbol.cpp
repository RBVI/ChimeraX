// Copyright (c) 1996,1998 The Regents of the University of California.
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

// $Id: Symbol.cpp 26655 2009-01-07 22:02:30Z gregc $

#include "Symbol.h"

#ifndef USE_STD_HASH
# include <set>
#elif defined(__GNUC__)
# include <ext/hash_set>
namespace std {
	using __gnu_cxx::hash;
	using __gnu_cxx::hash_set;
}
#else
# include <hash_set>
#endif
#include <functional>
#include <utility>

using std::string;
typedef const string *StringP;

#ifdef USE_STD_HASH
template <> struct std::hash<StringP> {
	size_t operator()(StringP a) const
	{
		std::hash<string> h;
		return h(*a);
	}
};

template <> struct std::equal_to<StringP> {
	bool operator()(StringP x, StringP y) const { return *x == *y; }
};

#else /* !USE_STD_HASH */

// workaround for GCC 2.95.2
namespace std {
template <> struct less<StringP> {
	bool operator()(StringP a, StringP b) const
	{
		return *a < *b;
	}
};
} // namespace std
#endif

namespace {

#ifdef USE_STD_HASH
typedef std::hash_set<StringP>	StringPSet;
#else
typedef std::set<StringP>	StringPSet;
#endif

string		_nilString;
StringPSet	*sym_table = 0;
}

int Symbol::Init::count = 0;

Symbol::Init::Init()
{
	if (count++ == 0)
		sym_table = new StringPSet;
}

Symbol::Init::~Init()
{
	if (--count == 0) {
#ifdef NO_MEMORY_LEAKS
		delete sym_table;
#else
		;
#endif
	}
}

Symbol::Symbol()
{
	sym = &_nilString;
}

Symbol::Symbol(char c)
{
	if (c == 0) {
		sym = &_nilString;
		return;
	}
	StringP s = new string(1, c);
	std::pair<StringPSet::iterator, bool> p = sym_table->insert(s);
	if (!p.second)
		delete s;
	sym = *p.first;
}

Symbol::Symbol(const char *cp)
{
	if (cp == NULL) {
		sym = &_nilString;
		return;
	}
	StringP s = new string(cp);
	std::pair<StringPSet::iterator, bool> p = sym_table->insert(s);
	if (!p.second)
		delete s;
	sym = *p.first;
}

Symbol::Symbol(const string &orig)
{
	if (orig.empty()) {
		sym = &_nilString;
		return;
	}
	StringP s = &orig;
	StringPSet::iterator i = sym_table->find(s);
	if (i != sym_table->end()) {
		sym = *i;
		return;
	}
	s = new string(orig);
	std::pair<StringPSet::iterator, bool> p = sym_table->insert(s);
	sym = *p.first;
}

unsigned int
Symbol::hash() const
{
#ifdef USE_STD_HASH
	std::hash<StringP> h;
	return h(sym);
#else
	unsigned int result = 0;
	for (string::const_iterator i = sym->begin(); i != sym->end(); ++i)
		result = 5 * result + static_cast<unsigned char>(*i);
	return result;
#endif
}
