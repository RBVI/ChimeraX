// Copyright (c) 1996-1999 The Regents of the University of California.
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

// $Id: Lex.cpp 37613 2012-10-24 22:23:05Z gregc $

#include "Lex.h"
#include "common.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <string.h>
#ifdef OTF_NO_LOCALE
# include <ctype.h>
#endif

using std::string;

inline bool
isSpace(string::traits_type::int_type c)
{
#ifdef OTF_NO_LOCALE
	return isspace(c);
#else
	return ct.is(ct.space, string::traits_type::to_char_type(c));
#endif
}

inline bool
isDigit(string::traits_type::int_type c)
{
#ifdef OTF_NO_LOCALE
	return isdigit(c);
#else
	return ct.is(ct.digit, string::traits_type::to_char_type(c));
#endif
}

inline bool
isAlpha(string::traits_type::int_type c)
{
#ifdef OTF_NO_LOCALE
	return isalpha(c);
#else
	return ct.is(ct.alpha, string::traits_type::to_char_type(c));
#endif
}

inline bool
isAlnum(string::traits_type::int_type c)
{
#ifdef OTF_NO_LOCALE
	return isalnum(c);
#else
	return ct.is(ct.alnum, string::traits_type::to_char_type(c));
#endif
}

char const *const Lex::tokenNames[] = {
	"start of input",
	"end of input",
	"any token expected"
	"expression",
	"identifier",
	"constant",
	"comment",
	// keywords
	"namespace",
	"class",
	"struct",
	"typedef",
	"enum",
	"using",
	"extern",
	"public",
	"protected",
	"private",
	"virtual",
	"friend",
	"template",
};

const char *Lex::tokenName(Lex::traits_type::int_type t)
{
	if (t >= EOI)
		return tokenNames[t - EOI];
	static char tmp[2];
	tmp[0] = traits_type::to_char_type(t);
	tmp[1] = '\0';
	return tmp;
}

inline Lex::traits_type::int_type
Lex::nextChar()
{
	traits_type::char_type c;
	if (!saved.empty()) {
		c = saved.back();
		saved.pop_back();
		if (c == '\n')
			line_num += 1;
		return traits_type::to_int_type(c);
	}
	f->get(c);
	if (!f->good() || f->eof())
		return traits_type::eof();
	if (c == '\n')
		line_num += 1;
	return c;
}

inline void
Lex::saveChar(Lex::traits_type::int_type c)
{
	if (c == traits_type::eof())
		return;
	if (c == '\n')
		line_num -= 1;
	saved.push_back(traits_type::to_char_type(c));
}

Lex::Lex(const string &fname): line_num(1), t(SOI), foundError(false)
{
	if (fname.empty() || fname == "-") {
		filename = Symbol("standard input");
		f = &std::cin;
		return;
	}

	filename = Symbol(fname);
	f = new std::ifstream(fname.c_str());
	if (f == NULL || f->fail()) {
		foundError = true;
		std::cerr << programName << " error: unable to open " << fname
							<< " for reading\n";
		t = EOI;
	}
}

Lex::Lex(Symbol fname, int lineNum, const string &input):
	filename(Symbol(fname)), line_num(lineNum), t(SOI),
	foundError(false)
{
	f = new std::istringstream(input);
}

bool
Lex::nextToken(Lex::traits_type::int_type tok, bool skipComments)
{
	Lex::traits_type::int_type c;
	if (f == NULL)
		return false;

	// skip leading whitespace
	tstr = string();
	leading = string();
	for (c = nextChar(); c != traits_type::eof() && isSpace(c);
							c = nextChar()) {
		if (c == '\n') {
			if (leading.length() == 1 && leading[0] == '\n')
				// collapse multiple leading lines into one
				continue;
		}
		leading += traits_type::to_char_type(c);
	}
	// look for comment
	if (c == '/') {
		c = nextChar();
		switch (c) {
		  default:
			saveChar(c);
			c = '/';
			break;
		  case '/': {
			// comment is rest of line
			t = COMMENT;
			tstr = "";
			for (c = nextChar(); c != traits_type::eof()
					&& c != '\n' && isSpace(c);
					c = nextChar())
				continue;	// skip leading spaces
			for (; c != traits_type::eof() && c != '\n';
								c = nextChar())
				tstr += traits_type::to_char_type(c);
			if (c == traits_type::eof())
				break;
			// strip trailing spaces
			string::size_type last = tstr.find_last_not_of(SPACES);
			if (last != string::npos)
				tstr.erase(last + 1);
			goto finish_up;
		  }
		  case '*': {
			t = COMMENT;
			tstr = "";
			for (c = nextChar(); c != traits_type::eof()
					&& c != '\n' && isSpace(c);
					c = nextChar())
				continue;	// skip leading spaces
			for (; c != traits_type::eof(); c = nextChar()) {
				if (c == '*') {
					c = f->peek();
					if (c == '/') {
						nextChar();	// skip '/'
						break;
					}
				}
				tstr += traits_type::to_char_type(c);
			}
			// strip trailing spaces
			string::size_type last = tstr.find_last_not_of(SPACES);
			if (last != string::npos)
				tstr.erase(last + 1);
			goto finish_up;
		  }
		}
	}

	if (c == traits_type::eof()) {
		t = EOI;
		goto finish_up;
	}

	if (c == '\'' || c == '"') {
		t = CONSTANT;
		traits_type::int_type start = c;
		tstr = traits_type::to_char_type(c);
		for (c = nextChar(); c != traits_type::eof(); c = nextChar()) {
			tstr += traits_type::to_char_type(c); 
			if (c == '\\') {
				c = nextChar();
				tstr += traits_type::to_char_type(c);
			} else if (c == start)
				break;
		}
		goto finish_up;
	}
	if (isDigit(c)) {
		t = CONSTANT;
		// TODO: hex
		tstr = traits_type::to_char_type(c);
		for (c = nextChar(); c != traits_type::eof() && isDigit(c);
								c = nextChar())
			tstr += traits_type::to_char_type(c); 
		if (c == '.') {
			tstr += traits_type::to_char_type(c);
			for (c = nextChar(); c != traits_type::eof()
						&& isDigit(c); c = nextChar())
				tstr += traits_type::to_char_type(c); 
			if (c == 'e' || c == 'E') {
				tstr += traits_type::to_char_type(c);
				c = nextChar();
				if (c == '-' || c == '+') {
					tstr += traits_type::to_char_type(c);
					c = nextChar();
				}
				for (c = nextChar(); c != traits_type::eof()
					&& isDigit(c); c = nextChar())
					tstr += traits_type::to_char_type(c); 
			}
		}
		if (c == 'f' || c == 'F' || c == 'L' || c == 'U')
			tstr += c;
		else
			saveChar(c);
		goto finish_up;
	}

	if (!isAlpha(c) && c != '_' && c != '~') {
		t = c;
		tstr = traits_type::to_char_type(c);
		goto finish_up;
	}

	tstr = traits_type::to_char_type(c);
	if (tok != EXPRESSION)
		for (c = nextChar(); c != traits_type::eof()
		&& (isAlnum(c) || c == '_' || c == ':' || c == '~');
								c = nextChar())
			tstr += traits_type::to_char_type(c);
	else
		for (c = nextChar(); c != traits_type::eof()
		&& (isAlnum(c) || c == '_' || c == ':' || c == '~'
				|| c == '-' || c == '>' || c == '.' || c == '('
				|| c == ')'); c = nextChar())
			tstr += traits_type::to_char_type(c);
	saveChar(c);
	for (string::size_type i = tstr.length(); i > 1; ) {
		--i;
		if (tstr[i] != ':')
			break;
		saveChar(':');
		tstr.replace(i, 1, string());
	}

	if (tok == IDENT || tok == EXPRESSION)
		t = tok;
	else if (tstr == "namespace")
		t = NAMESPACE;
	else if (tstr == "class")
		t = CLASS;
	else if (tstr == "struct")
		t = STRUCT;
	else if (tstr == "typedef")
		t = TYPEDEF;
	else if (tstr == "enum")
		t = ENUM;
	else if (tstr == "using")
		t = USING;
	else if (tstr == "extern")
		t = EXTERN;
	else if (tstr == "public")
		t = PUBLIC;
	else if (tstr == "protected")
		t = PROTECTED;
	else if (tstr == "private")
		t = PRIVATE;
	else if (tstr == "virtual")
		t = VIRTUAL;
	else if (tstr == "friend")
		t = FRIEND;
	else if (tstr == "template")
		t = TEMPLATE;
	else
		t = IDENT;
finish_up:
	if (std::find(ignoreTokens.begin(), ignoreTokens.end(), tstr)
							!= ignoreTokens.end())
		return nextToken(tok, skipComments);
	if (skipComments && t == COMMENT)
		return nextToken(tok, skipComments);
	if (tok == ANY || tok == t)
		return true;
	foundError = true;

        std::cerr << programName << " error:  " << filename << " line "
		<< line_num << ":  expected " << tokenName(tok) << ", got ";
	if (t == IDENT)
		std::cerr << tstr;
	else
		std::cerr << tokenName(t);
	std::cerr << ", skipping rest of input\n";
	t = EOI;
	return false;
}

string
Lex::snarfText(traits_type::char_type end_char)
{
	traits_type::int_type c;

	if (f == NULL)
		return "";

	int pair;
	int num_pair = 0;
	switch (end_char) {
	default:	pair = -1; break;
	case '}':	pair = '{'; break;
	case ')':	pair = '('; break;
	case '>':	pair = '<'; break;
	}

	string s;
	while (c = nextChar(), c != traits_type::eof()) {
		s += traits_type::to_char_type(c);
		if (num_pair == 0 && c == end_char)
			break;
		if (c == pair)
			num_pair += 1;
		else if (c == end_char)
			num_pair -= 1;
	}
	return s;
}

string
Lex::snarfText(const traits_type::char_type *end_chars)
{
	traits_type::int_type c;

	if (f == NULL)
		return "";

	string s;
	while (c = nextChar(), c != traits_type::eof()) {
		s += traits_type::to_char_type(c);
		if (strchr(end_chars, c) != NULL)
			break;
	}
	return s;
}

string
Lex::getFilename()
{
	traits_type::int_type c;

	if (f == NULL)
		return "";

	for (c = nextChar(); c != traits_type::eof() && isSpace(c);
								c = nextChar())
		continue;

	string s;
	if (c != '"') {
		for (; c != traits_type::eof() && !isSpace(c) && c != ';';
								c = nextChar())
			s += traits_type::to_char_type(c);
		saveChar(c);
	} else {
		c = nextChar();
		for (; c != traits_type::eof() && c != '"'; c = nextChar()) {
			if (c == '\\')
				c = nextChar();
			s += traits_type::to_char_type(c);
		}
	}
	return s;
}
