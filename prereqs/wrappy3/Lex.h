// Copyright (c) 1996-2000 The Regents of the University of California.
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

// $Id: Lex.h 37613 2012-10-24 22:23:05Z gregc $

#ifndef Lex_h
# define Lex_h

# include <iostream>
# include <vector>
# include <string>
# include "Symbol.h"

class Lex {
public:
	typedef std::string::traits_type traits_type;
	enum Token {
		SOI = 65536, EOI,
		ANY, EXPRESSION,	// ANY and EXPRESSION are not returned
		IDENT, CONSTANT, COMMENT,
		NAMESPACE, CLASS, STRUCT, TYPEDEF, ENUM, USING, EXTERN,
		PUBLIC, PROTECTED, PRIVATE, VIRTUAL, FRIEND, TEMPLATE,
		COLON = ':', SEMICOLON = ';', HASH = '#',
		LEFT_CURLY = '{', RIGHT_CURLY = '}',
		LEFT_PAREN = '(', RIGHT_PAREN = ')'
	};
	explicit Lex(const std::string &fname);
	// the next constructor is for istringstream's
	Lex(Symbol fname, int lineNum, const std::string &input);
			~Lex();
	bool		nextToken(traits_type::int_type tok = ANY,
						bool skipComments = false);
	traits_type::int_type
			token() const;
	const std::string &str() const;
	Symbol		fileName() const;
	int		lineNum() const;
	std::string	snarfText(traits_type::char_type end_char);
	std::string	snarfText(const traits_type::char_type *end_chars);
	std::string	getFilename();
	bool		error() const;
	const std::string &whitespace() const;
	static const char *
			tokenName(traits_type::int_type);
private:
	Lex(const Lex *);		// disable copy constructor
	Lex &operator=(const Lex &);	// disable assignment op
	static char const *const tokenNames[];
	std::istream	*f;
	std::vector<traits_type::char_type> saved;
	Symbol		filename;
	int		line_num;
	std::string	leading;
	std::string	tstr;
	traits_type::int_type		t;
	bool		foundError;
	inline traits_type::int_type	nextChar();
	inline void	saveChar(traits_type::int_type c);
};

inline Lex::~Lex()
{
	if (f != NULL && f != &std::cin)
		delete f;
}

inline Lex::traits_type::int_type
Lex::token() const
{
	return t;
}

inline const std::string &
Lex::str() const
{
	return tstr;
}

inline Symbol
Lex::fileName() const
{
	return filename;
}

inline int
Lex::lineNum() const
{
	return line_num;
}

inline bool
Lex::error() const
{
	return foundError;
}

inline const std::string &
Lex::whitespace() const
{
	return leading;
}

#endif
