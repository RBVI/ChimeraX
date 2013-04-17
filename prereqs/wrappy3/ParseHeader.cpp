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

// $Id: ParseHeader.cpp 37613 2012-10-24 22:23:05Z gregc $

#include "ParseHeader.h"
#include "Lex.h"
#include "common.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <list>
#include <stdexcept>
#include <sstream>
#include "Reg.h"
#include "istream_extras.h"
#ifdef OTF_NO_LOCALE
# include <ctype.h>
#endif
#include <stdlib.h>

using std::string;

struct Token {
	const int	token;
	const string	whitespace;
	const string	tokenStr;
	const int	lineNum;
	Token(int t, const string &w, const string &ts, int l):
			token(t), whitespace(w), tokenStr(ts), lineNum(l) {}
};

struct TList {
	typedef std::list<Token> TL;
	typedef TL::iterator iterator;
	typedef TL::const_iterator const_iterator;
	iterator begin() { return tl.begin(); }
	iterator end() { return tl.end(); }
	const_iterator begin() const { return tl.begin(); }
	const_iterator end() const { return tl.end(); }
	Symbol fileName;
	bool empty() const { return tl.empty(); }
	Token &front() { return tl.front(); }
	const Token &front() const { return tl.front(); }
	void push_back(const Token &t) { tl.push_back(t); }
	void pop_front() { tl.pop_front(); }
	TL::size_type size() const { return tl.size(); }
private:
	TL tl;
};

static Reg	oper("\\boperator[[:space:]]*"
		// three letter operators
		"(>>=" "|<<=" "|->\\*"
		// two letter operators
		"|\\+=" "|-=" "|\\*=" "|/=" "|%=" "|^=" "|&="
		"|\\|=" "|<<" "|>>" "|==" "|!=" "|<=" "|>="
		"|&&" "|\\|\\|" "|\\+\\+" "|--" "|->"
		"|\\[[[:space:]]*\\]" "|\\([[:space:]]*\\)"
		// single letter operators
		"|\\+" "|-" "|\\*" "|/" "|%" "|^" "|&" "|\\|"
		"|~" "|!" "|=" "|<" "|>" "|,"
		// special case new[] and delete[]
		"|new[[:space:]]*\\[[[:space:]]*\\]"
		"|delete[[:space:]]*\\[[[:space:]]*\\]"
		// () and casts (subsumes new and delete)
		"|[[:alpha:]_][^(]*"
		")[[:space:]]*\\(", Reg::EXTENDED);
// identifiers include colons for namespace/class qualified names
static Reg	identSep(
		"~?[[:alpha:]_][[:alpha:]_:~[:digit:]]*[[:space:]]*[;()=[]",
								Reg::EXTENDED);
static Reg	ident("~?[[:alpha:]_][[:alpha:]_:~[:digit:]]*", Reg::EXTENDED);
static Reg	funcConst("[[:space:]]*\\bconst\\b", Reg::EXTENDED);
static Reg	funcThrow("[[:space:]]*\\bthrow[[:space:]]*\\((.*)\\)",
								Reg::EXTENDED);
static Reg	include("[[:space:]]*#[[:space:]]*include[[:space:]]*(.*)[[:space:]]*", Reg::EXTENDED);
static Reg	ifndef("(ifndef[[:space:]]*WrapPy|if[[:space:]]*0)\\b", Reg::EXTENDED);
static Reg	defaultArg("=.*", Reg::EXTENDED);
static Reg	pure("=[[:space:]]*0\\b", Reg::EXTENDED);
static Reg	nullOk("[[:space:]]*(NULL|0)\\b", Reg::EXTENDED);

void
DeclList::append(Decl *d)
{
	inOrder.push_back(d);
	randomAccess.insert(std::pair<const Symbol, Decl *>(d->tag, d));
	if (!d->tag.empty())
		uniqueTags.insert(d->tag);
}

bool
DeclList::has(Symbol name, Access level) const
{
	RAConstRange range = randomAccess.equal_range(name);
	for (DeclMap::const_iterator i = range.first; i != range.second; ++i)
		if (i->second->access == level || level == ANY)
			return true;
	return false;
}

Decl *
DeclList::unique(Symbol name) const
{
	RAConstRange range = randomAccess.equal_range(name);
	if (std::distance(range.first, range.second) != 1)
		return NULL;
	return range.first->second;
}

void
DeclList::remove(Decl *d)
{
	DeclVec::iterator i = std::find(inOrder.begin(), inOrder.end(), d);
	if (i != inOrder.end())
		inOrder.erase(i);

	RARange range = randomAccess.equal_range(d->tag);
	for (RAiterator i = range.first; i != range.second; ++i)
		if (d == i->second) {
			randomAccess.erase(i);
			break;
		}
	if (randomAccess.find(d->tag) == randomAccess.end())
		uniqueTags.erase(d->tag);
}

static const char *semiColonLeftBrace = ";{"; /*}*/

const string &
ClassDecl::str() const
{
	static string tmp;
	if (tag.empty())
		return text;
	tmp = "class " + name.str();
	if (!baseClasses.empty()) {
		const char *prefix = ": ";
		for (DeclList::const_iterator i = baseClasses.begin();
						i != baseClasses.end(); ++i) {
			tmp += prefix + (*i)->str();
			prefix = ", ";
		}
	}
	tmp += " {\n";
	for (DeclList::const_iterator i = members.begin(); i != members.end();
									++i) {
		tmp += (*i)->str() + '\n';
	}
	tmp += "};";
	return tmp;
}

const string &
NamespaceDecl::str() const
{
	static string tmp;
	tmp = "namespace " + tag.str();
	tmp += " {\n";
	for (DeclList::const_iterator i = decls.begin(); i != decls.end();
									++i) {
		tmp += (*i)->str() + '\n';
	}
	tmp += "};";
	return tmp;
}

void
lexToTList(Lex *lex, TList *tl)
{
	tl->fileName = lex->fileName();
	for (lex->nextToken(); lex->token() != Lex::EOI; lex->nextToken())
		tl->push_back(Token(lex->token(), lex->whitespace(),
						lex->str(), lex->lineNum()));
}

// remove all blanks from string

string
removeAllBlanks(const string &str)
{
	string tmp = str;
	for (;;) {
		string::size_type first = tmp.find_first_of(SPACES);
		if (first == string::npos)
			return tmp;
		string::size_type last = tmp.find_first_not_of(SPACES,
								first + 1);
		tmp.replace(first, last - first, "");
	}
}

// collapse contiguious blanks in string into one space

string
collapseBlanks(const string &str)
{
	string tmp = str;
	string::size_type start = 0;
	for (;;) {
		string::size_type first = tmp.find_first_of(SPACES, start);
		if (first == string::npos)
			return tmp;
		string::size_type last = tmp.find_first_not_of(SPACES, first);
		tmp.replace(first, last - first, " ");
		start = first + 1;
	}
}

// remove extraneous blanks and comments from string

string
removeBlanks(const string &str)
{
	string result;
	Lex lex(Symbol("<string>"), 0, str);
	for (lex.nextToken(); lex.token() != Lex::EOI; lex.nextToken()) {
		if (lex.token() == Lex::COMMENT)
			continue;
		const string &tmp = lex.str();
		if (!result.empty())
			if (
#ifdef OTF_NO_LOCALE
			((isalnum(tmp[0]) || tmp[0] == '_')
				&& !ispunct(*result.rbegin()))
#else
			((ct.is(ct.alnum, tmp[0]) || tmp[0] == '_')
				&& !ct.is(ct.punct, *result.rbegin()))
#endif
			|| (tmp == ">" && *result.rbegin() == '>'))
				result += ' ';
		result += tmp;
	}
	return result;
}

static Reg ignorePrefix(
	"^(inline|extern|explicit|virtual|volatile)\\b[[:space:]]*",
	Reg::EXTENDED
);

// Extract type and declaration identifier from a function or variable
// definition.  Assume that any leading enum/class/struct introduces a
// previously delcared type.

// worst case?   void (*func(int arg1) const throw ())(int) const throw ()
// worst case?   void *((*func)(int arg1) throw ())
// worst case?   TYPE(...): initializers

void
extractTagAndType(const string &str, Symbol *tag, string *type,
		bool *function = NULL, string *args = NULL,
		string *except = NULL, bool *isConst = NULL,
		bool *isPure = NULL)
{
	if (tag == NULL || type == NULL)
		throw std::runtime_error("tag and type must be non-NULL");
	*tag = Symbol();
	string tmp = str;
	string::size_type idLen;
	string::size_type idPos = ident.find(tmp, &idLen);

	// look for operator XX
	string::size_type opLen;
	string::size_type opPos = oper.find(tmp, &opLen);
	if (opPos != string::npos) {
		*tag = Symbol(removeBlanks(tmp.substr(opPos, opLen - 1)));
		if (function)
			*function = true;
		string::size_type argStart = opLen;
		int parentCount = 0;
		// extend opLen to include function arguments
		while (opPos + opLen < tmp.length()) {
			if (parentCount == 0 && tmp[opPos + opLen] == ')')
				break;
			switch (tmp[opPos + opLen]) {
			  case '(':
				parentCount += 1;
				break;
			  case ')':
				parentCount -= 1;
				break;
			}
			++opLen;
		}
		if (args)
			*args = tmp.substr(opPos + argStart, opLen - argStart);

		// handle trailing const
		string::size_type clen;
		if (funcConst.find(tmp.substr(opPos + opLen + 1), &clen) == 0) {
			opLen += clen;
			if (isConst)
				*isConst = true;
		}
		// look for trailing "= 0"
		string::size_type len, where;
		if ((where = pure.find(tmp, &len, opPos + opLen + 1)) != string::npos) {
			tmp.replace(where, len, "");
			if (isPure)
				*isPure = true;
		}
		tmp.replace(opPos, opLen + 1, " ");
		// parse "throw (....)"
		if (funcThrow.exec(tmp.c_str())) {
			if (except) {
				Reg::Interval i = funcThrow.match(1);
				*except = tmp.substr(i.start, i.end - i.start);
			}
			Reg::Interval i = funcThrow.match(0);
			tmp.replace(i.start, i.end - i.start, "");
		}
		*tmp.rbegin() = ' ';	// nuke separator
		tmp = removeBlanks(tmp);
		for (;;) {
			string::size_type len;
			if (ignorePrefix.find(tmp, &len) != 0)
				break;
			tmp.replace(0, len, "");
		}
		*type = tmp;
		return;
	}

	// typedef's, function's and variable id's
	// First look to see if we might have a pointer to a function
	// that would be if there is a (...)(...) pattern.
	// TODO: what if instead a template parameter?
	string::size_type paren = tmp.find('(');
	if (paren != string::npos) {
		int parenCount = 1;
		for (;;) {
			string::size_type nextParen
					= tmp.find_first_of("()", paren + 1);
			if (nextParen == string::npos)
				break;
			paren = nextParen;
			switch (tmp[paren]) {
			  case ')':
				--parenCount;
				break;
			  case '(':
				++parenCount;
				if (parenCount == 1)
					idPos = paren;
				break;
			}
			if (parenCount == 0)
				break;
		}
	}

	idPos = identSep.find(tmp, &idLen, idPos);
	if (idPos == string::npos) {
//TODO: what to do
//std::cerr << "unable to find identifier in '" << tmp << "'" << std::endl;
		*tag = Symbol("__UNKNOWN__");
		*type = tmp;
		return;
	}
	bool isFunction = tmp[idPos + idLen - 1] == '(';
	if (function)
		*function = isFunction;
	if (isFunction) {
		string::size_type argStart = idLen;
		int parenCount = 0;
		// extend idLen to include function arguments
		while (idPos + idLen < tmp.length()) {
			if (parenCount == 0 && tmp[idPos + idLen] == ')')
				break;
			if (tmp[idPos + idLen] == '(')
				parenCount += 1;
			else if (tmp[idPos + idLen] == ')')
				parenCount -= 1;
			++idLen;
		}
		if (args)
			*args = tmp.substr(idPos + argStart, idLen - argStart);

		// add trailing const to arguments
		string::size_type clen;
		if (funcConst.find(tmp.substr(idPos + idLen + 1), &clen) == 0) {
			idLen += clen;
			if (isConst)
				*isConst = true;
		}
		// look for ": initializers"
		if (tmp.substr(idPos + idLen + 1, 1) == ":")
			tmp.erase(idPos + idLen + 1);
		// look for "throw (....)" after const
		if (funcThrow.exec(tmp.c_str() + idPos + idLen + 1)) {
			Reg::Interval i = funcThrow.match(0);
			if (i.start == 0) {
				if (except) {
					Reg::Interval j = funcThrow.match(1);
					*except = tmp.substr(j.start + idPos
						+ idLen + 1, j.end - j.start);
				}
				idLen += i.end;
			}
		}
		// remove function arguments
		tmp.replace(idPos + argStart - 1, idLen - argStart + 2, " ");
		// look for trailing "= 0"
		string::size_type len, where;
		if ((where = pure.find(tmp, &len)) != string::npos) {
			tmp.replace(where, len, "");
			if (isPure)
				*isPure = true;
		}
	}
	(void) ident.find(tmp, &idLen, idPos);
	*tag = Symbol(tmp.substr(idPos, idLen));

	tmp.replace(idPos, idLen, " ");
	*tmp.rbegin() = ' ';	// nuke separator
	tmp = removeBlanks(tmp);
	for (;;) {
		string::size_type len;
		if (ignorePrefix.find(tmp, &len) != 0)
			break;
		tmp.replace(0, len, "");
	}
	string::size_type len, where;
	if ((where = defaultArg.find(tmp, &len)) != string::npos) {
		tmp.replace(where, len, "");
	}
	if (!tmp.empty())
		*type = tmp;
	else {
		*type = tag->str();
		*tag = Symbol();
	}
}

void
printError(const Lex *lex, const char *str)
{
	std::cerr << programName << ": " << lex->fileName() << ", line "
				<< lex->lineNum() << ": " << str << std::endl;
}

void
printError(const Lex *lex, const string &str)
{
	std::cerr << programName << ": " << lex->fileName() << ", line "
				<< lex->lineNum() << ": " << str << std::endl;
}

void
parseArguments(Lex *lex, ArgList *al)
{
	string arg;
	Argument a;
	int argCount = 0;
	int parenCount = 0;
	int angleCount = 0;
	TList tl;
	lexToTList(lex, &tl);
	for (TList::iterator i = tl.begin(); i != tl.end(); ++i) {
		if (parenCount != 0 || angleCount != 0) {
			switch (i->token) {
			  case '(':
				parenCount += 1;
				break;
			  case ')':
				parenCount -= 1;
				break;
			  case '<':
				angleCount += 1;
				break;
			  case '>':
				angleCount -= 1;
				break;
			}
			arg += i->whitespace + i->tokenStr;
			continue;
		}
		switch (i->token) {
		  default:
			if (arg.empty())
				arg = i->tokenStr;
			else
				arg += i->whitespace + i->tokenStr;
			break;
		  case '(':
			parenCount += 1;
			arg += i->whitespace + i->tokenStr;
			break;
		  case '<':
			angleCount += 1;
			arg += i->whitespace + i->tokenStr;
			break;
		  case Lex::COMMENT:
			if (i->tokenStr == "IN") {
				a.in = true;
				a.out = false;
			} else if (i->tokenStr == "OUT") {
				a.in = false;
				a.out = true;
			} else if (i->tokenStr == "INOUT") {
				a.in = true;
				a.out = true;
			} else if (i->tokenStr == "NULL_OK") {
				a.null_ok = true;
			} else if (i->tokenStr.substr(0, 3) == "LEN") {
				std::istringstream iss(i->tokenStr.substr(3));
				// parse LENn: name
				iss >> a.array_len >> ':' >> a.array;
				if (!iss)
					a.array_len = -1; // didn't parse
			}
			break;
		  case '=':
			for (++i; i != tl.end(); ++i) {
				if (parenCount == 0 && i->token == ',')
					break;
				switch (i->token) {
				  case '(':
					parenCount += 1;
					break;
				  case ')':
					parenCount -= 1;
					break;
				}
				a.defValue += i->whitespace + i->tokenStr;
			}
			if (i == tl.end())
				goto atEnd;
			// FALLTHROUGH
		  case ',':
			arg += ';';
			extractTagAndType(arg, &a.name, &a.type);
			a.defValue = removeBlanks(a.defValue);
			if (*a.type.rbegin() == '*'
			&& nullOk.matches(a.defValue))
				a.null_ok = true;
			if (a.name.empty())
				printError(lex, "missing name for argument "
							+ itostr(argCount));
			al->push_back(a);
			arg = "";
			a = Argument();
			++argCount;
			continue;
		}
	}
atEnd:
	if (!arg.empty() && arg != "void") {
		arg += ';';
		extractTagAndType(arg, &a.name, &a.type);
		a.defValue = removeBlanks(a.defValue);
		if (*a.type.rbegin() == '*'
		&& nullOk.matches(a.defValue))
			a.null_ok = true;
		if (a.name.empty())
			printError(lex, "missing name for argument "
							+ itostr(argCount));
		al->push_back(a);
	}
}

void
parseExceptions(Lex *lex, std::vector<string> *el)
{
	for (lex->nextToken(); lex->token() != Lex::EOI; lex->nextToken()) {
		if (lex->token() == Lex::COMMENT)
			continue;
		if (lex->token() != Lex::IDENT) {
			printError(lex,
				"expected class name in exception list");
			return;
		}
		el->push_back(lex->str());
		lex->nextToken(Lex::ANY, true);
		if (lex->token() != ',')
			break;
	}
}

void
parseBases(Lex *lex, Access level, ClassDecl *cd)
{
	for (lex->nextToken(); lex->token() != Lex::EOI; lex->nextToken()) {
		if (lex->token() == Lex::LEFT_CURLY)
			break;
		string tmp;
		Access protection = level;
		for (; lex->token() != Lex::IDENT; lex->nextToken()) {
			tmp += lex->whitespace() + lex->str();
			switch (lex->token()) {
			  default:
				printError(lex, "unexpected " + lex->str());
				break;
			  case Lex::COMMENT:
			  case Lex::VIRTUAL:
				break;
			  case Lex::PRIVATE:
				protection = PRIVATE;
				break;
			  case Lex::PROTECTED:
				protection = PROTECTED;
				break;
			  case Lex::PUBLIC:
				protection = PUBLIC;
				break;
			}
		}
		Decl *d = new Decl(Decl::BASE, protection, cd, lex->fileName());
		d->tag = Symbol(lex->str());
		// TODO: handle templated base classes
		d->text = tmp + lex->whitespace() + lex->str();
		d->text += lex->snarfText(",{"); /*}*/
		d->text = removeBlanks(d->text.substr(0, d->text.length() - 1));
		cd->baseClasses.append(d);
	}
}

void
parse(Lex *lex, Access level, DeclList *result, Decl *scope, bool skip,
								Symbol filename)
{
	int preprocDepth = 0;
	int skipDepth = -1;
	for (lex->nextToken(); lex->token() != Lex::EOI; lex->nextToken()) {
		std::auto_ptr<Decl> d;
		switch (lex->token()) {
		  default: {
			printError(lex, "unexpected " + lex->str());
			return;
		  }
		  case '#': {
			std::auto_ptr<Decl> t(new Decl(Decl::PREPROCESSOR,
						level, scope, filename));
			d = t;
			lex->nextToken(Lex::ANY, true);
			string directive = lex->str();
			d->text = directive;
			do {
				d->text += lex->snarfText('\n');
			} while (*(d->text.rbegin() + 1) == '\\');
			if (directive == "if" || directive == "ifdef"
			|| directive == "ifndef")
				preprocDepth += 1;
			else if (directive == "endif")
				preprocDepth -= 1;
			if (!skip) {
				string::size_type len;
				if (ifndef.find(d->text, &len) == 0) {
					skip = true;
					skipDepth = preprocDepth;
				}
			} else if (preprocDepth == skipDepth - 1
			&& directive == "endif")
				skip = false;
			else if (preprocDepth == skipDepth
			&& (directive == "else" || directive == "elif"
						|| directive == "elifdef"))
				skip = false;
			break;
		  }
		  case Lex::ENUM: {
			lex->nextToken(Lex::ANY, true);
			if (lex->token() == Lex::IDENT) {
				std::auto_ptr<Decl> t(new Decl(Decl::ENUM,
						level, scope, filename));
				d = t;
				d->tag = Symbol(lex->str());
				lex->nextToken(Lex::ANY, true);
			} else {
				printError(lex, "missing enum tag");
				d->tag = Symbol("??enum??");
			}
			if (lex->token() == ';') {
				// just a forward declaration
				break;
			}
			if (lex->token() != Lex::LEFT_CURLY) {
				printError(lex, "expected { after enum"/*}*/);
				return;
			}
			for (;;) {
				lex->nextToken(Lex::ANY, true);
				if (lex->token() == Lex::RIGHT_CURLY)
					break;	// allow for trailing ,
				if (lex->token() != Lex::IDENT) {
					printError(lex, "expected identifier"); return;
				}
				VarDecl *vd = new VarDecl(level, scope,
								filename);
				vd->dtype = Decl::CONSTANT;
				vd->tag = Symbol(lex->str());
				vd->type = d->tag.str();
				if (!skip)
					result->append(vd);
				lex->nextToken(Lex::ANY, true);
				if (lex->token() == Lex::RIGHT_CURLY)
					break;
				if (lex->token() == ',')
					continue;
				if (lex->token() != '=') {
					printError(lex, /*{*/"expected } = or ,");
					return;
				}
				int pair = 0;
				for (;;) {
					lex->nextToken(Lex::ANY, true);
					if (lex->token() == '(')
						pair += 1;
					else if (lex->token() == ')')
						pair -= 1;
					else if (lex->token() == ','
					|| lex->token() == Lex::RIGHT_CURLY)
						if (pair == 0)
							break;
				}
				if (lex->token() == Lex::RIGHT_CURLY)
					break;
			}
			if (!lex->nextToken(';', true))
				return;
			break;
		  }
		  case Lex::COMMENT: {
			std::auto_ptr<Decl> t(new Decl(Decl::COMMENT, level,
							scope, filename));
			d = t;
			d->text = lex->str();
			if (include.exec(d->text.c_str())) {
				Reg::Interval i = include.match(1);
				filename = Symbol(d->text.substr(i.start,
							i.end - i.start));
			}
			break;
		  }
		  case Lex::FRIEND: {
			std::auto_ptr<Decl> t(new Decl(Decl::FRIEND, level,
							scope, filename));
			d = t;
			d->text = lex->snarfText(';');
			break;
		  }
		  case Lex::PRIVATE: {
			std::auto_ptr<Decl> t(new Decl(Decl::ACCESS, PRIVATE,
							scope, filename));
			d = t;
			if (level == GLOBAL) {
				printError(lex, "private only in class scope");
				return;
			}
			if (!lex->nextToken(':', true))
				return;
			level = PRIVATE;
			break;
		  }
		  case Lex::PROTECTED: {
			std::auto_ptr<Decl> t(new Decl(Decl::ACCESS, PROTECTED,
							scope, filename));
			d = t;
			if (level == GLOBAL) {
				printError(lex,
					"protected only in class scope");
				return;
			}
			if (!lex->nextToken(':', true))
				return;
			level = PROTECTED;
			break;
		  }
		  case Lex::PUBLIC: {
			std::auto_ptr<Decl> t(new Decl(Decl::ACCESS, PUBLIC,
						scope, filename));
			d = t;
			if (level == GLOBAL) {
				printError(lex, "public only in class scope");
				return;
			}
			if (!lex->nextToken(':', true))
				return;
			level = PUBLIC;
			break;
		  }
		  case Lex::TYPEDEF: {
			std::auto_ptr<Decl> t(new Decl(Decl::TYPEDEF, level,
							scope, filename));
			string tmp = lex->snarfText(';');
			extractTagAndType(tmp, &t->tag, &t->text);
			// don't add redundant typedefs
			DeclList::RAConstRange range
						= result->rAequal_range(t->tag);
			if (range.first == range.second)
				d = t;
			break;
		  }
		  case Lex::USING: {
			// These need to be expanded in later once all of
			// the declarations have been read in.
			std::auto_ptr<Decl> t(new Decl(Decl::USING, level,
							scope, filename));
			d = t;
			d->text = "using ";
			d->text += removeBlanks(lex->snarfText(';'));
			break;
		  }
		  case Lex::CLASS:
		  case Lex::STRUCT: {
			Access defaultAccess = lex->token() == Lex::CLASS
							? PRIVATE : PUBLIC;
			std::auto_ptr<ClassDecl> cd(
				new ClassDecl(lex->token() == Lex::CLASS, level,
							scope, filename));
			if (!lex->nextToken(Lex::IDENT, true))
				return;
			cd->name = Symbol(lex->str());
			cd->text = "class" + lex->whitespace() + lex->str();
			lex->nextToken(Lex::ANY, true);
			if (lex->token() == ';') {
				// ignore forward declarations
				break;
			}
			cd->tag = Symbol(cd->name);
			if (lex->token() == ':') {
				int startLine = lex->lineNum();
				string text = lex->snarfText('{'); /*}*/
				Lex basesLex(lex->fileName(), startLine, text);
				parseBases(&basesLex, defaultAccess, cd.get());
			} else if (lex->token() != Lex::LEFT_CURLY) {
				printError(lex,
				"expected { after class/struct name"); /*}*/
				return;
			}
			int startLine = lex->lineNum();
			string tmp = /*{*/ lex->snarfText('}');
			tmp.erase(tmp.size() - 1);
			Lex classLex(lex->fileName(), startLine, tmp);
			parse(&classLex, defaultAccess, &cd->members, cd.get(),
								skip, filename);
			cd->text += /*{*/ '}';
			lex->nextToken(Lex::ANY, true);
			if (lex->token() != ';') {
				printError(lex, /*{*/ "expected ; after }");
				return;
			}
			std::auto_ptr<Decl> t(cd.release());
			d = t;
			break;
		  }
		  case Lex::NAMESPACE: {
			std::auto_ptr<NamespaceDecl> nd(new
					NamespaceDecl(level, scope, filename));
			lex->nextToken(Lex::ANY, true);
			if (lex->token() == Lex::IDENT) {
				nd->tag = Symbol(lex->str());
				lex->nextToken(Lex::ANY, true);
			}
			if (lex->token() != Lex::LEFT_CURLY) {
				printError(lex, 
				"expected { after namespace [name]"); /*}*/
				return;
			}
			// search for namespace in current scope, and extend
			Decl *previous = result->unique(nd->tag);
			if (previous && previous->dtype != Decl::NAMESPACE)
				previous = NULL;
			int startLine = lex->lineNum();
			string tmp = /*{*/ lex->snarfText('}');
			tmp.erase(tmp.size() - 1);
			Lex nsLex(lex->fileName(), startLine, tmp);
			if (previous == NULL) {
				parse(&nsLex, level, &nd->decls, nd.get(), skip,
								filename);
				std::auto_ptr<Decl> t(nd.release());
				d = t;
			} else {
				NamespaceDecl *ond
				= dynamic_cast<NamespaceDecl *>(previous);
				parse(&nsLex, level, &ond->decls, ond, skip,
								filename);
			}
			break;
		  }
		  case Lex::TEMPLATE: {
			if (!lex->nextToken('<', true))
				return;
			//d = TemplateDecl(....)
			//d->text = lex->whitespace() + lex->str();
			//d->text +=
			lex->snarfText('>');
			lex->nextToken(Lex::ANY, true);
			if (lex->token() == Lex::CLASS
			|| lex->token() == Lex::STRUCT) {
				if (!lex->nextToken(Lex::IDENT, true))
					return;
				lex->nextToken(Lex::ANY, true);
				if (lex->token() == '<') {
					// specialization
					lex->snarfText('>');
					lex->nextToken(Lex::ANY, true);
				}
				if (lex->token() == ';') {
					// ignore forward declarations
					break;
				}
				if (lex->token() == ':') {
					lex->snarfText('{'); /*}*/
				} else if (lex->token() != Lex::LEFT_CURLY) {
					printError(lex,
					"expected { after class/struct name");
					/*}*/
					return;
				}
				/*{*/ lex->snarfText('}');
				lex->nextToken(Lex::ANY, true);
				if (lex->token() != ';') {
					printError(lex,
						/*{*/ "expected ; after }");
					return;
				}
			} else {
				// a function
				string tmp = lex->snarfText(semiColonLeftBrace);
				if (*tmp.rbegin() == '{' /*}*/) {
					// function definition
					/*{*/ lex->snarfText('}');
				}
			}
			break;
		  }
		  case Lex::VIRTUAL: {
			if (level == GLOBAL)
				printError(lex, "virtual at global scope?");
			// ignore virtual
			break;
		  }
		  case Lex::EXTERN: {
			lex->nextToken(Lex::ANY, true);
			if (lex->token() == Lex::CONSTANT) {
				// assume constant is "C"
				if (!lex->nextToken('{', true)) /*}*/
					return;
				int startLine = lex->lineNum();
				string tmp = /*{*/ lex->snarfText('}');
				tmp.erase(tmp.size() - 1);
				Lex eLex(lex->fileName(), startLine, tmp);
				parse(&eLex, level, result, scope, skip,
								filename);
				break;
			}
			if (lex->token() != Lex::IDENT) {
				printError(lex,
					"expected identifier after extern");
				return;
			}
			// FALLTHROUGH
		  }
		  case Lex::IDENT: {
			// function/variable definition
			static const char NoExcept[] = "no exceptions";
			Symbol tag;
			string type, args, except(NoExcept);
			bool function;
			string tmp(lex->whitespace());
			tmp += lex->str();
			int startLine = lex->lineNum();
			tmp += lex->snarfText(semiColonLeftBrace);
			bool isConst = false, isPure = false;
			extractTagAndType(tmp, &tag, &type, &function, &args,
						&except, &isConst, &isPure);
			if (tag.empty())
				// must be constructor/destructor
				tag = Symbol(type);
			if (function) {
				FuncDecl *fd = new FuncDecl(level, scope,
							filename);
				fd->tag = tag;
				fd->returnType = type;
				fd->isConst = isConst;
				fd->isPure = isPure;
				Lex fLex(lex->fileName(), startLine, args);
				parseArguments(&fLex, &fd->args);
				if (except != NoExcept) {
					fd->hasExceptionSpec = true;
					Lex eLex(lex->fileName(), startLine,
									except);
					parseExceptions(&eLex,
							&fd->exceptionSpec);
				}
				std::auto_ptr<Decl> t(fd);
				d = t;
			} else {
				VarDecl *vd = new VarDecl(level, scope,
								filename);
				vd->tag = tag;
				if (type.compare(0, 7, "static ") == 0) {
					// Since we don't support class
					// static variables, assume it is
					// a constant.
					vd->dtype = Decl::CONSTANT;
					vd->type = type.substr(7);
				} else {
					// all variables
					if (*type.rbegin() != '&'
					&& *type.rbegin() != '*')
						// variables are by reference
						type += '&';
					vd->type = type;
				}
				std::auto_ptr<Decl> t(vd);
				d = t;
			}
			if (*tmp.rbegin() == '{' /*}*/) {
				// function definition, variable initialization
				/*{*/ lex->snarfText('}');
				if (!function)
					if (!lex->nextToken(';', true))
						return;
			}
			break;
		  }
		}
		if (d.get() != NULL && !skip)
			result->append(d.release());
	}
}

void
parseHeader(const string &fname, /*OUT*/ DeclList *result)
{
	Lex lex(fname);
	parse(&lex, GLOBAL, result, NULL, false, lex.fileName());
}

void
parseHeader(Symbol fname, const string &input, /*OUT*/ DeclList *result)
{
	Lex lex(fname, 0, input);
	parse(&lex, GLOBAL, result, NULL, false, lex.fileName());
}

// TODO: create another function to expand using directive
