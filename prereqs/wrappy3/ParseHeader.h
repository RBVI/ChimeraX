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

// $Id: ParseHeader.h 26655 2009-01-07 22:02:30Z gregc $

#ifndef ParseHeader_h
# define ParseHeader_h

# include "Symbol.h"
# include <map>
# include <set>
# include <vector>
# include <iostream>

enum Access { GLOBAL, PRIVATE, PROTECTED, PUBLIC, ANY };

// a header file is composed of a sequence of declarations
typedef std::set<Symbol> SymbolSet;

class Decl;
class DeclList {
	// pseudo-container for declarations, want to be able to retreive
	// declarations in the order of declaration, and by their tags.
public:
	void append(Decl *);
	void remove(Decl *);
	typedef std::vector<Decl *> DeclVec;
	typedef DeclVec::iterator iterator;
	typedef DeclVec::const_iterator const_iterator;
	iterator begin();
	iterator end();
	const_iterator begin() const;
	const_iterator end() const;
	typedef std::multimap<Symbol, Decl *> DeclMap;
	typedef DeclMap::iterator RAiterator;
	typedef DeclMap::const_iterator RAconst_iterator;
	RAiterator rAbegin();
	RAiterator rAend();
	RAiterator rAlower_bound(Symbol);
	RAiterator rAupper_bound(Symbol);
	typedef std::pair<RAiterator, RAiterator> RARange;
	RARange rAequal_range(Symbol);
	RAconst_iterator rAbegin() const;
	RAconst_iterator rAend() const;
	RAconst_iterator rAlower_bound(Symbol) const;
	RAconst_iterator rAupper_bound(Symbol) const;
	typedef std::pair<RAconst_iterator, RAconst_iterator> RAConstRange;
	RAConstRange rAequal_range(Symbol) const;
	bool empty() const;
	bool has(Symbol name, Access level) const;
	Decl *unique(Symbol name) const;
	RAiterator rAfind(Symbol s);
	RAconst_iterator rAfind(Symbol s) const;
	const SymbolSet &tags() const;
private:
	DeclVec		inOrder;
	DeclMap		randomAccess;
	SymbolSet	uniqueTags;
};

class Decl {
public:
	// enum, typedef, class/struct, variable/function/constant, using,
	// #...
	enum DeclType {
		COMMENT, NAMESPACE, CONSTANT, TYPEDEF, CLASS, STRUCT, FUNCTION,
		VARIABLE, USING, PREPROCESSOR, ACCESS, BASE, ENUM, FRIEND
	};
	Decl(DeclType t, Access a, Decl *s, Symbol fname): dtype(t), access(a), scope(s), filename(fname) {}
	virtual ~Decl() {}
	virtual Decl	*copy() { return new Decl(*this); }
	DeclType	dtype;
	Access		access;
	Decl *const	scope;		// enclosing scope
	Symbol		tag;
	std::string	text;
	Symbol		filename;
	virtual const std::string &str() const { return text; }
	bool operator==(const Decl &d) const {
		return dtype == d.dtype
		&& access == d.access
		&& scope == d.scope
		&& tag == d.tag
		&& filename == d.filename;
	}
};

class Argument {
public:
	Symbol		name;
	std::string	type;
	std::string	defValue;	// default value
	bool		in, out;
	bool		null_ok;
	int		array_len;
	std::string	array;
	Argument(): in(true), out(false), null_ok(false), array_len(-1) {}
	bool operator==(const Argument &a) const {
		return name == a.name
		&& type == a.type
		&& defValue == a.defValue
		&& in == a.in && out == a.out;
	}
	bool operator!=(const Argument &a) const {
		return !operator==(a);
	}
};

typedef std::vector<Argument> ArgList;

class FuncDecl: public Decl {
public:
	FuncDecl(Access a, Decl *s, Symbol f):
		Decl(FUNCTION, a, s, f), isExplicit(false), isVirtual(false),
		isConst(false), isPure(false),
		hasExceptionSpec(false), notMemberFunction(false),
		reverseOp(false), allowThreads(false) {}
	virtual Decl	*copy() { return new FuncDecl(*this); }
	ArgList	args;
	std::string	returnType;
	std::vector<std::string>
			exceptionSpec;
	bool		isExplicit;
	bool		isVirtual;
	bool		isConst;
	bool		isPure;
	bool		hasExceptionSpec;
	mutable bool	notMemberFunction;	// for unaryOp/binOp
	mutable bool	reverseOp;		// reverse binary operator
	mutable bool	allowThreads;
	mutable Symbol	attrName;
	bool operator==(const FuncDecl &d) const {
		return Decl::operator==(d)
		&& args == d.args
		&& returnType == d.returnType
		&& exceptionSpec == d.exceptionSpec
		&& isVirtual == d.isVirtual
		&& hasExceptionSpec == d.hasExceptionSpec
		&& attrName == d.attrName;
	}
};

class VarDecl: public Decl {
public:
	// this is for both CONSTANTs and VARIABLEs
	VarDecl(Access a, Decl *s, Symbol f):
				Decl(VARIABLE, a, s, f) {}
	virtual Decl	*copy() { return new VarDecl(*this); }
	std::string	type;
};

class ClassDecl: public Decl {
public:
	ClassDecl(bool isClass, Access a, Decl *s, Symbol f):
				Decl(isClass ? CLASS : STRUCT, a, s, f),
				hasVirtualFuncs(false),
				hasPublicCopyConstructor(false),
				isWrapPySubclass(false) {}
	virtual Decl	*copy() { return new ClassDecl(*this); }
	Symbol		name;
	DeclList	baseClasses;		// all Other's
	DeclList	members;
	bool		hasVirtualFuncs;	// virtual funcs imply virtual
						// destructor
	bool		hasPublicCopyConstructor;
	bool		isWrapPySubclass;
	virtual const std::string &str() const;
};

class NamespaceDecl: public Decl {
public:
	NamespaceDecl(Access a, Decl *s, Symbol f):
						Decl(NAMESPACE, a, s, f) {}
	virtual Decl	*copy() { return new NamespaceDecl(*this); }
	DeclList	decls;
	virtual const std::string &str() const;
};

inline DeclList::iterator
DeclList::begin()
{
	return inOrder.begin();
}

inline DeclList::iterator
DeclList::end()
{
	return inOrder.end();
}

inline DeclList::const_iterator
DeclList::begin() const
{
	return inOrder.begin();
}

inline DeclList::const_iterator
DeclList::end() const
{
	return inOrder.end();
}

inline DeclList::RAiterator
DeclList::rAbegin()
{
	return randomAccess.begin();
}

inline DeclList::RAiterator
DeclList::rAend()
{
	return randomAccess.end();
}

inline DeclList::RAiterator
DeclList::rAlower_bound(Symbol s)
{
	return randomAccess.lower_bound(s);
}

inline DeclList::RAiterator 
DeclList::rAupper_bound(Symbol s)
{
	return randomAccess.upper_bound(s);
}

inline DeclList::RARange
DeclList::rAequal_range(Symbol s)
{
	return randomAccess.equal_range(s);
}

inline DeclList::RAconst_iterator
DeclList::rAbegin() const
{
	return randomAccess.begin();
}

inline DeclList::RAconst_iterator
DeclList::rAend() const
{
	return randomAccess.end();
}

inline DeclList::RAconst_iterator 
DeclList::rAlower_bound(Symbol s) const
{
	return randomAccess.lower_bound(s);
}

inline DeclList::RAconst_iterator 
DeclList::rAupper_bound(Symbol s) const
{
	return randomAccess.upper_bound(s);
}

inline DeclList::RAConstRange
DeclList::rAequal_range(Symbol s) const
{
	return randomAccess.equal_range(s);
}

inline bool
DeclList::empty() const
{
	return inOrder.empty();
}

inline DeclList::RAiterator
DeclList::rAfind(Symbol s)
{
	return randomAccess.find(s);
}

inline DeclList::RAconst_iterator
DeclList::rAfind(Symbol s) const
{
	return randomAccess.find(s);
}

inline const SymbolSet &
DeclList::tags() const
{
	return uniqueTags;
}

extern void parseHeader(const std::string &fname, /*OUT*/ DeclList *);
extern void parseHeader(Symbol fname, const std::string &input,
						/*OUT*/ DeclList *result);

extern std::string removeBlanks(const std::string &str);

#endif
