typecheck - A runtime typechecking module for Python

typecheck is Python module providing run-time typechecking facilities for
function parameters and return values.

The main workhorses of this module, the functions accepts,
returns and yields, are used as function/method decorators
(see Examples section).

A number of utility classes are provided to assist in building more complex
signatures, for example, by creating boolean expressions based on classes
and/or types.

Note that typechecking can be {en,dis}abled at runtime by toggling the
typecheck.enable_checking global boolean; a value of True (the default) causes
all typechecks (both for parameters and return values) to be run, while False
disables them.

It is possible to incorporate typechecking facilities into user-defined
classes. A mixin class, UnorderedIteratorMixin, is provided to allow easy
typechecking of iterators. See tests/test_examples.py for an illustration of
how to add typechecking capabilities to your classes.

### Install

python setup.py install

or

easy_install typecheck

### Testing

A suite of unit tests are included in the tests/ directory. The test scripts
themselves have filenames prefixed with test_.

The testsuite may be invoked by running `python test.py`. This will gather all
tests/test_*.py files and run them. Alternatively, the test scripts may be run
individually.

### Examples

@accepts( (int, str) )

	Takes a two-tuple composed of an integer and a string


@accepts( [int, int, str] )

	Takes a list, with types in a repeating pattern of integer, integer,
	string.
	The list must complete the pattern; for example, two integers would be
	invalid. For example, [3, 4, 's', 5, 6, 'f'] matches, but [3, 4] and
	[3, 4, 'f', 5, 6] do not because they do not complete the pattern.


@accepts( [(int, str)] )

	Takes a list of int+str two-tuples


@accepts( int, str, f=SomeRandomType )

	Keyword arguments are supported


@accepts( { str : int } )

	Takes a dictionary where all keys must be strings and all values must be
	integers.
	This can also be spelled as @accepts( Dict(str, int) )


@accepts( int, Or(int, str, SomeRandomType) )

	Takes two arguments, the second of which may be either an integer, a
	string or an instance of SomeRandomClass.


@accepts( int, And(ClassA, ClassB, ClassC) )

	Takes two arguments, the second of which must be an instance of a subtype
	of ClassA, ClassB and ClassC


@accepts( Not(int, float) )

	Takes a single argument which can be neither an int or a float.


@accepts( Xor(ClassA, ClassB) )

	The argument must be an instance of either ClassA or ClassB, but not both;
	this would preclude instances of ClassC(ClassA, ClassB).


@accepts( { (int, int): float } )

	The dictionary's keys must be 2-tuples of integers


@accepts( (int, int), int )
def foo((a, b), c):
	...

	Automatically unpacked tuples are also supported


@accepts(int, int, int, int):
def foo(a, b, *vargs, **kwargs):
	...
@accepts(int, int, [int], {str: int})
def foo(a, b, *vargs, **kwargs):
	...
@accepts(int, int, vargs=int, kwargs=int)
def foo(a, b, *vargs, **kwargs):
	...
@accepts(int, int, vargs=[int], kwargs={str: int})
def foo(a, b, *vargs, **kwargs):

	All four of the above signatures are identical. They say that
	there are two mandatory ints (a and b), then all extra positional
	arguments must be ints and all extra keyword arguments must also be
	ints.

	The first and third forms are coerced into the second and fourth
	forms, respectively.


@accepts(int, int)
def foo(a, b):
	return a, b
foo(4, 5.0)

	accepts will raise a TypeCheckError with an error message
	indicating exactly where the badly-typed object is.


@accepts(int, int, int)
def foo(a, b, (c, d)):
	return a, b, c, d

	This will cause accepts to raise a TypeSignatureError, indicating
	that the provided signature does not match the shape of the arguments
	expected by the function.


class Foo:
	@accepts(Self(), int, int)
	@returns(int, int, Self())
	def foo(self, a, b):
		return a, b, self

	A special Self callable may be imported from the typecheck module that
	serves as a placeholder in method signatures.

	Self can also be used with returns.


@accepts(Any(), int, Any())

	This signature indicates that any type of object may be provided for the
	first and third parameters.
	Note the () after Any; its use is similar to that of Or() and And().


@yields(int)
def foo(a):
	yield a+1
	yield a+2
	yield a+3

	yields is a version of returns, but specifically written
	to handle generators. In fact, if you use yields to decorate a
	non-generator, it will blow up.


@accepts(int)
@yields(int)
def foo(a):
	yield a+1
	yield a+2
	yield a+3

	yields can be used in conjunction with accepts


@accepts(IsIterable())
def foo(seq):
	for obj in seq:
		yield obj

	This signature demands that `seq` be of a type valid for use in a for
	statement


@accepts(IsCallable())
def foo(func):
	return func()

	Assert that callable(func) returns True


@accepts(HasAttr(['foo', 'bar']))
def foo(obj):
	...

	Demand that `obj` has attributes `foo` and `bar` without caring about
	their types

@accepts(HasAttr({'foo': int, 'bar': int}))
def foo(obj):
	...

	Insist that `obj` have attributes `foo` and `bar` and that they're both
	integers

@accepts(HasAttr(['foo'], {'bar': int}))
def foo(obj):
	...

	`obj.foo` must be present and `obj.bar` must both be present and be an
	integer


@yields(YieldSeq(int, float, int))
def foo():
	yield 5
	yield 5.0
	yield 5

	The YieldSeq() utility class can be used to specify that the type of the
	generator's return value will change from call to call.


@accepts(Or(Exact(6), Exact(7)))
def foo(a):
	...

	This signature means that foo's `a` parameter must be either 6 or 7. Any
	object can be used with Exact()


### Contributors

Primary contributors:
	+ Collin Winter <collinw@gmail.com>
	+ Iain Lowe <ilowe@cryogen.com>

With ideas/patches from:
	+ Knut Hartmann (doctest tests)
	+ David A. Wheeler (ideas from his own typecheck package)

### TERMS OF USE

This software is copyright (C) 2005-2006 by Collin Winter and Iain Lowe and
is released under the terms of the MIT License.
