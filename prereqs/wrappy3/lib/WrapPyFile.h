// Copyright (c) 2007 The Regents of the University of California.
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
//
// $Id: WrapPyFile.h 35469 2012-02-03 00:06:06Z gregc $

#ifndef WrapPyFile_h
# define WrapPyFile_h

// Allow for Python file objects to read from or written to by
// creating a pyistream or pyostream object that forwards to
// the Python object.

# include <Python.h>
# include <istream>
# include <ostream>
# include <streambuf>

#ifndef PY_STUPID
// workaround for Python API missing const's.
# define PY_STUPID (char*)
#endif

namespace wrappy {

template <class charT, class traits = std::char_traits<charT> >
class basic_stdiobuf: public std::basic_streambuf<charT, traits> {
	FILE* file;
	typedef std::basic_streambuf<charT, traits> streambuf;
	using streambuf::eback;
	using streambuf::egptr;
	using streambuf::gptr;
	using streambuf::pptr;
	using streambuf::pbase;
	using streambuf::pbump;
	using streambuf::setg;
	using streambuf::setp;
public:
	typedef charT char_type;
	typedef typename traits::int_type int_type;
	typedef typename traits::off_type off_type;
	typedef typename traits::pos_type pos_type;
	typedef traits traits_type;

	basic_stdiobuf(FILE* f): file(f) {
		if (file == NULL)
			throw std::invalid_argument("not a file");
	}
	virtual ~basic_stdiobuf() {
	}
protected:
	// output support
	virtual std::streamsize xsputn(const char_type* buf,
							std::streamsize num) {
		// optimization to avoid multiple sputc's
		size_t i;
		Py_BEGIN_ALLOW_THREADS
		i = fwrite(buf, 1, num, file);
		Py_END_ALLOW_THREADS
		if (i != static_cast<size_t>(num))
			return traits_type::eof();
		return num;
	}
	virtual int_type overflow(int_type c) {
		// buffer full
		if (c != traits_type::eof()) {
			// save character into reserved extra slot
			Py_BEGIN_ALLOW_THREADS
			putc(c, file);
			Py_END_ALLOW_THREADS
		}
		return c;
	}
	virtual int sync() {
		Py_BEGIN_ALLOW_THREADS
		if (fflush(file) == EOF) {
			Py_BLOCK_THREADS
			return -1;
		}
		Py_END_ALLOW_THREADS
		return 0;
	}

	// input support
	virtual std::streamsize xsgetn(char_type* buf, std::streamsize num) {
		std::streamsize i;
		Py_BEGIN_ALLOW_THREADS
		i = fread(buf, 1, num, file);
		Py_END_ALLOW_THREADS
		return i;
	}
	virtual int_type underflow() {
		int_type c;
		Py_BEGIN_ALLOW_THREADS
		c = getc(file);
		(void) ungetc(c, file);
		Py_END_ALLOW_THREADS
		return c;
	}
	virtual int_type uflow() {
		int_type c;
		Py_BEGIN_ALLOW_THREADS
		c = getc(file);
		Py_END_ALLOW_THREADS
		return c;
	}
	virtual int_type pbackfail(int_type c) {
		int_type uc;
		Py_BEGIN_ALLOW_THREADS
		uc = ungetc(c, file);
		Py_END_ALLOW_THREADS
		return uc;
	}
};

template <class charT, class traits = std::char_traits<charT> >
class basic_pyfilelikebuf: public std::basic_streambuf<charT, traits> {
	PyObject* readMethod;
	PyObject* writeMethod;
	static const int putback_size = 4;
	static const int buffer_size = 4096;
	charT	buffer[buffer_size + putback_size];
	typedef std::basic_streambuf<charT, traits> streambuf;
	using streambuf::eback;
	using streambuf::egptr;
	using streambuf::gptr;
	using streambuf::pptr;
	using streambuf::pbase;
	using streambuf::pbump;
	using streambuf::setg;
	using streambuf::setp;
public:
	typedef charT char_type;
	typedef typename traits::int_type int_type;
	typedef typename traits::off_type off_type;
	typedef typename traits::pos_type pos_type;
	typedef traits traits_type;

	basic_pyfilelikebuf(PyObject* obj, int mode):
		readMethod(NULL), writeMethod(NULL)
	{
		if ((mode & std::ios::in) != 0) {
			PyObject* o = PyObject_GetAttrString(obj, PY_STUPID "read");
			if (PyCallable_Check(o))
				readMethod = o;
			char_type *pos = buffer + putback_size;
			setg(pos, pos, pos);
		}
		if ((mode & std::ios::out) != 0) {
			PyObject* o = PyObject_GetAttrString(obj, PY_STUPID "write");
			if (PyCallable_Check(o))
				writeMethod = o;
		}
		if (readMethod == NULL && writeMethod == NULL)
			throw std::invalid_argument("not file-like");
	}
	virtual ~basic_pyfilelikebuf() {
	}
protected:
	// output support
	virtual std::streamsize xsputn(const char_type* buf,
							std::streamsize num) {
		// optimization to avoid multiple sputc's
		if (!writeMethod
		|| NULL == PyObject_CallFunction(writeMethod, PY_STUPID "s#", buf, num)) {
			return traits_type::eof();
		}
		return num;
	}
	virtual int_type overflow(int_type c) {
		// buffer full
		if (c != traits_type::eof()) {
			if (!writeMethod
			|| NULL == PyObject_CallFunction(writeMethod, PY_STUPID "c", c)) {
				return traits_type::eof();
			}
		}
		return c;
	}

	// input support
	virtual int_type underflow() {
		if (gptr() < egptr())
			// is read position before end of buffer?
			return traits_type::to_int_type(*gptr());
		int numPutback = gptr() - eback();
		if (numPutback > putback_size)
			numPutback = putback_size;
		std::copy(gptr() - numPutback, gptr(),
					buffer + (putback_size - numPutback));
		// get new characters
#if 0
		int num = read(buffer + putback_size, buffer_size);
		if (num <= 0)
			return traits_type::eof();
#else
#if PY_VERSION_HEX < 0x02050000
		int num = buffer_size;
#else
		Py_ssize_t num = buffer_size;
#endif
		PyObject *s = NULL;
		if (!readMethod
		|| NULL == (s = PyObject_CallFunction(readMethod, PY_STUPID "i", num))) {
			return traits_type::eof();
		}
		char *data;
		if (PyString_AsStringAndSize(s, &data, &num) == -1
		|| num == 0)
			return traits_type::eof();
		strncpy(buffer + putback_size, data, num);
#endif
		// reset buffer pointers
		setg(buffer + (putback_size - numPutback),
			buffer + putback_size, buffer + putback_size + num);
		// return next character
		return traits_type::to_int_type(*gptr());
	}
};

template <class charT, class traits = std::char_traits<charT> >
class basic_pyistream: public std::basic_istream<charT, traits> {
	PyObject* o;
	std::basic_streambuf<charT, traits>* pb;
	typedef std::basic_istream<charT, traits> istream;
	using istream::clear;
	using istream::setstate;
public:
	typedef charT char_type;
	typedef typename traits::int_type int_type;
	typedef typename traits::off_type off_type;
	typedef typename traits::pos_type pos_type;
	typedef traits traits_type;
	typedef std::basic_ios<charT, traits> ios_type;
	typedef std::basic_istream<charT, traits> istream_type;
	typedef basic_stdiobuf<charT, traits> streambuf_type;
	typedef basic_stdiobuf<charT, traits> stdiobuf_type;
	typedef basic_pyfilelikebuf<charT, traits> pyfilelikebuf_type;

	explicit basic_pyistream(PyObject* obj, int mode = std::ios::in):
			std::basic_istream<charT, traits>(0), o(obj), pb(NULL) {
		// pick appropriate streambuf type
		// if a PyFile_Type:
		//   use a stdio stream
		// otherwise:
		//   use version that uses obj as a file object
		FILE* f = PyFile_AsFile(o);
		if (f != NULL) {
			pb = new stdiobuf_type(f);
		} else try {
			pb = new pyfilelikebuf_type(obj, mode);
		} catch (std::invalid_argument &) {
		}
		if (pb == NULL)
			throw std::logic_error("Only works with Python files.");
		this->init(pb);
		Py_INCREF(o);
	}
	virtual ~basic_pyistream() { Py_DECREF(o); delete pb; }
	basic_stdiobuf<charT, traits>* rdbuf() { return pb; }
};

template <class charT, class traits = std::char_traits<charT> >
class basic_pyostream: public std::basic_ostream<charT, traits> {
	PyObject* o;
	std::basic_streambuf<charT, traits>* pb;
	typedef std::basic_ostream<charT, traits> ostream;
	using ostream::clear;
	using ostream::setstate;
public:
	typedef charT char_type;
	typedef typename traits::int_type int_type;
	typedef typename traits::off_type off_type;
	typedef typename traits::pos_type pos_type;
	typedef traits traits_type;
	typedef std::basic_ios<charT, traits> ios_type;
	typedef std::basic_ostream<charT, traits> ostream_type;
	typedef std::basic_streambuf<charT, traits> streambuf_type;
	typedef basic_stdiobuf<charT, traits> stdiobuf_type;
	typedef basic_pyfilelikebuf<charT, traits> pyfilelikebuf_type;

	explicit basic_pyostream(PyObject* obj, int mode = std::ios::out):
			std::basic_ostream<charT, traits>(0), o(obj), pb(NULL) {
		// pick appropriate streambuf type
		// if a PyFile_Type:
		//   use a stdio stream
		// otherwise
		//   use version that uses obj as a file object
		FILE* f = PyFile_AsFile(o);
		if (f != NULL) {
			pb = new stdiobuf_type(f);
		} else try {
			pb = new pyfilelikebuf_type(obj, mode);
		} catch (std::invalid_argument &) {
		}
		if (pb == NULL)
			throw std::logic_error("Only works with Python files.");
		this->init(pb);
		Py_INCREF(o);
	}
	virtual ~basic_pyostream() { Py_DECREF(o); delete pb; }
	streambuf_type* rdbuf() { return pb; }
};

template <class charT, class traits = std::char_traits<charT> >
class basic_pyiostream: public std::basic_iostream<charT, traits> {
	PyObject* o;
	std::basic_streambuf<charT, traits>* pb;
	typedef std::basic_iostream<charT, traits> iostream;
	using iostream::clear;
	using iostream::setstate;
public:
	typedef charT char_type;
	typedef typename traits::int_type int_type;
	typedef typename traits::off_type off_type;
	typedef typename traits::pos_type pos_type;
	typedef traits traits_type;
	typedef std::basic_ios<charT, traits> ios_type;
	typedef std::basic_streambuf<charT, traits> streambuf_type;
	typedef basic_stdiobuf<charT, traits> stdiobuf_type;
	typedef basic_pyfilelikebuf<charT, traits> pyfilelikebuf_type;

	explicit basic_pyiostream(PyObject* obj,
				int mode = std::ios::in|std::ios::out):
			std::basic_iostream<charT, traits>(0), o(obj),
			pb(NULL) {
		// pick appropriate streambuf type
		// if a PyFile_Type:
		//   use a stdio stream
		// otherwise
		//   use version that uses obj as a file object
		FILE* f = PyFile_AsFile(o);
		if (f != NULL) {
			pb = new stdiobuf_type(f);
		} else try {
			pb = new pyfilelikebuf_type(obj, mode);
		} catch (std::invalid_argument &) {
		}
		if (pb == NULL)
			throw std::logic_error("Only works with Python files.");
		this->init(pb);
		Py_INCREF(o);
	}
	virtual ~basic_pyiostream() { Py_DECREF(o); delete pb; }
	streambuf_type* rdbuf() { return pb; }
};

typedef basic_pyistream<char> pyistream;
typedef basic_pyostream<char> pyostream;
typedef basic_pyiostream<char> pyiostream;

extern WRAPPY_IMEX bool PythonFileLike_Check(PyObject*, int mode);

} // namespace wrappy

# undef PY_STUPID

#endif
