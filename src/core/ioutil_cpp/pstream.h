// vi: set expandtab ts=4 sw=4:
// Copyright (c) 2001 The Regents of the University of California.
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
// $Id: pstream.h 36239 2012-04-26 00:09:34Z goddard $

#ifndef ioutil_pstream
# define ioutil_pstream

# include <istream>
# include <ostream>
# include <streambuf>
# include <algorithm>

namespace ioutil {

class raw_pipe2 {
public:
    static const int doNotBlock = std::ios::app;
protected:
# ifdef _WIN32
    void    *child;
    void    *fd[2];
# else
    int child;
    int fd[2];
# endif
    std::streamsize bufsize[2];
    bool useBlockingWait;
public:
    raw_pipe2();
    ~raw_pipe2() { close(); }
    int open(const char *cmd, int mode = std::ios::in|std::ios::out);
    int close(int mode = std::ios::in|std::ios::out);
    bool is_open() const;
    int read(void *buf, int count);
    int write(const void *buf, int count);
};

template <class charT, class traits = std::char_traits<charT> >
class basic_pipebuf: private raw_pipe2,
                public std::basic_streambuf<charT, traits> {
    charT       *buf[2];
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
    static const int putback_size = 4;
#if 0
    // fails in gcc/g++ 4.1
    using raw_pipe2::is_open;
#else
    bool is_open() const { return raw_pipe2::is_open(); }
#endif

    typedef charT char_type;
    typedef typename traits::int_type int_type;
    typedef typename traits::off_type off_type;
    typedef typename traits::pos_type pos_type;
    typedef traits traits_type;

    basic_pipebuf() {
        bufsize[0] = bufsize[1] = 0;
        buf[0] = buf[1] = NULL;
    }
    virtual ~basic_pipebuf() {
        (void) close();
    }
    int close(int mode = std::ios::in|std::ios::out)
    {
        if ((mode & std::ios::in) && buf[0] != NULL) {
            delete [] buf[0];
            buf[0] = NULL;
        }
        if ((mode & std::ios::out) && buf[1] != NULL) {
            sync();
            delete [] buf[1];
            buf[1] = NULL;
            setp(NULL, 0);
        }
        return raw_pipe2::close(mode);
    }
    int open(const char *cmd, int mode = std::ios::in|std::ios::out) {
        if (is_open())
            (void) close();
        if (raw_pipe2::open(cmd, mode) == -1)
            return -1;
        if ((mode & std::ios::in) != 0) {
            bufsize[0] += putback_size;
            buf[0] = new char_type[bufsize[0]];
            char_type *pos = buf[0] + putback_size;
            setg(pos, pos, pos);
        }
        if ((mode & std::ios::out) != 0) {
            buf[1] = new char_type[bufsize[1]];
            setp(buf[1], buf[1] + bufsize[1] - 1);
        }
        useBlockingWait = (mode & doNotBlock) == 0;
        return 0;
    }
protected:
    // output support
# if 0
    // TODO: violates buffering?
    virtual std::streamsize xsputn(const char_type *buf,
                            std::streamsize num) {
        // optimization to avoid multiple sputc's
        if (write(buf[1], num) != num)
            return traits_type::eof();
        return num;
    }
# endif
    int_type flush() {
        std::streamsize num = pptr() - pbase();
        if (write(buf[1], num) != num)
            return traits_type::eof();
        pbump(-num);
        return num;
    }
    virtual int_type overflow(int_type c) {
        // buffer full
        if (c != traits_type::eof()) {
            // save character into reserved extra slot
            *pptr() = c;
            pbump(1);
        }
        if (flush() == traits_type::eof())
            return traits_type::eof();
        return c;
    }
    virtual int sync() {
        if (flush() == traits_type::eof())
            return -1;
        return 0;
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
                    buf[0] + (putback_size - numPutback));
        // get new characters
        int num = read(buf[0] + putback_size,
                        bufsize[0] - putback_size);
        if (num <= 0)
            return traits_type::eof();
        // reset buffer pointers
        setg(buf[0] + (putback_size - numPutback),
            buf[0] + putback_size, buf[0] + putback_size + num);
        // return next character
        return traits_type::to_int_type(*gptr());
    }
};

template <class charT, class traits = std::char_traits<charT> >
class basic_ipipestream: public std::basic_istream<charT, traits> {
    basic_pipebuf<charT, traits> pb;
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
    typedef basic_pipebuf<charT, traits> streambuf_type;

    explicit basic_ipipestream(const char *cmd, int mode = std::ios::in):
                    std::basic_istream<charT, traits>(0) {
        this->init(&pb);
        if ((mode & (std::ios::in|std::ios::out)) == 0)
            mode |= std::ios::in;
        open(cmd, mode);
    }
    virtual ~basic_ipipestream() {}
    basic_pipebuf<charT, traits> *rdbuf() { return &pb; }
    bool is_open() const { return pb.is_open(); }
    void open(const char *cmd, int mode = std::ios::in) {
        clear();
        if (pb.open(cmd, mode) == -1)
            setstate(std::ios::failbit);
    }
    void close() {
        if (pb.close() == -1)
            setstate(std::ios::failbit);
    }
};

template <class charT, class traits = std::char_traits<charT> >
class basic_opipestream: public std::basic_ostream<charT, traits> {
    basic_pipebuf<charT, traits> pb;
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
    typedef basic_pipebuf<charT, traits> streambuf_type;

    explicit basic_opipestream(const char *cmd, int mode = std::ios::out):
                    std::basic_ostream<charT, traits>(0) {
        this->init(&pb);
        if ((mode & (std::ios::in|std::ios::out)) == 0)
            mode |= std::ios::out;
        open(cmd, mode);
    }
    virtual ~basic_opipestream() {}
    basic_pipebuf<charT, traits> *rdbuf() { return &pb; }
    bool is_open() const { return pb.is_open(); }
    void open(const char *cmd, int mode = std::ios::out) {
        clear();
        if (pb.open(cmd, mode) == -1)
            setstate(std::ios::failbit);
    }
    void close() {
        if (pb.close() == -1)
            setstate(std::ios::failbit);
    }
};

template <class charT, class traits = std::char_traits<charT> >
class basic_pipestream: public std::basic_iostream<charT, traits> {
    basic_pipebuf<charT, traits> pb;
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

    explicit basic_pipestream(const char *cmd,
                    int mode = std::ios::in|std::ios::out):
                std::basic_iostream<charT, traits>(0) {
        this->init(&pb);
        open(cmd, mode);
    }
    virtual ~basic_pipestream() {}

    basic_pipebuf<charT, traits> *rdbuf() { return &pb; }
    bool is_open() const { return pb.is_open(); }
    void open(const char *cmd, int mode = std::ios::in|std::ios::out) {
        clear();
        if (pb.open(cmd, mode) == -1)
            setstate(std::ios::failbit);
    }
    void close(int mode = std::ios::in|std::ios::out) {
        if (pb.close(mode) == -1)
            setstate(std::ios::failbit);
    }
};

typedef basic_ipipestream<char> ipipestream;
typedef basic_opipestream<char> opipestream;
typedef basic_pipestream<char> pipestream;

} // namespace ioutil

#endif
