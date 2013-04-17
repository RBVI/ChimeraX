#include "Reg.h"
#include <memory>
#include <stdexcept>
#include <string.h>

#include <sys/types.h>
#include <pcreposix.h>

namespace {

inline int
compFlags(int flags)
{
	int regcompFlags = 0;
	if (flags & Reg::EXTENDED)
		regcompFlags |= REG_EXTENDED;
	if (flags & Reg::ICASE)
		regcompFlags |= REG_ICASE;
	if (flags & Reg::NOSUB)
		regcompFlags |= REG_NOSUB;
	if (flags & Reg::NEWLINE)
		regcompFlags |= REG_NEWLINE;
	return regcompFlags;
}

inline int
execFlags(int flags)
{
	int regexecFlags = 0;
	if (flags & Reg::NOTBOL)
		regexecFlags |= REG_NOTBOL;
	if (flags & Reg::NOTEOL)
		regexecFlags |= REG_NOTEOL;
	return regexecFlags;
}

}

using std::string;

class RegData {
public:
	regex_t		preg;
	regmatch_t	*matchResults;
};

RegData *
Reg::init(char const *pattern, int flags) throw (Error)
{
	std::auto_ptr<RegData> tmp(new RegData);
	if (tmp.get() == NULL)
		throw Error(NULL, -2);
	int result = regcomp(&tmp->preg, pattern, compFlags(flags));
	if (result != 0)
		throw Error(tmp.get(), result);
	tmp->matchResults = new regmatch_t [tmp->preg.re_nsub + 1];
	return tmp.release();
}

Reg::Reg(char const *pattern, int flags) throw (Error)
						: data(init(pattern, flags))
{
}

Reg::Reg(const string &pattern, int flags) throw (Error)
					: data(init(pattern.c_str(), flags))
{
}

Reg::~Reg()
{
	if (data == NULL)
		return;
	if (data->matchResults != NULL)
		delete [] data->matchResults;
	regfree(&data->preg);
	delete data;
}
Reg::Error::Error(const RegData *data, int i): val(i), buf(0), buflen(0)
{
	switch (val) {
	  default:
		buflen = regerror(val, &data->preg, NULL, 0);
		buf = new char [buflen];
		(void) regerror(val, &data->preg, buf, buflen);
		break;
	  case -1: {
		char const badInterval[] = "interval index out of bounds";
		buflen = sizeof badInterval;
		buf = new char [buflen];
		::strcpy(buf, badInterval);
		break;
	  }
	  case -2: {
		char const outMemory[] = "out of memory";
		buflen = sizeof outMemory;
		buf = new char [buflen];
		::strcpy(buf, outMemory);
		break;
	  }
	}
}

Reg::Error::~Error()
{
	if (buf)
		delete [] buf;
}

Reg::Error::Error(const Error &err)
{
	val = err.val;
	if (err.buf) {
		buflen = err.buflen;
		buf = new char [buflen];
		::strcpy(buf, err.buf);
	} else {
		buf = 0;
		buflen = 0;
	}
}

Reg::Error &
Reg::Error::operator =(const Error &err)
{
	if (this != &err) {
		val = err.val;
		if (err.buf) {
			buflen = err.buflen;
			buf = new char [buflen];
			::strcpy(buf, err.buf);
		} else {
			buf = 0;
			buflen = 0;
		}
	}
	return *this;
}

bool
Reg::exec(char const *search, int flags) throw (Error)
{
	int result = regexec(&data->preg, search, data->preg.re_nsub + 1,
					data->matchResults, execFlags(flags));
	switch (result) {
	  case 0:
		return true;
	  case REG_NOMATCH:
		return false;
	  default:
		throw Error(data, result);
	}
#ifdef _MSC_VER
	// NOTREACHED
	return false;
#endif
}

Reg::size_type
Reg::find(const string &search, size_type pos, int flags) const throw (Error)
{
	if (pos > search.size())
		throw std::out_of_range("position beyond search string");

	regmatch_t tmp;
	int result = regexec(&data->preg, search.c_str() + pos, 1, &tmp,
							execFlags(flags));
	switch (result) {
	  case 0:
		return tmp.rm_so + pos;
	  case REG_NOMATCH:
		return string::npos;
	  default:
		throw Error(data, result);
	}
#ifdef _MSC_VER
	// NOTREACHED
	return 0;
#endif
}

Reg::size_type
Reg::find(const string &search, size_type *length, size_type pos, int flags)
							const throw (Error)
{
	if (pos > search.size())
		throw std::out_of_range("position beyond search string");

	regmatch_t tmp;
	int result = regexec(&data->preg, search.c_str() + pos, 1, &tmp,
							execFlags(flags));
	switch (result) {
	  case 0:
		*length = tmp.rm_eo - tmp.rm_so;
		return tmp.rm_so + pos;
	  case REG_NOMATCH:
		*length = 0;
		return string::npos;
	  default:
		throw Error(data, result);
	}
#ifdef _MSC_VER
	// NOTREACHED
	return 0;
#endif
}

bool
Reg::matches(const string &search, size_type pos, int flags) const throw (Error)
{
	regmatch_t tmp;
	int result = regexec(&data->preg, search.c_str() + pos, 1, &tmp,
							execFlags(flags));
	switch (result) {
	  case 0:
		return tmp.rm_so == 0 && tmp.rm_eo == regoff_t(search.length() - pos);
	  case REG_NOMATCH:
		return false;
	  default:
		throw Error(data, result);
	}
#ifdef _MSC_VER
	// NOTREACHED
	return false;
#endif
}

size_t
Reg::numMatches() const throw ()
{
	return data->preg.re_nsub + 1;
}

Reg::Interval
Reg::match(size_t m) const throw (Error)
{
	if (m > data->preg.re_nsub)
		throw Error(data, -1);
	return Interval(data->matchResults[m].rm_so, data->matchResults[m].rm_eo);
}

size_t
Reg::matchLen(size_t m) const throw (Error)
{
	if (m > data->preg.re_nsub)
		throw Error(data, -1);
	return data->matchResults[m].rm_eo - data->matchResults[m].rm_so;
}

const Reg Reg::white("[[:space:]]+", Reg::EXTENDED);
const Reg Reg::integer("-?[[:digit:]]+", Reg::EXTENDED);
const Reg Reg::real("-?(([[:digit:]]+\\.[[:digit:]]*)|([[:digit:]]+)|(\\.[[:digit:]]+))([eE][-+]?[[:digit:]]+)?", Reg::EXTENDED);
const Reg Reg::alpha("[[:alpha:]]+", Reg::EXTENDED);
const Reg Reg::lowercase("[[:lower:]]+", Reg::EXTENDED);
const Reg Reg::uppercase("[[:upper:]]+", Reg::EXTENDED);
const Reg Reg::alphanum("[[:alnum:]]+", Reg::EXTENDED);
const Reg Reg::identifier("[[:alpha:]_][[:alpha:]_[:digit:]]*");
