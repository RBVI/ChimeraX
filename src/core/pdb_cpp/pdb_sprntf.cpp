// vi: set expandtab ts=4 sw-4:

#include "PDB.h"
#include <ctype.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

static const char OVERFLOW_CHAR = '*';

	// scratch must be big enough to hold the largest number
static char	scratch[256];

static char *
e_out(int width, char *where)
{
	for (; width > 0; --width)
		*where++ = OVERFLOW_CHAR;
	return where;
}

static char *
outint(int value, int width, int radix, char fill_char, char hex,
					bool left_justify, char *p, char zero)
{
	char	*s;
	int	n;
	bool	negative;

	negative = value < 0;
	if (negative) {
		value = -value;
		--width;
	}
	s = scratch;
	if (value)
		do {
			n = value % radix;
			*s++ = n < 10 ? '0' + n : hex + n - 10;
			value /= radix;
		} while (value);
	else
		*s++ = zero;
	n = s - scratch;
	if (width != -1 && n > width)
		return e_out(width + negative, p);

	if (negative && fill_char == '0')
		*p++ = '-';
	if (!left_justify)
		for (; width > n; --width)
			*p++ = fill_char;
	if (negative && fill_char == ' ')
		*p++ = '-';
	while (--s >= scratch)
		*p++ = *s;
	if (left_justify)
		for (; width > n; --width)
			*p++ = fill_char;
	return p;
}

static char *
outunsigned(unsigned int value, int width, char fill_char, bool left_justify,
									char *p)
{
	char	*s;
	int	n;

	s = scratch;
	while (value) {
		*s++ = value % 10 + '0';
		value /= 10;
	}
	n = s - scratch;
	if (n == 0)
		*s++ = '0', n = 1;
	if (width != -1 && n > width)
		return e_out(width, p);

	if (!left_justify)
		for (; width > n; --width)
			*p++ = fill_char;
	while (--s >= scratch)
		*p++ = *s;
	if (left_justify)
		for (; width > n; --width)
			*p++ = fill_char;
	return p;
}

static char *
outstr(char *s, int width, int maxstr, char fill_char, bool left_justify, char *p)
{
	int	len;

	len = strlen(s);
	if (maxstr >= 0 && len > maxstr)
		len = maxstr;
	if (width != -1 && len > width)
		return e_out(width, p);

	if (!left_justify)
		for (; width > len; --width)
			*p++ = fill_char;
	else
		width -= len;
	for (; len > 0; --len)
		*p++ = *s++;
	if (left_justify)
		for (; width > 0; --width)
			*p++ = fill_char;
	return p;
}

static char *
outfloat(double value, int width, int nplace, bool left_justify,
							char *p, char zero)
{
	int	intval;
	bool	negative;

	negative = copysign(1, value) < 0.0;
	if (negative)
		value = -value;

	for (int i = 0; i < nplace; ++i)
		value *= 10.0;

	intval = (int) (value + 0.5);

	if (width == -1)
		width = nplace + 4;		/* TODO: fix */
	else if (nplace + (nplace == 0 ? 1 : 2) > width)
		return e_out(width, p);

	if (value == 0 && zero == ' ') {
		for (; width > 0; --width)
			*p++ = ' ';
		return p;
	}

	char *s = scratch;	// holds backwards version of number
	for (int i = 0; i < width; ++i) {
		if (i == nplace)
			*s++ = '.';
		*s++ = '0' + intval % 10;
		intval /= 10;
		if (intval == 0 && i >= nplace)
			break;
	}
	if (negative)
		*s++ = '-';
	*s = '\0';

	if (s - scratch > width + nplace)
		return e_out(width, p);
	if (left_justify) {
		for (int i = 0; i < width; ++i) {
			if (s > scratch)
				*p++ = *--s;
			else
				*p++ = ' ';
		}
	} else {
		for (int i = width - (s - scratch); i > 0; --i)
			*p++ = ' ';
		for (int i = 0; i < width; ++i) {
			if (s > scratch)
				*p++ = *--s;
			else
				break;
		}
	}
	return p;
}

static char *
outexp(double value, int width, int nplace, char fill_char, bool left_justify,
									char *p)
{
	int	n;
	char	*s;
	bool	negative;
	double	fraction;

	negative = copysign(1, value) < 0;
	if (negative) {
		value = -value;
		--width;
	}

	n = 0;
	while (value > 10)
		++n, value /= 10;
	if (value)
		while (value < 1)
			--n, value *= 10;

	s = scratch;
	if (n < 0) {
		n = -n;
		*s++ = n % 10 + '0';
		*s++ = n / 10 + '0';
		*s++ = '-';
	}
	else {
		*s++ = n % 10 + '0';
		*s++ = n / 10 + '0';
		*s++ = '+';
	}
	*s = 'e';

	s = scratch + nplace + 4;	/* 4 == strlen("e+00") */
	fraction = value - (int) value;
	for (n = 0; n < nplace; ++n) {
		fraction *= 10.0;
		*--s = '0' + (int) fraction;
		fraction -= (int) fraction;
	}

	s = scratch + nplace + 4;
	if (nplace)
		*s++ = '.';
	n = (int) value;
	if (n)
		*s++ = n % 10 + '0';
	else
		*s++ = '0';
	n = s - scratch;
	if (width != -1 && n > width)
		return e_out(width + negative, p);

	if (negative && fill_char == '0')
		*p++ = '-';
	if (!left_justify)
		for (; width > n; --width)
			*p++ = fill_char;
	if (negative && fill_char == ' ')
		*p++ = '-';
	while (--s >= scratch)
		*p++ = *s;
	if (left_justify)
		for (; width > n; --width)
			*p++ = fill_char;
	return p;
}

namespace pdb {

int
PDB::sprintf(char *outbuf, const char *fmt, ...)
{
	va_list	argv;
	char		*p;
	const char	*f;
	int		field1, field2;
	char		c, fill_char;
	int		inum;
	unsigned 	unum;
	double		fnum;
	bool		left_justify;

	va_start(argv, fmt);
	f = fmt;
	p = outbuf;
	while (*f) {
		if (*f == '%') {
			++f;
			left_justify = *f == '-';
			if (left_justify)
				++f;

			if (*f == '0')
				fill_char = '0', ++f;
			else
				fill_char = ' ';

			if (isdigit(*f)) {
				field1 = *f++ - '0';
				while (isdigit(*f))
					field1 = field1 * 10 + *f++ - '0';
			}
			else
				field1 = -1;

			if (*f == '.') {
				++f;
				field2 = 0;
				while (isdigit(*f))
					field2 = field2 * 10 + *f++ - '0';
			}
			else
				field2 = -1;

			if (*f == 'l' || *f == 'h')
				++f;

			while (isspace(*f))
				++f;
			switch (*f) {
			  case 'c':
				c = (char) va_arg(argv, int);
				if (c == '\0')
					c = ' ';
				if (left_justify)
					*p++ = c;
				while (--field1 > 0)
					*p++ = fill_char;
				if (!left_justify)
					*p++ = c;
				break;
			  case 'd':
			  case 'D':
				inum = va_arg(argv, int);
				p = outint(inum, field1, 10, fill_char, 'a',
					left_justify, p, (*f == 'D') ? ' ':'0');
				break;
			  case 'e':
				fnum = va_arg(argv, double);
				if (field2 < 0)
					field2 = 6;
				p = outexp(fnum, field1, field2, fill_char,
					left_justify, p);
				break;
			  case 'f':
			  case 'F':
				fnum = va_arg(argv, double);
				if (field2 < 0)
					field2 = 6;
				p = outfloat(fnum, field1, field2,
					left_justify, p, (*f == 'F') ? ' ':'0');
				break;
			  case 'o':
				inum = va_arg(argv, int);
				p = outint(inum, field1, 8, fill_char, 'a',
					left_justify, p, '0');
				break;
			  case 's':
				p = outstr(va_arg(argv, char *), field1,
					field2, fill_char, left_justify, p);
				break;
			  case 'u':
				unum = va_arg(argv, unsigned);
				p = outunsigned(unum, field1, fill_char,
					left_justify, p);
				break;
			  case 'x':
				inum = va_arg(argv, int);
				p = outint(inum, field1, 16, fill_char, 'a',
					left_justify, p, '0');
				break;
			  case 'X':
				inum = va_arg(argv, int);
				p = outint(inum, field1, 16, fill_char, 'A',
					left_justify, p, '0');
				break;
			  default:
				if (left_justify)
					*p++ = *f;
				while (--field1 > 0)
					*p++ = fill_char;
				if (!left_justify)
					*p++ = *f;
				break;
			}
			++f;
		}
		else if (*f == '\\') {		/* Special character */
			switch (*++f) {
			  case 'n':
				*p++ = '\n';
				break;
			  case 'r':
				*p++ = '\r';
				break;
			  case 'b':
				*p++ = '\b';
				break;
			  case 't':
				*p++ = '\t';
				break;
			  case 'f':
				*p++ = '\f';
				break;
			  case '0': case '1': case '2': case '3':
			  case '4': case '5': case '6': case '7':
				inum = *f++ - '0';
				if (*f >= '0' && *f <= '7') {
					inum = inum * 8 + *f++ - '0';
					if (*f >= '0' && *f <= '7')
						inum = inum * 8 + *f++ - '0';
				}
				--f;
				*p++ = (char) inum;
				break;
			  default:
				*p++ = *f;
			}
			++f;
		}
		else				/* Normal character */
			*p++ = *f++;
	}
	*p = '\0';
	va_end(argv);
	return p - outbuf;
}

} // namespace pdb
