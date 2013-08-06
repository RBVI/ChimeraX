#include "PDB.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

//
//	pdb_sscanf performs similarly to sscanf, execept that fields are of
//	fixed length and a complete line is always consumed.  The field
//	width defaults to one.  If the line is shorter than expected then
//	the default is returned.
//
//		d	get an integer.  Default:  0.
//		f	get a floating point number (C double).  Default:  0.0.
//		(space) ignore characters within field
//		s	get a C string, leading and trailing spaces are
//			stripped; the field width is used as a limit on
//			the string length, the null character is appended
//			to the end of the string.  Default:  empty string.
//		c	get a character(s); no stripping of spaces, nor is
//			a null character appended.  Default:  space(s).
//

static const int MAXFIELDSIZE = PDB::BUF_LEN;

#define endbuf(buf)	(*buf == '\0' || *buf == '\n' || *buf == '\r')

int
PDB::sscanf(const char *buffer, const char *fmt, ...)
{
	va_list	ap;
	int		i, field_width;
	int		nmatch;
	char		*s, *t;
	char		tmp[MAXFIELDSIZE];

	va_start(ap, fmt);
	nmatch = 0;
	for (; *fmt != '\0'; fmt++) {
		if (*fmt != '%') {
			if (*buffer == *fmt)
				buffer++;
			else if (!endbuf(buffer))
				return -1;
			continue;
		}

		// calculate field_width
		field_width = 0;
		for (++fmt; isdigit(*fmt); fmt++)
			field_width = field_width * 10 + *fmt - '0';
		if (field_width == 0)
			field_width = 1;	// default
		if (field_width > MAXFIELDSIZE - 1)
			abort();		// should never happen
		if (*buffer != '\0' && *buffer != '\n')
			nmatch++;

		switch (*fmt) {

		case 'd':			// integer
			// if we've already seen the end of the buffer, don't
			// try to get anymore characters
			if (endbuf(buffer)) {
				*(va_arg(ap, int *)) = 0;
				break;
			}

			s = tmp;
			for (i = 0; i < field_width; i++) {
				if (endbuf(buffer))
					break;
				*s++ = *buffer++;
			}
			*s = '\0';
			// remove trailing spaces
			while (s > tmp && isspace(*(s - 1)))
				*--s = '\0';
			*(va_arg(ap, int *)) = (int) strtol(tmp, &t, 10);
			if (t != s)
				return -1;
			break;

		case 'f':			// floating point
			// if we've already seen the end of the buffer, don't
			// try to get anymore characters
			if (endbuf(buffer)) {
				*(va_arg(ap, double *)) = 0.0;
				break;
			}

			s = tmp;
			for (i = 0; i < field_width; i++) {
				if (endbuf(buffer))
					break;
				*s++ = *buffer++;
			}
			*s = '\0';
			// remove trailing spaces
			while (s > tmp && isspace(*(s - 1)))
				*--s = '\0';
			*(va_arg(ap, double *)) = strtod(tmp, &t);
			if (t != s)
				return -1;
			break;

		case 's':			// string
			// if we've already seen the end of the buffer, don't
			// try to get anymore characters
			if (endbuf(buffer)) {
				*(va_arg(ap, char *)) = '\0';
				break;
			}

			s = t = va_arg(ap, char *);
			for (i = 0; i < field_width; i++) {
				if (endbuf(buffer))
					break;
				*s++ = *buffer++;
			}
			*s = '\0';
			// remove trailing spaces
			while (s > t && isspace(*--s))
				*s = '\0';
			break;

		case 'c':			// character(s)
			s = va_arg(ap, char *);
			for (i = 0; i < field_width; i++)
				s[i] = ' ';	// default

			// if we've already seen the end of the buffer, don't
			// try to get anymore characters
			if (endbuf(buffer))
				break;

			for (i = 0; i < field_width; i++) {
				if (endbuf(buffer))
					break;
				*s++ = *buffer++;
			}
			break;

		case ' ':			// space (ignore)
			// if we've already seen the end of the buffer, don't
			// try to get anymore characters
			if (endbuf(buffer))
				break;

			for (i = 0; i < field_width; i++, buffer++)
				if (endbuf(buffer))
					break;
			break;

		default:
			va_end(ap);
			return -1;
		}
	}
	va_end(ap);
	return nmatch;
}
