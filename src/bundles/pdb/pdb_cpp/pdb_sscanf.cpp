// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include "PDB.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

namespace pdb {

//
//  pdb_sscanf performs similarly to sscanf, execept that fields are of
//  fixed length and a complete line is always consumed.  The field
//  width defaults to one.  If the line is shorter than expected then
//  the default is returned.
//
//      d   get an integer.  Default:  0.
//      f   get a floating point number (C double).  Default:  0.0.
//      (space) ignore characters within field
//      s   get a C string, leading and trailing spaces are
//          stripped; the field width is used as a limit on
//          the string length, the null character is appended
//          to the end of the string.  Default:  empty string.
//      c   get a character(s); no stripping of spaces, nor is
//          a null character appended.  Default:  space(s).
//

static const int MAXFIELDSIZE = PDB::BUF_LEN;

#define endbuf(buf) (*buf == '\0' || *buf == '\n' || *buf == '\r')

static int
ipow(int base, int exp)
{
    int val = 1;
    while (exp-- > 0) {
        val *= base;
    }
    return val;
}

static int
h36_to_int(char *buf, char **end)
{
    while (*buf == ' ') ++buf;
    if ((*buf >= '0' && *buf <= '9') || *buf == '\0' || *buf == '-')
        return strtol(buf, end, 10);

    int field_width = 0;
    int ret_val = 0;
    while (*buf != '\0') {
        // for some unknown reason, if the variable 'c' is
        // declared as char, the computed value is wrong!
        int c = *buf++;
        ++field_width;
        int val;
        if (isupper(c))
            val = c - 'A' + 10;
        else if (isdigit(c))
            val = c - '0';
        else {
            *end = nullptr;
            return 0;
        }
        ret_val = 36 * ret_val + val;
    }
    if (field_width == 0) {
        *end = nullptr;
        return 0;
    }
    // To make 'A0..00' one more than '99..99'...
    int target = ipow(10, field_width);
    int Aval = 10 * ipow(36, field_width-1);
    ret_val -= Aval;
    ret_val += target;
    *end = buf;
    return ret_val;
}

int
PDB::sscanf(const char *buffer, const char *fmt, ...)
{
    va_list ap;
    int     i, field_width;
    int     nmatch;
    char        *s, *t;
    char        tmp[MAXFIELDSIZE];

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
            field_width = 1;    // default
        if (field_width > MAXFIELDSIZE - 1)
            abort();        // should never happen
        if (*buffer != '\0' && *buffer != '\n')
            nmatch++;

        switch (*fmt) {

        case 'd':           // integer
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
            *(va_arg(ap, int *)) = h36_to_int(tmp, &t);
            if (t != s)
                return -1;
            break;

        case 'f':           // floating point
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

        case 's':           // string
        case 'S':
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
            if (*fmt == 'S')
                break;
            // remove trailing spaces
            while (s > t && isspace(*--s))
                *s = '\0';
            break;

        case 'c':           // character(s)
            s = va_arg(ap, char *);
            for (i = 0; i < field_width; i++)
                s[i] = ' '; // default

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

        case ' ':           // space (ignore)
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

}  // namespace pdb
