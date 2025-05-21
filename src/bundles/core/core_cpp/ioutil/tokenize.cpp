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

#include <ctype.h>
#include <string.h>
#include "tokenize.h"

namespace ioutil {

/*
 * tokenize:
 *  Break string into array of words at spaces, keeping strings
 *  intact and recognizing backslash escapes, returning number
 *  of words found or -1 if more than n found or -2 if non-printable
 *  character encountered or -3 if mismatched quotes.
 */
int
tokenize(char *string, char **array, int n)
// char *string,    /* string to be tokenized */
//  **array;    /* returned array of pointers to tokens in string */
// int  n;      /* maximum number of tokens to look for */
{
    int i;

    for (i = 0; i < n; i++) {
        while (isspace(*string))
            string++;
        if (*string == '"') {
            *array++ = ++string;
            while (isprint(*string) && *string != '"') {
                if (*string == '\\') { /* backslash escapes */
                    strcpy(string, string+1);
                    switch (*string) {
                      case 't': /* tab */
                        *string = '\t';
                        break;
                      case 'b': /* backspace */
                        *string = '\b';
                        break;
                      case 'n': /* newline */
                        *string = '\n';
                        break;
                      case 'r': /* carriage return */
                        *string = '\r';
                        break;
                      case 'f': /* formfeed */
                        *string = '\f';
                        break;
                      default: /* treat as normal */
                        break;
                    }
                }
                string++;
            }
            if (*string == '\0')
                return -3;
            if (!isprint(*string))
                return -2;
            *string++ = '\0';
            continue;
        }
        *array++ = string;
        if (*string == '\0')
            break;
        if (!isprint(*string))
            return -2;
        while (!isspace(*string) && isprint(*string)) {
            if (*string == '\\') { /* backslash escapes */
                strcpy(string, string+1);
                switch (*string) {
                  case 't': /* tab */
                    *string = '\t';
                    break;
                  case 'b': /* backspace */
                    *string = '\b';
                    break;
                  case 'n': /* newline */
                    *string = '\n';
                    break;
                  case 'r': /* carriage return */
                    *string = '\r';
                    break;
                  case 'f': /* formfeed */
                    *string = '\f';
                    break;
                  default: /* treat as normal */
                    break;
                }
            }
            string++;
        }
        if (isspace(*string))
            *string++ = '\0';
    }
    while (isspace(*string))
        string++;
    return *string == '\0' ? i : -1;
}

} // namespace ioutil
