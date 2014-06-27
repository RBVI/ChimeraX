/*
 * Copyright (c) 1987, 1993, 1994
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)getopt.c	8.2 (Berkeley) 4/2/94";
#endif /* LIBC_SCCS and not lint */

#include <iostream>
#include <string.h>
#include "GetOpt.h"

namespace ioutil {

#ifdef EOF
# undef EOF
#endif
#ifndef NULL
# define NULL 0
#endif

static int const	EOF = -1;
static int const	BADCH = (int)'?';
static int const	BADARG = (int)':';
static char const	*EMSG = "";

GetOpt::GetOpt(int argc, char * const *argv, char const *opts, bool err):
	opterr(err), optind(1), optopt(0), optreset(false), optarg(0),
	nargc(argc), nargv(argv), ostr(opts), place(EMSG)
{
}

int
GetOpt::index() const
{
	return optind;
}

char const *
GetOpt::arg() const
{
	return optarg;
}

int
GetOpt::option() const
{
	return optopt;
}

void
GetOpt::reset()
{
	optreset = true;
	optind = 1;
}

void
GetOpt::skip(int count)
{
	optind += count;
	if (optind < 1)
		optind = 1;
	else if (optind > nargc)
		optind = nargc;
}

void
GetOpt::printErrors(bool b)
{
	opterr = b;
}

/*
 * getopt --
 *	Parse argc/argv argument vector.
 */
int
GetOpt::operator()()
{
	const char *oli;			/* option letter list index */

	if (optreset || !*place) {		/* update scanning pointer */
		optreset = false;
		if (optind >= nargc || *(place = nargv[optind]) != '-') {
			place = EMSG;
			return (EOF);
		}
		if (place[1] && *++place == '-') {	/* found "--" */
			++optind;
			place = EMSG;
			return (EOF);
		}
	}					/* option letter okay? */
	if ((optopt = (int)*place++) == (int)':' ||
	    !(oli = ::strchr(ostr, optopt))) {
		/*
		 * if the user didn't specify '-' as an option,
		 * assume it means EOF.
		 */
		if (optopt == (int)'-')
			return (EOF);
		if (!*place)
			++optind;
		if (opterr && *ostr != ':')
			std::cerr << nargv[0] << ": illegal option -- "
						<< char(optopt) << std::endl;
		return (BADCH);
	}
	if (*++oli != ':') {			/* don't need argument */
		optarg = NULL;
		if (!*place)
			++optind;
	}
	else {					/* need an argument */
		if (*place)			/* no white space */
			optarg = place;
		else if (nargc <= ++optind) {	/* no arg */
			place = EMSG;
			if (*ostr == ':')
				return (BADARG);
			if (opterr)
				std::cerr << nargv[0]
					<< ": option requires an argument -- "
					<< char(optopt) << std::endl;
			return (BADCH);
		}
	 	else				/* white space */
			optarg = nargv[optind];
		place = EMSG;
		++optind;
	}
	return (optopt);			/* dump back option letter */
}

} // namespace ioutil
