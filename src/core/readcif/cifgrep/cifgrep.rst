=======
cifgrep
=======

------------------------------------
Scan CIF files for particular values
------------------------------------

:Author: gregc@cgl.ucsf.edu
:Date: 2014-9-03
:Copyright: RBVI Open Source
:Manual section: 1
:Manual group: text processing

SYNOPSIS
========

  cifgrep [-d] [-m] [-l] [-v] CIF_tags filename(s)

DESCRIPTION
===========

Scan the input for the given CIF tags and output the matches.

CIF tags are comma separated, DDL2-style, category.id values.
Subsequent id's can elide the category (.id is sufficient).
Since only one category is supported, the default output is 
tab separated columns.

If no matches are found, then the exit status code is 1.

OPTIONS
=======

-d  If a match is found, suppress output and list the data block code.
-m  Speed up processing by assuming mmCIF style -- lowercase keyword/tags at the beginning of a line.
-l  If a match is found, suppress output and list the filename.
-v  Verbose.  Give reason for search failure.

The -d and -l options are mutually exclusive.

SEE ALSO
========

* `CIF resources <http://www.iucr.org/resources/cif>`_
* `mmCIF resources <http://mmcif.wwpdb.org/>`_
