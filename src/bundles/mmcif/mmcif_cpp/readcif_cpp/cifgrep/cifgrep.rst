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

-d  Show data block instead of filename.
-h  Suppress filename.
-H  Always show filename.
-l  If a match is found, suppress output and just list the filename.
-m  Speed up processing by assuming mmCIF style -- lowercase keyword/tags at the beginning of a line.
-v  Verbose.  Give reason for search failure.

SEE ALSO
========

* `CIF resources <https://www.iucr.org/resources/cif>`_
* `mmCIF resources <https://mmcif.wwpdb.org/>`_
