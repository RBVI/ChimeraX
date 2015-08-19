Benchmarking readcif
====================

    :Author: Greg Couch
    :Organization: RBVI_, `University of California at San Francisco`_
    :Contact: gregc@cgl.ucsf.edu
    :Copyright: © Copyright 2014 by the Regents of the University of Californa.  All Rights reserved.
    :Last modified: 2014-6-17

.. _RBVI: http://www.rbvi.ucsf.edu/
.. _University of California at San Francisco: http://www.ucsf.edu/

The goal of this benchmark is to compare the performance of
readcif versus other C++ mmCIF readers, `cifparse-obj`_ and ucif_,
and to quantify how much faster the stylized PDBx/mmCIF can be parsed.

.. _cifparse-obj: http://sw-tools.pdb.org/apps/CIFPARSE-OBJ/
.. _ucif: http://cctbx.sourceforge.net/ucif/
.. _iotbx.cif: http://dx.doi.org/10.1107/S0021889811041161


Benchmark Results
~~~~~~~~~~~~~~~~~

    Tests were all on a Dell Studio XPS 435MT with an Intel i7-920 at 2.66 GHz,
    6 GiB of memory, and a 250GiB Samsung 840 SSD running Ubuntu 14.04.

    The test case was to read a mmCIF file, and for each atom in the
    atom_site table extract its element, its name, its residue name,
    its chain identifier, its residue number, and its x, y, and z coordinates.
    This test case isolates the performance of extracting the atom data.
    A more realistic test case would include building the connectivity,
    but that would have the same overhead for all test programs.

    Four programs were compared:

        simple
            A stylized PDBx/mmCIF reader that is
            a step up from grepping the mmCIF file for ATOM and HETATM records.
            It parses the atom_site table headers
            to find the columns it is interested in,
            scans the first row of the atom_site table for the column offsets,
            and then extracts the atomic data.
            And it stops scanning file after reading atom_site table,
            so it only works for files with just one data block.

        readcif
            A fully conformant CIF reader that can switch between the
            PDBx/mmCIF stylized parsing and traditional parsing that
            tokenizes the input.  It uses a callbacks for each table
            that an application wants parsed, and callbacks for individual
            columns.  It knows the order in which tables need to be parsed,
            so it can skip tables and later jump back to reparse a table
            if necessary.
            Works for files with multiple data blocks.

            Two variations are benchmarked, a version that takes advantage
            of PDBx/mmCIF styling, and one that just uses the default
            tokenizing code.

        `cifparse-obj`_  V7-1-05
            The PDB's example mmCIF parser that tokenizes the input
            and saves it as tables for later processing.  No backtracking
            is needed, since everything is saved.  It can also write mmCIF
            files.
            Works for files with multiple data blocks.

        ucif_ svn revision 18662, 2013-11-19
            Part of "`iotbx.cif`_: a comprehensive CIF toolbox".  Tokenizes
            the input file and has a virtual function for are CIF loops,
            and a virtual function for table data items that are not in a loop.
            Uses a lot of memory, but unsure if it saves everything like
            cifparse-obj, or it's just a single pass through the file.
            Works for files with multiple data blocks.


    And the results were:

        .. tabularcolumns:: |C|L|J|J|J|J|

        +----------------+-------------+------------+------------+------------+-------------+
        | | program name | | PDB ID    | | 9rsa.cif | | 2kzt.cif | | 2kox.cif | | 3j3q.cif  |
        | | / code size  | | # of atoms| | 2106     | | 203816   | | 787840   | | 2440800   |
        |                | | file size | | 260 KiB  | | 25 MiB   | | 76 MiB   | | 254 MiB   |
        +================+=============+============+============+============+=============+
        |                | time        |   633 usec | .0353 sec  | .127 sec   | .413 sec    |
        | | simple       +-------------+------------+------------+------------+-------------+
        | | 15 KiB       | memory      |  12.8 MiB  |  48.4 MiB  |  136 MiB   |  458 MiB    |
        +----------------+-------------+------------+------------+------------+-------------+
        | | readcif      | time        |  988 usec  | .0447 sec  | .147 sec   | .497 sec    |
        |   stylized     +-------------+------------+------------+------------+-------------+
        | | 87 KiB       | memory      |  12.8 MiB  |  48.5 MiB  |  136 MiB   |  458 MiB    |
        +----------------+-------------+------------+------------+------------+-------------+
        | | readcif      | time        |  2248 usec |  .160 sec  | .553 sec   | 1.81 sec    |
        |   tokenized    +-------------+------------+------------+------------+-------------+
        | | 83 KiB       | memory      |  12.8 MiB  |  48.5 MiB  |  136 MiB   |  458 MiB    |
        +----------------+-------------+------------+------------+------------+-------------+
        |                | time        | 36951 usec |  3.18 sec  | 10.1 sec   | 33.8 sec    |
        | | cifparse-obj +-------------+------------+------------+------------+-------------+
        | | 514 KiB      | memory      |  15.8 MiB  |   319 MiB  |  905 MiB   | 2.98 GiB    |
        +----------------+-------------+------------+------------+------------+-------------+
        |                | time        | 48602 usec |  5.46 sec  | 17.2 sec   | | *out of*  |
        | | ucif         +-------------+------------+------------+------------+ | *memory*  |
        | | 184 KiB      | memory      |  28.2 MiB  |  1.39 GiB  | 4.51 GiB   | |           |
        +----------------+-------------+------------+------------+------------+-------------+

        The time is lowest time of 20 consecutive runs.
        Memory use is the peak memory use.

Discussion
~~~~~~~~~~

**cifparse-obj** and **ucif** are fundamentally slower because they convert
every data value into a C++ string, a dynamically allocated resource.
**readcif** also tokenizes the input, but avoids this overhead
by returning pointers to the start and ending characters of a data value.

As expected,
the **simple** code is the fastest for stylized PDBx/mmCIF files.
The **readcif** code for stylized PDBx/mmCIF files is next best
at ~1.2 times slower.
The fully tokenizing **readcif** code is ~3.6 times slower
than the sylized code and ~4.5 times slower than the **simple** code.
The **cifparse-obj** code is ~63 times slower than the stylized **readcif** code
and consumes more memory -- this is expected because it saves all of the data.
**ucif** is ~110 times slower and consumes way more memory -- this was
unexpected and deserves a closer look by the ucif developers.

Further Work
~~~~~~~~~~~~

It should be possible to speed up **readcif** a little bit more
by exposing more of the tokenizing internals to the parsing code
at the expense of having to write separate code for PDB mmCIF files.
But **readcif** is already close to optimal,
and it is unclear if any other improvements would be noticible
once connectivity and other derived information is computed.

Benefits of `PDBx/mmCIF Styling`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is currently not possible to robustly detect if a mmCIF file is stylized
or not.
It is likely that it stylized if the filename looks like a PDB identifier
followed by ``.cif``
and the dictionary is mmcif_pdbx.dic version 4 or newer.
But that guess could be wrong, and if it is wrong,
there is no indication of that fact that the input is corrupted.
As of 10 June 2014,
the above heuristic appears to work for the mmCIF files for PDB entries
appears to work.
However, the PDB's `large structure examples
<http://mmcif.wwpdb.org/docs/large-pdbx-examples/index.html>`_
files have the numbers in tables right-justified
instead of left-justified, so the stylized reading might fail.
Luckily, those file names are not a 4-character PDB identifier.

Looking at one test case, 3j3q.cif, let's examine the benefits of
various PDBx/mmCIF styling rules:

    .. tabularcolumns:: |L|J|J|

    +-----------------------------------------+-----------+---------+
    |                                         | 3j3q.cif  | Speedup |
    +=========================================+===========+=========+
    | fully tokenized                         | 1.81 sec  | 1x      |
    +-----------------------------------------+-----------+---------+
    | with tags/keywords at start of line     | 1.73 sec  | 1.05x   |
    +-----------------------------------------+-----------+---------+
    | with fixed columns                      | 0.603 sec | 3.00x   |
    +-+---------------------------------------+-----------+---------+
    | | \+ fixed length rows (trailing spaces)| 0.594 sec | 3.05x   |
    +-+---------------------------------------+-----------+---------+
    | | \+ tables terminated with comment     | 0.570 sec | 3.18x   |
    +-+---------------------------------------+-----------+---------+
    | with everything                         | 0.485 sec | 3.73x   |
    +-+---------------------------------------+-----------+---------+

Appendix: PDBx/mmCIF Styling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PDBx/mmCIF files are formatted for fast parsing.
This can be taken advantage of for speedier extraction of needed data.

    Outside of a data table:

        1. CIF keywords and data tags only appear immediately
           after an ASCII newline.

        2. CIF keywords are in lowercase.

        3. Data tags are case sensitive (category names and item names
           are mixed-case as specified in mmcif_pdbx.dic).

    Inside a data table:

        1. If the data values for each row can't fit on one line
           (due to a multiline string), then the first row is split
           into multiple lines.  Needed to robustly fallback to tokenizing
           the input.

        2. All columns are left-aligned.  Needed to robustly figure out
           column boundaries.

        3. All rows have trailing spaces so they are the same length.
           Optimization to speed up advancing to the next row.

        4. Rows are terminated by a comment line.
           Optimization to detect the end of a table.
