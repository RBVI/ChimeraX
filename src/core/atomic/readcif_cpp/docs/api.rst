..  vim: set expandtab shiftwidth=4 softtabstop=4:

readcif -- a C++11 CIF and mmCIF parser
=======================================

    :Author: Greg Couch
    :Organization: RBVI_, `University of California at San Francisco`_
    :Contact: gregc@cgl.ucsf.edu
    :Copyright: © Copyright 2014-2017 by the Regents of the University of California.  All Rights reserved.
    :Last modified: 2017-3-9

.. _RBVI: http://www.rbvi.ucsf.edu/
.. _University of California at San Francisco: http://www.ucsf.edu/

**readcif** is a `C++11`_ library for quickly extracting data
from mmCIF_ and CIF_ files.
It fully conforms to the CIF 1.1 standard for data files,
and can be easily extended to handle CIF dictionaries.
In addition, it supports stylized PDBx/mmCIF files for even
quicker parsing.

.. _C++11: http://isocpp.org/wiki/faq/cpp11
.. _CIF: http://www.iucr.org/resources/cif
.. _mmCIF: http://mmcif.wwpdb.org/

License
-------

The **readcif** library is available with an open source license:

    Copyright © 2014 The Regents of the University of California.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

       1. Redistributions of source code must retain the above copyright
          notice, this list of conditions, and the following disclaimer.

       2. Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions, and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.

       3. Redistributions must acknowledge that this software was
          originally developed by the UCSF Resource for Biocomputing,
          Visualization, and Informatics with support from the National
          Institute of General Medical Sciences, grant P41-GM103311.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY
    OF CALIFORNIA BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Usage
-----

CIF files are essentially a text version of a database.
Each table in the database corresponds to a category,
with named columns, and the rows contain the values.
CIF tags are a concatenation of the category name and the column name.

**readcif** provides a base class, :cpp:class:`CIFFile`,
that should be subclassed to implement an application's specific needs.
Virtual functions are used to support CIF reserved words,
that way the application can choose what to do if there is more than one
data block or handle dictionaries.
And callback functions are used to extract the data the
application wants from a category.
Finally, the category callback functions need to provide a set of
callback functions to parse the value of interesting columns.

So, in pseudo-code, an application's parser would look like::

    class ExtractCIF: public readcif::CIFFile {
    public:
        ExtractCIF() {
            initialize category callbacks
        }
        in each category callback:
            create parse value callback vector for interesting columns
            while (parse_row(parse_value_callbacks))
                continue;
    };

and be used by::

    ExtractCIF extract;

    const char* whole_file = ....
    extract.parse_file(filename)

See the associated :download:`example code <readcif_example.cpp>` file
for a working example
that reads a subset of the atom_site data from a PDB mmCIF file.

PDBx/mmCIF Styling
------------------

`PDBx/mmCIF`_ files from the World-Wide PDB are formatted for fast parsing.
If a CIF file is known to use `PDBx/mmCIF`_ stylized formatting,
then parsing can be up to 4 times faster.
**readcif** supports taking advantage of the PDBx/mmCIF styling with an API
and code.

PDBx/mmCIF styling constrains the CIF format by:

    Outside of a category:

        1. CIF reserved words and tags only appear immediately
           after an ASCII newline.

        2. CIF reserved words are in lowercase.

        3. Tags are case sensitive (category names and item names
           are expected to match the case given in the associated dictionary,
           *e.g.*, mmcif_pdbx.dic_).

        Support for this is controlled with the
        :cpp:func:`CIFFile::set_PDBx_keywords` function.

    Inside a category:

        1. All columns are left-aligned.

        2. Each row of data has all of the columns.

        3. All rows have trailing spaces so they are the same length.

        4. The end of the category's data values
           is terminated by a comment line.

        Support for this is controlled with the
        :cpp:func:`CIFFile::set_PDBx_fixed_width_columns` function.

..        1. If the data values for each row can't fit on one line
           (due to a multiline string), then the first row is split
           into multiple lines.

The :download:`example code <readcif_example.cpp>` shows how a derived class would turn on stylized parsing.
The **audit_conform** category is examined for explicit references to **pdbx_keywords** and **pdbx_fixed_width_columns**.
And if they are present, they control the options.
Otherwise, a heuristic is used: if the **dict_name** is "mmcif_pdbx.dic_"
and **dict_version** is greater than 4,
then it is assumed that there is keyword styling and that the **atom_site** and the **atom_site_anistrop** categories have fixed width columns.

.. _mmcif_pdbx.dic: http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx.dic/Index/
.. _PDBx/mmCIF: http://mmcif.wwpdb.org/docs/faqs/pdbx-mmcif-faq-general.html

C++ API
-------

    All of the public symbols are in the **readcif** namespace.

.. cpp:type:: StringVector

    A std::vector of std::string's.

.. cpp:function:: int is_whitespace(char c)

    **is_whitespace** and **is_not_whitespace** are
    inline functions to determine if a character is CIF whitespace or not.
    They are similar to the C/C++ standard library's **isspace** function,
    but only recognize ASCII HT (9), LF (10), CR (13), and SPACE (32)
    as whitespace characters.  They are not inverses because
    ASCII NUL (0) is both not is_whitespace and not is_not_whitespace.

.. cpp:function:: int is_not_whitespace(char c)

    See :cpp:func:`is_whitespace`.

.. cpp:function:: double str_to_float(const char* s)

    Non-error checking inline function to convert a string to a
    floating point number.  It is similar to the C/C++ standard library's
    **atof** function, but returns NaN if no digits are found.
    Benchmarked by itself, it is slower than **atof**,
    but is empirically much faster when used in shared libraries.
    This is probably due to CPU cache behavior, but needs further investigation.

.. cpp:function:: int str_to_int(const char* s)

    Non-error inline function to convert a string to an integer.
    It is similar to the C/C++ standard library's **atoi** function.
    Same rational for use as :cpp:func:`str_to_float`.
    Returns zero if no digits are found.

.. cpp:class:: CIFFile

    The CIFFile is designed to be subclassed by an application to extract
    the data the application is interested in.

    Public section:

        .. cpp:type:: ParseCategory

            A typedef for **std::function<void (bool in_loop)>**.

        .. cpp:function:: void register_category(const std::string& category, \
            ParseCategory callback, \
            const StringVector& dependencies = StringVector())

            Register a callback function for a particular category.

            :param category: name of category
            :param callback: function to retrieve data from category
            :param dependencies: a list of categories that must be parsed
                before this category.

            A null callback function, removes the category.
            Dependencies must be registered first.
            A category callback function can find out which category
            it is processing with :cpp:func:`category`.

        .. cpp:function:: void set_unregistered_callback(ParseCategory callback)

            Set callback function that will be called
            for unregistered categories.

        .. cpp:function:: void parse_file(const char* filename)

            :param filename: Name of file to be parsed

            If possible, memory-map the given file to get the buffer
            to hand off to :cpp:func:`parse`.  On POSIX systems,
            files whose size is a multiple of the system page size,
            have to be read into an allocated buffer instead.

        .. cpp:function:: void parse(const char* buffer)

            Parse the input and invoke registered callback functions

            :param buffer: Null-terminated text of the CIF file

            The text must be terminated with a null character.
            A common technique is to memory map a file
            and pass in the address of the first character.
            The whole file is required to simplify backtracking
            since data tables may appear in any order in a file.
            Stylized parsing is reset each time :cpp:func:`parse` is called.

        .. cpp:function:: void set_PDBx_keywords(bool stylized)

            Turn on and off PDBx/mmCIF keyword styling as described in
            `PDBx/mmCIF Styling`.

            :param stylized: if true, assume PDBx/mmCIF keyword style

            This is reset every time :cpp:func:`CIFFile::parse` 
            or :cpp:func:`CIFFile::parse_file` is called.
            It may be switched on and off at any time,
            *e.g.*, within a particular category callback function.

        .. cpp:function:: bool PDBx_keywords() const

            Return if the PDBx_keywords flag is set.
            See :cpp:func:`set_PDBx_keywords`.

        .. cpp:function:: void set_PDBx_fixed_width_columns(const std::string& category)

            Turn on `PDBx/mmCIF`_ fixed width column parsing for a given
            category as described in `PDBx/mmCIF Styling`.

            :param category: name of category

            This option must be set in each category callback that is needed.
            This option is ignored if :cpp:func:`PDBx_keywords` is false.
            This is not a global option because there is no reliable way
            to detect if the preconditions are met for each record without
            losing all of the speed advantages.

        .. cpp:function:: bool has_PDBx_fixed_width_columns() const

            Return if there were any fixed width column categories specified.
            See :cpp:func:`set_PDBx_fixed_width_columns`.

        .. cpp:function:: bool PDBx_fixed_width_columns() const

            Return if the current category has fixed width columns.
            See :cpp:func:`set_PDBx_fixed_width_columns`.

        .. cpp:function:: int get_column(const char \*name, bool required=false)
            
            :param tag: column name to search for
            :param required: true if tag is required

            Search the current categories tags to figure out which column
            the name corresponds to.
            If the name is not present,
            then -1 is returned unless it is required,
            then an error is thrown.

        .. cpp:type:: ParseValue1
         
            **typedef std::function<void (const char\* start)> ParseValue1;**

        .. cpp:type:: ParseValue2
         
            **typedef std::function<void (const char\* start, const char\* end)> ParseValue2;**

        .. cpp:class:: ParseColumnn
        
            .. cpp:member:: int column_offset

                The column offset for a given tag,
                returned by :cpp:func:`get_column`.

            .. cpp:member:: bool need_end

                **true** if the end of the column needed -- not needed for numbers,
                since all columns are terminated by whitespace.

            .. cpp:member:: ParseValue1 func1

                The function to call if :cpp:member:`need_end` is **false**.

            .. cpp:member:: ParseValue2 func2

                The function to call if :cpp:member:`need_end` is **true**.

            .. cpp:function:: ParseColumn(int c, ParseValue1 f)

                Set :cpp:member:`column_offset` and :cpp:member:`func1`.

            .. cpp:function:: ParseColumn(int c, ParseValue2 f)

                Set :cpp:member:`column_offset` and :cpp:member:`func2`.

        .. cpp:type:: ParseValues

            **typedef std::vector<ParseColumn> ParseValues;**

        .. cpp:function:: bool parse_row(ParseValues& pv)

            Parse a single row of a table

            :param pv: The per-column callback functions
            :return: if a row was parsed

            The category callback functions should call :cpp:func:`parse_row`:
            to parse the values for columns it is interested in.  If in a loop,
            :cpp:func:`parse_row`: should be called until it returns false,
            or to skip the rest of the values, just return from the category
            callback.
            The first time :cpp:func:`parse_row` is called for a category,
            *pv* will be sorted in ascending order.
            Columns with negative offsets are skipped.

        .. cpp:function:: StringVector& parse_whole_category()

            Return complete contents of a category as a vector of strings.

            :return: vector of strings

        .. cpp:function:: void parse_whole_category(ParseValue2 func)

            Tokenize complete contents of category
            and call function for each item in it.

            :param func: callback function

        .. cpp:function:: const std::string& version()

            :return: the version of the CIF file if it is given

            For mmCIF files it is typically empty.

        .. cpp:function:: const std::string& category()

           :return: the category that is currently being parsed

           Only valid within a :cpp:type:`ParseCategory` callback.

        .. cpp:function:: const std::string& block_code()

           :return: the data block code that is currently being parsed

           Only valid within a :cpp:type:`ParseCategory` callback
           and :cpp:func:`finished_parse`.

        .. cpp:function:: const StringVector& colnames()

           :return: the set of column names for the current category

           Only valid within a :cpp:type:`ParseCategory` callback.

        .. cpp:function:: bool multiple_rows() const

            :return: if current category may have multiple rows 

        .. cpp:function:: size_t line_number() const

            :return: current line number

        .. cpp:function:: std::runtime_error error(const std::string& text)

            :param text: the error message
            :return: a exception with " on line #" appended
            :rtype: std::runtime_error

            Localize error message with the current line number
            within the input.
            # is the current line number.

    Protected section:

        .. cpp:function:: void data_block(const std::string& name)

            :param name: name of data block

            **data_block** is a virtual function that is called whenever
            a new data block is found.
            Defaults to being ignored.
            Replace in subclass if needed.

        .. cpp:function:: void save_frame(const std::string& code)

            :param code: the same frame code

            **save_fame** is a virtual function that is called
            when a save frame header or terminator is found.
            It defaults to throwing an exception.
            It should be replaced if the application
            were to try to parse a CIF dictionary.

        .. cpp:function:: void global_block()

            **global_block** is a virtual function that is called whenever
            the **global\_** reserved word is found.
            It defaults to throwing an exception.
            In CIF files, **global\_** is unused.
            However, some CIF-like files, *e.g.*, the CCP4 monomer library,
            use the global\_ keyword.

        .. cpp:function:: void reset_parse()

            **reset_parse** is a virtual function that is called whenever
            the parse function is called.
            For example, PDB stylized parsing can be turned on here.

        .. cpp:function:: void finished_parse()

            **finished_parse** is a virtual function that is called whenever
            the parse function has successfully finished parsing.
