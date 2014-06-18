readcif -- a C++11 CIF and mmCIF parser
=======================================

    :Author: Greg Couch
    :Organization: RBVI_, `University of California at San Francisco`_
    :Contact: gregc@cgl.ucsf.edu
    :Copyright: © Copyright 2014 by the Regents of the University of California.  All Rights reserved.
    :Last modified: 2014-6-17

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

Usage
-----

CIF files are essentially a text version of a database.
Each table in the database corresponds to a category,
and the columns of the table are labelled with tags,
and the rows contain the values.

**readcif** provides a base class, :cpp:class:`CIFFile`,
that should be subclassed to implement an application's specific needs.
Virtual functions are used to support CIF keywords,
that way the application can choose what to do if there is more than one
data block or handle dictionaries.
And callback functions are used to extract the data the
application wants from a category.
Finally, the category callback functions need to provide a set of
callback functions to parse the value of interesting tags.

So, in pseudo-code, an application's parser would look like::

    class ExtractCIF: public readcif::CIFFile {
    public:
        ExtractCIF() {
            initialize category callbacks
        }
        in each category callback:
            create parse value callback vector for interesting tags
            while (parse_row(parse_value_callbacks))
                continue;
    };

and be used by::

    ExtractCIF extract;

    const char* whole_file = ....
    extract.parse(whole_file)

See the associated :download:`readcif_example.cpp` file for a working example
that reads a subset of the atom_site data from a PDB mmCIF file.

Stylized PDBx/mmCIF files
-------------------------

If a CIF file is known to use PDBx/mmCIF stylized formatting,
then parsing can be up to 4 times faster.
:cpp:func:`set_PDBx_stylized` turns on and off stylized parsing,
and is reset every time :cpp:func:`CIFFile::parse` is called.
Stylized PDBx/mmCIF file parsing may be switched on and off at
any time, *e.g.*, within a particular category callback function.

The example code turns on stylized parsing if the **audit_conform.dict_name**
is mmcif_pdbx.dic_ and **audit_conform.dict_version** is greater than 4.
This is hack that is needed until the PDB adds
an explicit tag and value near the beginning
of a CIF file that indicates that its tables are stylized.

.. _mmcif_pdbx.dic: http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx.dic/Index/

PDBx/mmCIF Styling
~~~~~~~~~~~~~~~~~~

PDBx/mmCIF files are formatted for fast parsing.
readcif expects the following syntax for stylized files:

    Outside of a data table:

        1. CIF keywords and data tags only appear immediately
           after an ASCII newline.

        2. CIF keywords are in lowercase.

        3. Data tags are case sensitive (category names and item names
           are mixed-case as specified in mmcif_pdbx.dic_).

    Inside a data table:

        1. If the data values for each row can't fit on one line
           (due to a multiline string), then the first row is split
           into multiple lines.

        2. All columns are left-aligned.

        3. All rows have trailing spaces so they are the same length.

        4. Rows are terminated by a comment line.

C++ API
-------

    All of the public symbols are in the **readcif** namespace.

.. cpp:type:: StringVector

    A std::vector of std::string's.

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

        .. cpp:function:: void parse(const char* whole_file)

            Parse the input and invoke registered callback functions

            :param whole_file: all of the text of the CIF file

            The text must be terminated with a null character.
            A common technique is to memory map a file
            and pass in the address of the first character.
            The whole file is required to simplify backtracking
            since data tables may appear in any order in a file.
            Stylized parsing is reset each time :cpp:func:`parse` is called.

        .. cpp:function:: void set_PDBx_stylized(bool stylized)

            Turn on and off PDBx/mmCIF stylized parsing

            :param stylized: true to use PDBx/mmCIF stylized parsing

            Indiate that CIF file follows the PDBx/mmCIF style guide
            and that the style can be followed to speed up lexical
            analysis of the CIF file.
            This can be set and unset for an individual category.

        .. cpp:function:: bool PDBx_stylized() const

            Return if the PDBx_stylized flag is set.
            See :cpp:func:`set_PDBx_stylized`.

        .. cpp:function:: int get_column(const char \*tag, bool required=false)
            
            :param tag: column tag to search for
            :param required: true if tag is required

            Search the current categories tags to figure out which column
            the tag corresponds to.
            If the tag is not present,
            then -1 is returned unless it is required,
            then an error is thrown.

        .. cpp:type:: ParseValue
         
            **typedef std::function<void (const char\* start, const char\* end)> ParseValue;**

        .. cpp:type: ParseColumnn
        
            .. cpp:member:: int column_offset

                The column offset for a given tag,
                returned by :cpp:func:`get_column`.

            .. cpp:member:: bool need_end

                **true** if the end of the column needed -- not needed for numbers,
                since all columns are terminated by whitespace.

            .. cpp:member:: ParseValue func

                The function to call.

        .. cpp:type:: ParseValues

            typedef std::vector<ParseColumn> ParseValues;

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

        .. cpp:function:: const std::string& version()

            :return: the version of the CIF file if it is given

            For mmCIF files it is typically empty.

        .. cpp:function:: const std::string& category()

           :return: the category that is currently being parsed

           Only valid within a :cpp:type:`ParseCategory` callback.

        .. cpp:function:: const StringVector& tags()

           :return: the set of column tags for the current category

           Only valid within a :cpp:type:`ParseCategory` callback.

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
            were to try to parse a dictionary.

        .. cpp:function:: void global_block()

            **global_block** is a virtual function that is called whenever
            the global\_ keyword is found.
            It defaults to throwing an exception.
            In CIF files, the global\_ keyword is reserved, but unused.
            However, some CIF-like files, *e.g.*, the CCP4 monomer library,
            use the global\_ keyword.
