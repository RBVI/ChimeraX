// vi: set expandtab ts=4 sw=4:
/*
 * Copyright (c) 2014 The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions, and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions, and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *   3. Redistributions must acknowledge that this software was
 *      originally developed by the UCSF Resource for Biocomputing,
 *      Visualization, and Informatics with support from the National
 *      Institute of General Medical Sciences, grant P41-GM103311.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
 *   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *   PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY
 *   OF CALIFORNIA BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef READCIF_H
# define READCIF_H

# include <string>
# include <vector>
# include <unordered_map>
# include <unordered_set>
# include <functional>
# include <algorithm>
# include <stdexcept>
# include <limits>

// readcif -- parse CIF and mmCIF files with minimal overhead
//
// CIF files may contain multiple data blocks.
// Each data block is a collection of tables.  A callback function must
// be given for each table the programmer wishes to parse.

namespace readcif {

// ASCII HT (9), LF (10), CR (13), and SPACE (32)
// are the only whitespace characters recognized in CIF files.
// ASCII NUL is both not is_whitespace and not is_not_whitespace.

extern const int Whitespace[];
extern const int NotWhitespace[];

inline int
is_whitespace(char c)
{
    return Whitespace[(unsigned char) c];
}

inline int
is_not_whitespace(char c)
{
    return NotWhitespace[(unsigned char) c];
}

// non-error checking replacement for the standard library's strtof
// for non-scientific notation
// returns NaN if not a floating point number
inline float str_to_float(const char* s)
{
    bool saw_digit = false;
    bool neg = false;
    float fa = 0, v = 0;
    for (;;) {
        char c = *s;
        if (c >= '0' && c <= '9') {
            saw_digit = true;
            if (fa) {
                v += fa * (c - '0');
                fa *= 0.1f;
            } else
                v = 10 * v + (c - '0');
        }
        else if (c == '.')
            fa = 0.1f;
        else if (c == '-')
            neg = true;
        else
            break;
        s += 1;
    }
    if (saw_digit)
        return (neg ? -v : v);
    return std::numeric_limits<float>::quiet_NaN();
}

// non-error checking replacement for the standard library's atoi/strtol
// returns zero if not an integer
inline int str_to_int(const char* s)
{
    bool neg = (*s == '-');
    int v = 0;
    if (neg)
        s += 1;

    for (;;) {
        char c = *s;
        if (c >= '0' && c <= '9')
            v = 10 * v + (c - '0');
        else
            break;
        s += 1;
    }
    return (neg ? -v : v);
}

typedef std::vector<std::string> StringVector;

class CIFFile {
public:
    // If subclassed to with the functions necessary to hold the
    // parsed data, then the constructor of the SUBCLASS would call
    // register_category as follows:
    //
    //  using std::placeholder;
    //  register_category("atom_site", 
    //      std::bind(&SUBCLASS::parse_atom_site, this, _1, _2, _3));
    CIFFile();

    // Use register_category to indicate which categories should be parsed.
    // The ParseCategory's arguments are the category name, the list
    // of tags in the category (without the category prefix), and
    // whether the category is given in a loop or not.
    typedef std::function<void (bool in_loop)> ParseCategory;
    void register_category(const std::string& category,
            ParseCategory callback, 
            const StringVector& dependencies = StringVector());

    // The parsing functions
    void parse_file(const char* filename);  // open file and parse it
    void parse(const char* buffer); // null-terminated whole file

    // Indicate that CIF file follows the PDBx/mmCIF style guide
    // with lowercase keywords and tags at beginning of lines
    bool PDB_style() const { return stylized; }
    void set_PDB_style(bool stylized) { this->stylized = stylized; }

    // Indicate that the next CIF table uses PDBx/mmCIF style
    // fixed column widths
    bool PDB_fixed_columns() const { return fixed_columns; }
    void set_PDB_fixed_columns(bool fc) { fixed_columns = fc; }

    // version() returns the version of the CIF file if it is given.
    // For mmCIF files it is typically empty.
    const std::string&  version() const;  // CIF version of current parse

    // The category callback functions should call parse_row()
    // to parse the values for columns it is interested in.  If in a loop,
    // parse_row() should be called until it returns false, or to skip the
    // rest of the values, just return from the category callback.
    typedef std::function<void (const char* start, const char* end)>
        ParseValue;
    struct ParseColumn {
        int column;
        bool need_end;
        ParseValue func;
        ParseColumn(int c, bool n, ParseValue f):
            column(c), need_end(n), func(f) {}
    };
    typedef std::vector<ParseColumn> ParseValues;
    // ParseValues will be sorted in ascending order, the first time
    // parse_row is called for a category
    bool parse_row(ParseValues& pv);

    // Return current category.
    const std::string& category() const;

    // Return current block code
    const std::string& block_code() const;

    // Return current category column tags.
    const StringVector& tags() const;

    // Return current line number
    size_t line_number() const;

    // Convert tag to column position.
    int get_column(const char* tag, bool required=false);

    // return text + " on line #"
    std::runtime_error error(const std::string& text);
protected:
    // data_block is called whenever a new data block is found.
    // Defaults to being ignored.
    virtual void data_block(const std::string& name);

    // save_fame is called save frame header or terminator is found.
    // Defaults to throwing an exception.
    virtual void save_frame(const std::string& code);

    // global_block is called whenever the global_ keyword is found.
    // Defaults to throwing an exception.
    virtual void global_block();

    // reset_parse is called whenever parse is called
    virtual void reset_parse();

    // reset_parse is called whenever parse has successfully finished
    virtual void finished_parse();
private:
    void internal_reset_parse();
    void internal_parse(bool one_table=false);
    void next_token();
    void stylized_next_keyword(bool tag_okay=false);
    std::vector<int> find_column_offsets();
    struct CategoryInfo {
        ParseCategory func;
        StringVector dependencies;
        CategoryInfo(ParseCategory f, const StringVector& d):
                        func(f), dependencies(d) {}
    };
    typedef std::unordered_map<std::string, CategoryInfo> Categories;
    Categories  categories;
    StringVector    categoryOrder;  // order categories were registered in

    // parsing state
    bool        parsing;
    bool        stylized;   // true for PDBx/mmCIF keyword style
    bool        fixed_columns;  // true for PDBx/mmCIF fixed column widths
    std::string version_;   // version given in CIF file
    const char* whole_file;
    std::string current_data_block;
    std::string current_category;
    StringVector    current_tags;   // data tags without category name
    StringVector    values;
    bool        first_row;
    std::vector<int> columns;   // for stylized files
    // backtracking support:
    std::unordered_set<std::string> seen;
    struct StashInfo {
        const char* start;  // start of first token
        size_t lineno;      // the line it was on
        StashInfo(const char* s, size_t l): start(s), lineno(l) {}
    };
    std::unordered_map<std::string, StashInfo> stash;
    void process_stash();

    // lexical state
    bool DDL_v2;        // true if '.' separates category in data names
    enum Token {
        T_SOI, /* Start Of Input */
        // keywords
        T_DATA, T_GLOBAL, T_LOOP, T_SAVE, T_STOP,
        // other
        T_TAG, T_VALUE, T_LEFT_BRACKET, T_RIGHT_BRACKET,
        T_EOI /* End Of Input */
    };
    Token       current_token;
    bool current_is_keyword() {
        return current_token >= T_DATA && current_token <= T_STOP;
    }
    // current_value for T_DATA, T_TAG, T_VALUE
    std::string current_value();
    const char* current_value_start;
    const char* current_value_end;
    std::string current_value_tmp;
    const char* line;       // current line being tokenized
    size_t      lineno;     // current line number
    const char* pos;        // current position in line/file (index)
    bool        save_values;    // true if T_VALUE values are needed
};

inline const std::string&
CIFFile::version() const
{
    return version_;
}

inline const std::string&
CIFFile::category() const
{
    return current_category;
}

inline const std::string&
CIFFile::block_code() const
{
    return current_data_block;
}

inline const StringVector&
CIFFile::tags() const
{
    return current_tags;
}

inline size_t
CIFFile::line_number() const
{
    return lineno;
}

} // namespace readcif

#endif
