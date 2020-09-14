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
 *      Institutes of Health R01-GM129325.
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
# include <cmath>

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

// Non-error checking replacement for the standard library's strtod.
// Returns NaN if not a floating point number.
// While this routine is slower than atof by iself, it is faster than
// atof when inlined within a parser. 
inline double str_to_float(const char* s)
{
    bool saw_digit = false;
    bool saw_decimal = false;
    bool saw_exp = false;
    int decimals = 0;
    bool neg = false;
    bool exp_neg = false;
    int exp = 0;
    long long iv = 0;
    double fv;
    for (; *s; ++s) {
    char c = *s;
    switch (c) {
        default:
            break;
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            if (saw_exp) {
                exp = exp * 10 + (c - '0');
                continue;
            }
            saw_digit = true;
            if (saw_decimal)
                decimals -= 1;
            iv = iv * 10 + (c - '0');
            continue;
        case '.':
            saw_decimal = true;
            continue;
        case '-':
            if (saw_exp)
                exp_neg = true;
            else
                neg = true;
            continue;
        case '+':
            if (saw_exp)
                continue;
            break;
        case 'E': case 'e':
            saw_exp = true;
            continue;
        }
        break;
    }
    if (saw_exp) {
        if (exp_neg)
            decimals -= exp;
        else
            decimals += exp;
    }
    switch (decimals) {
        case 9: fv = iv * 1e9; break;
        case 8: fv = iv * 1e8; break;
        case 7: fv = iv * 1e7; break;
        case 6: fv = iv * 1e6; break;
        case 5: fv = iv * 1e5; break;
        case 4: fv = iv * 1e4; break;
        case 3: fv = iv * 1e3; break;
        case 2: fv = iv * 1e2; break;
        case 1: fv = iv * 1e1; break;
        case 0: fv = iv * 1e0; break;
        case -1: fv = iv * 1e-1; break;
        case -2: fv = iv * 1e-2; break;
        case -3: fv = iv * 1e-3; break;
        case -4: fv = iv * 1e-4; break;
        case -5: fv = iv * 1e-5; break;
        case -6: fv = iv * 1e-6; break;
        case -7: fv = iv * 1e-7; break;
        case -8: fv = iv * 1e-8; break;
        case -9: fv = iv * 1e-9; break;
        default: fv = iv * std::pow(10, decimals); break;
    }
    if (saw_digit)
        return (neg ? -fv : fv);
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
    virtual ~CIFFile() {}

    // Use register_category to indicate which categories should be parsed.
    // The ParseCategory's arguments are the category name, the list
    // of tags in the category (without the category prefix), and
    // whether the category is given in a loop or not.
    typedef std::function<void ()> ParseCategory;
    void register_category(const std::string& category,
            ParseCategory callback, 
            const StringVector& dependencies = StringVector());

    // Set callback function for unregistered categories
    void set_unregistered_callback(ParseCategory callback);

    // The parsing functions
    void parse_file(const char* filename);  // open file and parse it
    void parse(const char* buffer); // null-terminated whole file

    // Indicate that CIF file follows the PDBx/mmCIF style guide
    // with lowercase keywords and tags at beginning of lines
    bool PDBx_keywords() const;
    void set_PDBx_keywords(bool stylized);

    // Indicate that CIF file follows the PDBx/mmCIF style guide
    // with fixed width columns within a category.
    void set_PDBx_fixed_width_columns(const std::string& category);

    // Return if there were any fixed width column categories specified.
    bool has_PDBx_fixed_width_columns() const;

    // version() returns the version of the CIF file if it is given.
    // For mmCIF files it is typically empty.
    const std::string&  version() const;  // CIF version of current parse

    // The category callback functions should call parse_row()
    // to parse the values for columns it is interested in.  If in a loop,
    // parse_row() should be called until it returns false, or to skip the
    // rest of the values, just return from the category callback.
    typedef std::function<void (const char* start)>
        ParseValue1;
    typedef std::function<void (const char* start, const char* end)>
        ParseValue2;
    struct ParseColumn {
        int column;
        bool need_end;
        ParseValue1 func1;
        ParseValue2 func2;
        ParseColumn(int c, ParseValue1 f):
            column(c), need_end(false), func1(f) {}
        ParseColumn(int c, ParseValue2 f):
            column(c), need_end(true), func2(f) {}
    };
    typedef std::vector<ParseColumn> ParseValues;
    // ParseValues will be sorted in ascending order, the first time
    // parse_row is called for a category.  Returns false if there is
    // no more data in table.
    bool parse_row(ParseValues& pv);

    // Return complete contents of a category as a vector of strings.
    StringVector& parse_whole_category();

    // Tokenize complete contents of category and Call func for each item in it
    void parse_whole_category(ParseValue2 func);

    // Return current category.
    const std::string& category() const;

    // Return current block code
    const std::string& block_code() const;

    // Return current category column names
    const StringVector& colnames() const;

    // Return if current category has multiple rows 
    bool multiple_rows() const;

    // Return current line number
    size_t line_number() const;

    // Convert tag to column position.
    int get_column(const char* name, bool required=false);

    // return text + " on line #"
    std::runtime_error error(const std::string& text, size_t lineno=0);
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
        std::string name;
        ParseCategory func;
        StringVector dependencies;
        CategoryInfo(const std::string &n, ParseCategory f, const StringVector& d):
                        name(n), func(f), dependencies(d) {}
    };
    typedef std::unordered_map<std::string, CategoryInfo> Categories;
    Categories  categories;
    StringVector    categoryOrder;  // order categories were registered in
    ParseCategory   unregistered;

    // parsing state
    bool        parsing;
    bool        stylized;   // true for PDBx/mmCIF keyword style
    std::string version_;   // version given in CIF file
    const char* whole_file;
    std::string current_data_block;
    std::string current_category;
    std::string current_category_cp;    // case-preserved category name
    StringVector    current_colnames;   // data tags without category name
    StringVector    current_colnames_cp;   // case-preserved colnames
    StringVector    values;
    bool        in_loop;
    bool        first_row;
    std::vector<int> columns;   // for stylized files
    std::unordered_set<std::string> use_fixed_width_columns;
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
        T_TAG, T_VALUE, T_LEFT_SQUARE_BRACKET, T_RIGHT_SQUARE_BRACKET,
        T_EOI /* End Of Input */
    };
    static const char* token_names[T_EOI + 1];
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
    if (!current_category_cp.empty())
        return current_category_cp;
    return current_category;
}

inline const std::string&
CIFFile::block_code() const
{
    return current_data_block;
}

inline const StringVector&
CIFFile::colnames() const
{
    if (!current_colnames_cp.empty())
        return current_colnames_cp;
    return current_colnames;
}

inline size_t
CIFFile::line_number() const
{
    return lineno;
}

inline bool
CIFFile::multiple_rows() const
{
    return in_loop;
}

inline void
CIFFile::set_unregistered_callback(ParseCategory callback)
{
    unregistered = callback;
}

inline bool
CIFFile::PDBx_keywords() const
{
	return stylized;
}

inline void
CIFFile::set_PDBx_keywords(bool stylized)
{
	this->stylized = stylized;
}

inline bool
CIFFile::has_PDBx_fixed_width_columns() const
{
    return !use_fixed_width_columns.empty();
}

} // namespace readcif

#endif
