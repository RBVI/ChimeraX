// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef util_cstring_cmp
#define util_cstring_cmp

#include <cstring>  // strcmp
#include <sstream>
#include <string>
#include <stdexcept>

namespace chutil {

// put character pack into stringstream

inline void unpack(std::stringstream& ss, char c) { ss << c; }

template<typename C, typename... CS>
void unpack(std::stringstream& ss, C c, CS... cs)
{
    ss << c;
    unpack(ss, cs...);
}

template <int len, char... description_chars>
class CString
{
public:
    typedef const char*  const_iterator;

private:
    char  _data[len];
    void  _report_error(const std::string& val) {
        std::stringstream error_msg;
        unpack(error_msg, description_chars...);
        error_msg << " \"";
        error_msg << val;
        error_msg << "\" too long, maximum ";
        error_msg << len-1;
        error_msg << " characters.";
        throw std::invalid_argument(error_msg.str());
    }

public:
    CString() { _data[0] = '\0'; }
    CString(const char* s) { *this = s; }
    CString(const char* s, std::ptrdiff_t length) {
        if (length+1 > len)
            _report_error(std::string(s, length));
        char* dest = _data;
        while (length-- > 0) {
            *dest++ = *s++;
        }
        *dest = '\0';
    }
    CString(std::initializer_list<char> s) {
        bool add_null = *(s.end()-1) != '\0';
        if (s.size() > len + (add_null ? 1 : 0))
            _report_error(std::string(s));
        int pos = 0;
        for (auto c: s) {
            _data[pos++] = c;
        }
        if (add_null)
            _data[pos] = '\0';
    }

    const_iterator  begin() const { return _data; }
    const_iterator  end() const { return _data + strlen(_data); }

    const char*  c_str() const { return _data; }
    void  clear() { _data[0] = '\0'; }
    const char*  data() const { return _data; }
    bool  empty() const { return _data[0] == '\0'; }
    std::size_t  length() const { return strlen(_data); }
    std::size_t  size() const { return length(); }

    void  operator=(const char* s) {
        auto end = _data + len;
        auto s_ptr = s;
        auto d_ptr = _data;
        for (; d_ptr < end; ++d_ptr, ++s_ptr) {
            *d_ptr = *s_ptr;
            if (*s_ptr == '\0')
                return;
        }
        _report_error(std::string(s));
    }
    operator const char*() const { return _data; }
    bool  operator==(const char* s) const { return strcmp(_data, s) == 0; }
    bool  operator==(const CString& s) const { return *this == s._data; }
    bool  operator==(char ch) const { return _data[0] == ch && _data[1] == '\0'; }
    bool  operator==(const std::string& s) const { return *this == s.c_str(); }
    bool  operator!=(const char* s) const { return strcmp(_data, s) != 0; }
    bool  operator!=(const CString& s) const { return *this != s._data; }
    bool  operator!=(char ch) const { return _data[0] != ch || _data[1] != '\0'; }
    bool  operator!=(const std::string& s) const { return *this != s.c_str(); }
    bool  operator<(const char* s) const { return strcmp(_data, s) < 0; }
    bool  operator<(const CString& cstr) const { return *this < cstr._data; }
    char&  operator[](const int i) { return _data[i]; }

    friend std::ostream&  operator<<(std::ostream& out, const CString& cstr) {
        out << cstr._data;
        return out;
    }

    size_t hash() const {
        // From Dan Bernstein post in comp.lang.c on 12/5/1990.
        size_t hash = 5381;
        int c;
        const char *str = _data;

        while ((c = *str++) != 0) {
            hash = ((hash << 5) + hash) ^ c;
        }
        return hash;
    }
};

}  // namespace chutil

namespace std {

template <int len, char... description_chars>
struct hash<chutil::CString<len, description_chars...>>
{
    size_t operator()(const chutil::CString<len, description_chars...>& cs) const
        {return cs.hash();}
};

}  // namespace std

#endif  // util_cstring_cmp
