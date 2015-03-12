// vi: set expandtab ts=4 sw=4:
#ifndef util_cstring_cmp
#define util_cstring_cmp

#include <cstring>  // strcmp
#include <iostream>
#include <string>
#include <stdexcept>

namespace chutil {

// comparison function for const char* in maps and so forth

template <int len>
class CString
{
public:
    typedef const char*  const_iterator;

private:
    char  _data[len];

public:
    CString() { _data[0] = '\0'; }
    CString(const char* s) { *this = s; }
    CString(const char* s, std::ptrdiff_t length) {
        if (length+1 > len)
            throw std::invalid_argument("String too long");
        char* dest = _data;
        while (length-- > 0) {
            *dest++ = *s++;
        }
        *dest = '\0';
    }
    CString(std::initializer_list<char> s) {
        if (s.size() > len)
            throw std::invalid_argument("String too long");
        int pos = 0;
        for (auto c: s) {
            _data[pos++] = c;
        }
    }

    const_iterator  begin() const { return _data; }
    const_iterator  end() const { return _data + strlen(_data); }

    void  clear() { _data[0] = '\0'; }
    bool  empty() const { return _data[0] == '\0'; }
    std::size_t  length() const { return strlen(_data); }
    std::size_t  size() const { return length(); }

    void  operator=(const char* s) {
        if (std::strlen(s) >= len)
            throw std::invalid_argument("String too long");
        std::strcpy(_data, s);
    }
    operator const char*() const { return _data; }
    bool  operator==(const char* s) const { return strcmp(_data, s) == 0; }
    bool  operator==(const CString& s) const { return *this == s._data; }
    bool  operator!=(const char* s) const { return strcmp(_data, s) != 0; }
    bool  operator!=(const CString& s) const { return *this != s._data; }
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

#endif  // util_cstring_cmp
