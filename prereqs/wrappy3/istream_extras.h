#ifndef istream_extras_h
# define istream_extras_h

namespace std {

// Add const character and const string input stream operators where the
// input has to match exactly.  If the input doesn't match, try to revert
// the input stream back to where the match started, so if the error state
// is cleared, then the reading of the input can be restarted at the same
// location.
//
// Example:
// 
// 	std::cin >> "KEYWORD";
// 	if (std::cin) {
// 		process_KEYWORD();
// 		continue;
//	}
// 	std::cin.clear();
// 	std::cin >> "OTHERWORD"
// 	if (std::cin) {
// 		process_OTHERWORD();
// 		continue;
//	}
// 	std::cerr << "error: expected either KEYWORD or OTHERWORD\n";

template <class _char_type, class _traits>
basic_istream<_char_type, _traits>&
operator>>(basic_istream<_char_type, _traits>& _is, const _char_type& _literal)
{
	typename basic_istream<_char_type, _traits>::sentry _se(_is);
	if (_se) {
		_char_type _c;
		if (_is.get(_c) && _c != _literal) {
			_is.unget();
			_is.setstate(ios_base::failbit);
		}
	}
	return _is;
}

template <class _char_type, class _traits>
basic_istream<_char_type, _traits>&
operator>>(basic_istream<_char_type, _traits>& _is, const _char_type* _s)
{
	typedef basic_istream<_char_type, _traits> _basic_istream;
	typename _basic_istream::sentry _se(_is);
	if (_se && _s) {
		_char_type _c;
		typename _basic_istream::off_type _count;
		for (_count = 0; *_s && _is.get(_c) && _c == *_s;
				++_s, ++_count)
			continue;
		if (*_s) {
			_is.seekg(-_count - 1, ios_base::cur);
			_is.setstate(ios_base::failbit);
		}
	}
	return _is;
}

template <class _char_type, class _traits, class _allocator>
basic_istream<_char_type, _traits>&
operator>>(basic_istream<_char_type, _traits>& _is,
		const basic_string<_char_type, _traits, _allocator>& _s)
{
	typedef basic_istream<_char_type, _traits> _basic_istream;
	typename _basic_istream::sentry _se(_is);
	if (_se) {
		for (typename basic_string<_char_type, _traits, _allocator>::const_iterator _i = _s.begin(), _e = _s.end(); _i != _e; ++_i) {
			_char_type _c;
			if (_is.get(_c) && _c == *_i)
				continue;
			_is.seekg(-(_i - _s.begin()) - 1, ios_base::cur);
			_is.setstate(ios_base::failbit);
			break;
		}
	}
	return _is;
}

} // namespace std

#endif
