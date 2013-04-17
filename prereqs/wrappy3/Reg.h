#ifndef Reg_h
# define Reg_h

# include <string>

class RegData;

// New code should use <pcrecpp.h> with UTF-8 support instead

class Reg {
	Reg(const Reg&);
	Reg& operator=(const Reg&);
public:
	static const Reg	white;
	static const Reg	integer;
	static const Reg	real;
	static const Reg	alpha;
	static const Reg	lowercase;
	static const Reg	uppercase;
	static const Reg	alphanum;
	static const Reg	identifier;
#ifndef OTF_USE_ENUM
	// Constructor Flags
	static const int EXTENDED = 0x01;
	static const int ICASE = 0x02;		// ignore case
	static const int NOSUB = 0x04;
	static const int NEWLINE = 0x08;	// newline is special
	// Execute Flags
	static const int NOTBOL = 0x01;		// not beginning of line
	static const int NOTEOL = 0x02;		// not end of line
#else
	enum {
		EXTENDED = 0x01, ICASE = 0x02, NOSUB = 0x04, NEWLINE = 0x08,
		NOTBOL = 0x01, NOTEOL = 0x02
	};
#endif

	struct Interval {
		size_t	start;
		size_t	end;
		Interval(size_t s, size_t e): start(s), end(e) {}
	};

	class Error {
		int	val;
		char	*buf;
		size_t	buflen;
	public:
		Error(const RegData *data, int i);
		Error(const Error &);
		Error	&operator=(const Error &);
		~Error();
		int		value() const { return val; }
		char const	*message() const { return buf; }
	};
	typedef std::string::size_type size_type;

	explicit Reg(char const *pattern, int flags = 0) throw (Error);
	explicit Reg(const std::string &pattern, int flags = 0) throw (Error);
	~Reg();

	// find returns std::string::npos if search string isn't in RE
	//	pos is starting position in search string
	//	length returns the length of the match
	//	flags can be NOTBOL and/or NOTEOL
	size_type	find(const std::string &search, size_type pos = 0,
					int flags = 0) const throw (Error);
	size_type	find(const std::string &search, size_type *length,
				size_type pos = 0, int flags = 0) const
				throw (Error);

	// matches returns true if whole search string matches RE
	//	pos is starting position in search string
	//	flags can be NOTBOL and/or NOTEOL
	bool		matches(const std::string &search, size_type pos = 0,
					int flags = 0) const throw (Error);

	bool 		exec(char const *search, int flags = 0) throw (Error);
	// using exec to lookd for the search string, these functions return
	// information on the submatches.  Match 0 is the whole match.
	size_t		numMatches() const throw();
	Interval	match(size_t m = 0) const throw (Error);
	size_t		matchLen(size_t m = 0) const throw (Error);
private:
	RegData	*data;
	RegData *init(char const *pattern, int flags = 0) throw (Error);
};

#endif
