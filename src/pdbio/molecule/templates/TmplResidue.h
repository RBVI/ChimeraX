#ifndef templates_TmplResidue
#define	templates_TmplResidue

#include <map>
#include <vector>
#ifdef UNPORTED
#include "TmplAtom.h"
#include "TmplMolecule.h"
#include "TAexcept.h"
#endif  // UNPORTED
#include "Residue_template.h"

class TmplAtom;
class TmplMolecule;

class TmplResidue {
	friend class TmplMolecule;
	void	operator=(const TmplResidue &);	// disable
		TmplResidue(const TmplResidue &);	// disable
		~TmplResidue();
public:
	void	add_atom(TmplAtom *element);
#ifdef UNPORTED
	void	removeAtom(TmplAtom *element);
	typedef std::vector<TmplAtom *> Atoms;
	inline Atoms atoms() const;
#endif  // UNPORTED
	typedef std::map<std::string, TmplAtom *> AtomsMap;
#ifdef UNPORTED
	inline const AtomsMap	&atomsMap() const;
	typedef std::vector<Symbol> AtomKeys;
	inline AtomKeys	atomNames() const;
#endif  // UNPORTED
	TmplAtom	*find_atom(const std::string &) const;
public:
	inline const std::string	name() const { return _name; }
#ifdef UNPORTED
	inline void			setType(Symbol t);
	inline bool			operator==(const TmplResidue &r) const;
	inline bool			operator<(const TmplResidue &r) const;
#endif  // UNPORTED
private:
	std::string	_name;
	AtomsMap	_atoms;
public:
	// return atoms that received assignments from the template
	std::vector<TmplAtom *>	template_assign(
				  void (*assign_func)(TmplAtom *, const char *),
				  const char *app,
				  const char *template_dir,
				  const char *extension
				) const;
	std::vector<TmplAtom *>	template_assign(
				  void (TmplAtom::*assign_func)(const char *),
				  const char *app,
				  const char *template_dir,
				  const char *extension
				) const;
	std::vector<TmplAtom *>	template_assign(tmpl_assigner, 
				  const char *app,
				  const char *template_dir,
				  const char *extension
				) const;
private:
	TmplAtom	*_chief, *_link;
	std::string	_description;
public:
	TmplAtom	*chief() const { return _chief; }
	void	chief(TmplAtom *a) { _chief = a; }
	TmplAtom	*link() const { return _link; }
	void	link(TmplAtom *a) { _link = a; }
	std::string	description() const { return _description; }
	void		description(const std::string &d) { _description = d; }
private:
#ifdef UNPORTED
	TmplResidue(TmplMolecule *, Symbol t, Symbol chain, int pos, char insert);
#endif  // UNPORTED
	TmplResidue(TmplMolecule *, const char *t);
};

#endif  // templates_TmplResidue
