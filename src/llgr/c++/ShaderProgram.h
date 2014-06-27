#ifndef Chimera_Shader_h
# define Chimera_Shader_h

# if defined(_MSC_VER) && (_MSC_VER >= 1020)
#  pragma once
# endif

# define GLEW_NO_GLU
# include <GL/glew.h>
# include <string>
# include <vector>

namespace llgr {

namespace internal {

class Texture;
class ShaderProgram;

class ShaderVariable
{
	// Used to describe uniform variables (for shader programs)
	// and vertex attribute variables (e.g., for colors).
public:
	// Every variable should has a name and type.
	enum Type {
		Float, Vec2, Vec3, Vec4,
		Int, IVec2, IVec3, IVec4,
		UInt, UVec2, UVec3, UVec4,
		Bool, BVec2, BVec3, BVec4,
		Mat2x2, Mat3x3, Mat4x4,
		Mat2x3, Mat3x2, Mat2x4, Mat4x2, Mat3x4, Mat4x3,
#ifdef HAVE_TEXTURE
		Sampler1D, Sampler2D, Sampler3D, SamplerCube, 
		Sampler1DShadow, Sampler2DShadow,
#endif
		Unknown
	};
	// ATTRIBUTE: name
	const std::string&	name() const;
	// ATTRIBUTE: type
	Type		type() const;

	int		location() const;
	unsigned	count() const;		// number of base_type elements
	Type		base_type() const;
	unsigned	byte_count() const;

	void		set_float(GLfloat f);
	void		set_floatv(const GLfloat* fv);
	void		set_float_matrixv(bool transpose, const GLfloat* fv);

	void		set_int(GLint i);
	void		set_intv(const GLint* iv);

	void		set_bool(GLint b);
	void		set_boolv(const GLint* bv);
#ifdef HAVE_TEXTURE
	Texture*	texture() const;
	void		set_texture(Texture* t);
#endif

	bool		has_value() const;
private:
	friend class ShaderProgram;
	ShaderVariable(ShaderProgram *sp, const std::string& n, Type t);
	~ShaderVariable();

	ShaderProgram	*sp_;
	std::string	name_;
	bool		transpose_;
	Type		type_;
	int		location_;
	unsigned char	*data;
	void	draw_uniform() const;
};

inline bool
ShaderVariable::has_value() const
{
	return data != NULL;
}

inline const std::string&
ShaderVariable::name() const
{
	return name_;
}

inline ShaderVariable::Type
ShaderVariable::type() const
{
	return type_;
}

inline int
ShaderVariable::location() const
{
	return location_;
}

class ShaderProgram
{
public:
	ShaderProgram(const std::string& vertex_shader, const std::string& fragment_shader, const std::string& attribute0_name);
	virtual	~ShaderProgram();

	void	setup() const throw ();
	void	cleanup() const throw ();

	typedef std::vector<ShaderVariable*>	Variables;
	// ATTRIBUTE: uniforms
	const Variables&	uniforms();
	ShaderVariable*	uniform(const std::string& name,
						bool exceptions = false) const;
	// ATTRIBUTE: attributes
	const Variables&	attributes();
	ShaderVariable*	attribute(const std::string& name,
						bool exceptions = false) const;
private:
	friend class ShaderVariable;
	typedef ShaderVariable Var;

	Variables	uniforms_, attributes_;
	GLuint		programObj;		// OpenGL display list(s)
	GLuint		vs, fs;			// vertex and fragment shaders
};

inline const ShaderProgram::Variables&
ShaderProgram::uniforms()
{
	return uniforms_;
}

inline const ShaderProgram::Variables&
ShaderProgram::attributes()
{
	return attributes_;
}

} // namespace internal

} // namespace llgr

#endif
