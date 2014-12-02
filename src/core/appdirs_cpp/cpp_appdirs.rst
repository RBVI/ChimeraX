=====================
C++ appdirs Interface
=====================

The cpp_addirs library provides an interface at the C++ level to the
values in Chimera's Python appdirs_ object.  You link against the library
with ``-L$(shlibdir) -lcpp_appdirs`` in your Makefile.  You use the library
by including ``cpp_appdirs/AppDirs.h`` in your code and calling
``cpp_appdirs::AppDirs::get()`` to get a const AppDirs instance.
That instance has six const string attributes that correspond to the
appdirs_ directory attributes and have the exact same names
(*e.g.* ``user_data_dir``).

.. _appdirs: https://pypi.python.org/pypi/appdirs/

The AppDirs class provides an important convenience function for 
generating file system paths based on the directory attributes
that uses the appropriate path separator character for the operating system:

.. cpp:function:: std::string AppDirs::form_path( \
	std::initializer_list<std::string> path_components) const

	Returns a path formed by joining the given strings with the
	appropriate path separator.  Example usage::

		std::string path = appdirs.form_path({appdirs.site_data_dir,
			"charge-database", "ATP.charges"});

