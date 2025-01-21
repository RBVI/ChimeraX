#######################################################################
# Process this file wth autoconf to produce a configure script.
#######################################################################
 
AC_PREREQ(2.50)
AC_INIT(xdrfile, 1.1.2, [gmx-users@gromacs.org])
AC_CONFIG_SRCDIR(src/xdrfile.c)
AC_CONFIG_AUX_DIR(config)
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE(tar-ustar)
AC_PREFIX_DEFAULT(/usr/local/)

AM_CONFIG_HEADER(config.h)
dnl This is the version info according to the libtool versioning system.
dnl It does *not* correspond to the release number.
SHARED_VERSION_INFO="4:0:0"
AC_SUBST(SHARED_VERSION_INFO)

# MPI builds on aix break with shared libs, so they are disabled by default.
AC_DISABLE_SHARED

#######################################################################
# Simple options and makefile variables
#######################################################################

### Fortran or not
AC_ARG_ENABLE(fortran,
 [  --Enable-fortran              use fortran (default on sgi,ibm,sun,axp)],, 
[case "${host_cpu}-${host_os}" in
  sparc*-solaris* | alpha*-* | rs6000*-aix* | powerpc*-aix* | mips*-irix*) enable_fortran=yes ;;
  *) enable_fortran=no ;;
 esac])
AM_CONDITIONAL([GMX_FORTRAN],[test "$enable_fortran" = "yes"])
if test "$enable_fortran" = "yes"; then
  AC_DEFINE(GMX_FORTRAN,,[Use Fortran innerloops])
fi
# always use CC for linking:
AC_SUBST(F77LINK,"\$(LINK)")


AC_ARG_ENABLE(all-static,
 [  --enable-all-static           make completely static binaries],, enable_all_static=no)
# Dont add the -all-static flag until after the compiler test.



############################################################
# Add some debug info: Who is building, and on what machine?
############################################################
AC_DEFINE_UNQUOTED([BUILD_TIME],["`date`"],[Date and time for build])
AC_DEFINE_UNQUOTED([BUILD_USER],["`whoami`@`hostname`"],[User doing build])
AC_DEFINE_UNQUOTED([BUILD_MACHINE],["`uname -srm`"],[Hardware and OS version for build host])



############################################################
# Checks for programs
############################################################

# normally we use the default compiler, but on alpha/linux we try the compaq ones first
case "${host_cpu}-${host_os}" in
  alpha*-linux*)
    cc_names="ccc cc gcc"
    f77_names="fort f77 g77"
    ;;
  *) 
    cc_names="cc icc xlc gcc"
    f77_names="f77 ifort xlf xlf77 cf77 fl32 g77 fort77 f90 xlf90 pgf77 cf77 fort fort77 pgf90"
    ;;
esac
# the (non-mpi) c compiler, which is also used for programs executed during build stage
AC_PROG_CC( $cc_names )
# Check for environment variable CC_FOR_BUILD before overwriting it
if test -z "$CC_FOR_BUILD"; then
  CC_FOR_BUILD=$CC
fi
AC_SUBST(CC_FOR_BUILD) 

if test "$enable_fortran" = "yes"; then 
  AC_PROG_F77( $f77_names )
  if test -z "$F77"; then
    AC_MSG_ERROR([No fortran compiler found])
  fi
fi

AM_PROG_AS( $CC )

# if we are using mpi, also get an MPICC. We cannot set that in the PROG_CC macro
# above, since autoconf checks that the created file can be executed. This would
# fail on platforms where MPI executables can only be run through a batchqueue.

if test "$enable_mpi" = "yes"; then
  AC_CHECK_PROGS(MPICC, mpxlc mpicc mpcc hcc, $CC)
# now change the normal cc to the MPI one - see the comment above.
  CC=$MPICC
  AC_MSG_CHECKING([whether the MPI cc command works]) # be paranoid
  AC_TRY_LINK([#include <mpi.h>],[int argc; char **argv; MPI_Init(&argc,&argv);],
AC_MSG_RESULT([yes]),AC_MSG_ERROR([Cannot compile and link MPI code with $CC]))
  AC_DEFINE(GMX_MPI,,[Make a parallel version of GENTOP using MPI])

# on the cray xt3 we have to tell autoconf that we 
# are actually cross-compiling even if the architecture
# of host and target are the same. so we enforce 
# cross-compilation if __QK_USER__ is defined
  AC_MSG_CHECKING([for catamount])
  AC_TRY_COMPILE([
#if defined __QK_USER__
#else
#error not catamount
#endif
],,[
  AC_MSG_RESULT([yes])
  cross_compiling=yes 
  AC_DEFINE(GMX_CRAY_XT3,[],[Enable special hacks for Cray XT3])
  AC_DEFINE(GMX_NO_SYSTEM,[],[Ignore calls to system(3)])
  AC_DEFINE(GMX_NO_NICE,[],[Ignore calls to nice(3)])
  AC_DEFINE(GMX_BROKEN_CALLOC,[],[Don't use calloc(3)])
  extra_LDFLAGS="$extra_LDFLAGS -lgmalloc -lacml"
],[
  AC_MSG_RESULT([no])
])
# end of "$enable_mpi" = "yes"
fi

if test "$enable_threads" = "yes"; then
   AC_MSG_CHECKING([for pthreads])
   AC_MSG_RESULT([yes])
   AC_DEFINE([GMX_THREAD_PTHREAD],,[Use pthreads for Gromacs multithreading])
#  ACX_PTHREAD([],AC_MSG_ERROR([Cannot compile with pthreads; use --disable-threads])
fi



AH_TEMPLATE([F77_OR_C_FUNC],
            [Set to F77_FUNC(name,NAME) if Fortran used, otherwise 'name' for C.])
AH_TEMPLATE([F77_OR_C_FUNC_],
            [Set to F77_FUNC_(name,NAME) if Fortran used, otherwise 'name' for C.])



if test "$enable_fortran" = "yes"; then
  AC_F77_LIBRARY_LDFLAGS
  AC_F77_WRAPPERS
  AC_DEFINE([F77_OR_C_FUNC(name,NAME)],[F77_FUNC(name,NAME)])
  AC_DEFINE([F77_OR_C_FUNC_(name,NAME)],[F77_FUNC_(name,NAME)])
else
  AC_DEFINE([F77_OR_C_FUNC(name,NAME)],[name])
  AC_DEFINE([F77_OR_C_FUNC_(name,NAME)],[name])
fi



AC_PROG_CPP

if test "$enable_cpu_optimization" = "yes"; then
  ACX_DETECT_GMXCPU
else
  gmxcpu=""
fi
ACX_COMPILER_MAXOPT




#############
# Check integer endian
#############
AC_C_BIGENDIAN([AC_DEFINE([GMX_INTEGER_BIG_ENDIAN],,[Integer byte order is big endian.])],[],[
AC_MSG_ERROR([Cannot determine endian in compiled output])])
# Little endian implied if GMX_INTEGER_BIG_ENDIAN not defined in source.

#############
# Check integer sizes, set to zero if e.g. long long is not supported.
#############
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(long int)
AC_CHECK_SIZEOF(long long int)


#############
# Make sure size_t can hold pointers.
# On some Nec machines, you reportedly need to specify a flag for size_t to be 64 bits wide.
#############
AC_MSG_CHECKING([that size_t can hold pointers])
AC_TRY_COMPILE([
#include <stdlib.h>
#include <stddef.h>
],[int chk[sizeof(size_t)-sizeof(int *)+1];],AC_MSG_RESULT([yes]),[
   AC_MSG_ERROR([size_t is smaller than int*. Check if your compiler needs a flag to make it 64 bit.])])



#############
# Check floating-point format and endian
#############
ACX_FLOAT_FORMAT


AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL
AC_SYS_LARGEFILE
AC_FUNC_FSEEKO




############################################################################
# Checks for libraries.
############################################################################
#
# Don't add math library for intel compilers
if $CC -V 2>&1 | grep 'Microsoft Corporation' > /dev/null 2>&1; then
  AC_MSG_NOTICE([Using built-in math library with intel compiler])
else
  AC_CHECK_LIB(m,sqrt,,AC_MSG_ERROR([No math library found]))
fi

#####
# Disable MASS support for now.
#####


######
# check for xtc headers
case "${host_os}" in
cygwin* | mingw*)   # use our own xdr stuff for cygwin/mingw
  GMXLIB_LIBOBJS="$GMXLIB_LIBOBJS gmx_system_xdr.lo"
  ;;
*)
  AC_CHECK_HEADERS([rpc/rpc.h])
  AC_CHECK_HEADERS([rpc/xdr.h],,,
[#if HAVE_RPC_RPC_H
# include <rpc/rpc.h>
#endif
])
  AC_CHECK_LIB(nsl,xdr_float)
  AC_TRY_LINK([
#if HAVE_RPC_RPC_H
#include<rpc/rpc.h> 
#else
#error NO rpc.h header
#endif
#if HAVE_RPC_XDR_H
#include<rpc/xdr.h>
#else
#error No xdr.h header
#endif
],[ XDR *xd; float f; xdr_float(xd,&f);],[have_xdr="yes"])

  if test "$have_xdr" != "yes"; then
    AC_DEFINE(GMX_INTERNAL_XDR,,[Use our own instead of system XDR libraries])	
      GMXLIB_LIBOBJS="${GMXLIB_LIBOBJS} gmx_system_xdr.lo"
    AC_MSG_WARN([Couldn't find XDR headers and/or libraries - using our own])
  fi
  ;;
esac

#####
# Checks for additional and/or optional functions or libraries.
#AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_TYPE_SIGNAL
AC_FUNC_VPRINTF
AM_WITH_DMALLOC
AC_CHECK_FUNCS(strcasecmp)
AC_CHECK_FUNCS(strdup)
# check for bool (must come late, since CFLAGS/includes might affect it)
AC_CHECK_TYPES([bool])

######
#### 
# And collect the f77 libs to the linker - on Solaris, f77 stuff must go first,
# and we need -L/usr/ucblib to find libucb... (stupid OS)
# On e.g. ibm the mass libraries must come before -lm... 
case "${host_cpu}-${host_os}" in
  *-solaris*) 
    LIBS="$FLIBS $LIBS" 
    LDFLAGS="$LDFLAGS -L/usr/ucblib"
    ;;
  *) LIBS="$LIBS $FLIBS" ;;
esac


# add the Accelerate framework if enabled above.
# LDFLAGS might have been set in the CPU optimization, so we cant assign it directly before that.
LDFLAGS="$LDFLAGS $extra_LDFLAGS"






########################################################################
# Checks for header files and types
########################################################################

AC_CHECK_HEADERS([limits.h],,AC_MSG_ERROR([Include headers not found]))
AC_CHECK_HEADERS(unistd.h)

#####
# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_TYPE_SIZE_T
AC_STRUCT_TM
AC_TYPE_UID_T
AC_C_INLINE
    




# Test stuff for ia64 assembly
#
# Nothing to test right now.





#############
# Check integer endian
#############
AC_C_BIGENDIAN([AC_DEFINE([GMX_INTEGER_BIG_ENDIAN],,[Integer byte order is big endian.])],[],[
AC_MSG_ERROR([Cannot determine endian in compiled output])])
# Little endian implied if GMX_INTEGER_BIG_ENDIAN not defined in source.



case "${host_cpu}" in
   i?86) 
         # Check if inline assembly works
         AC_MSG_CHECKING(if the compiler supports gcc inline assembly)
      AC_TRY_LINK([],[float f;int i; asm("fld %1\nfistpl %0\n" : "=m" (*&i) : "f" (f));],[AC_MSG_RESULT(yes)
AC_DEFINE(GMX_X86_GCC_INLINE_ASM,,[Enable x86 gcc inline assembly])],[AC_MSG_RESULT(no)])
         # Check for MSVC inline assembly
     AC_MSG_CHECKING(if the compiler supports MSVC inline assembly)
      AC_TRY_LINK([],[float f;int i; _asm { fld f } ; _asm { fistpl i };],[AC_MSG_RESULT(yes)
AC_DEFINE(GMX_X86_MSVC_INLINE_ASM,,[Enable x86 MSVC inline assembly])],[AC_MSG_RESULT(no)])
        ;;              
   *) ;;
esac






########################################################################
# Final output stage
########################################################################
AS="$CC"
AC_SUBST(ASFLAGS)
AC_SUBST(AS)
AC_SUBST(INCLUDES)          # should be automatic, but doesnt seem to be?
AC_SUBST(GMXLIB_LIBOBJS)
AC_SUBST(MDLIB_LIBOBJS)
AC_SUBST(GMXANA_LIBOBJS)

# To tell libraries apart, we use four versions of the name. For instance, libgmx:
# libgmx		single prec, no mpi.
# libgmx_d		double prec, no mpi.
# libgmx_mpi		single prec, mpi.
# libgmx_mpi_d		double prec, mpi.
# However, the non-suffixed names are linked to the _mpi and/or _d suffixed
# ones upon installation if that is the only version available.

if test "$enable_mpi" = "yes"; then
  LIBSUFFIX="_mpi"
fi
if test "$enable_float" = "no"; then
  LIBSUFFIX="${LIBSUFFIX}_d"
fi
AC_SUBST(LIBSUFFIX) 

# Unless the user has explicitly requested a prefix/suffix/transform, we
# use _d on double precision files. Don't add anything for mpi, but at the
# end we tell the user it is possible and smart to do in some cases!
if test "$program_transform_name" = "s,x,x,"; then
  name_transform_provided=no;
  if test "$enable_float" = "no"; then
    program_transform_name="s,\$\$,_d,"
  fi
else
  name_transform_provided=yes;
fi


# Use a variable for RPM - this way it can be 
# overridden with make RPM=rpm3 rpm
RPM=rpm
AC_SUBST(RPM)


if test "$enable_all_static" = "yes"; then
  LDFLAGS="$LDFLAGS -all-static"
fi

# we have asm source, so this is necessary for automake 1.6
CCAS=$CC
CCASFLAGS=$CFLAGS
 
AC_SUBST(CCAS)
AC_SUBST(CCASFLAGS)

AC_CONFIG_FILES([ Makefile ])
AC_CONFIG_FILES([ include/Makefile ])
AC_CONFIG_FILES([ src/Makefile ])
AC_CONFIG_FILES([ src/python/Makefile ])

AC_OUTPUT







#########################################################
# Echo some important info, to avoid stupid mistakes
#

if test "$enable_float" = "no" -a "$name_transform_provided" = "no"; then
  echo ""
  echo "* You are compiling a double precision version of Gromacs -"
  echo "  program names will be suffixed with _d to avoid overwriting single"
  echo "  precision files. You can override it with --program-suffix"
fi

if test "$name_transform_provided" = "no" -a "$enable_mpi" = "yes"; then
  echo ""
  echo "* Seems you are compiling with MPI support. You can install the MPI-"
  echo "  enabled programs with suffixed names to have both MPI and non-MPI"
  echo "  versions. This is useful e.g. on supercomputers where you usually"
  echo "  cannot run MPI-linked programs on the login node."
  echo "  Set a suffix with e.g. --program-suffix=_mpi (or _mpi_d for double)."
  echo "  You only need MPI for mdrun, so if you already have non-MPI stuff"
  echo "  installed you can issue make mdrun; make install-mdrun."
fi

echo ""

























