AC_PREREQ([2.63])

AC_INIT([BAMBI],[1.2])

AC_CONFIG_SRCDIR([src/nested.F90])
AC_CONFIG_AUX_DIR([gnuscripts])
AC_CONFIG_MACRO_DIR([gnuscripts])

AM_INIT_AUTOMAKE([foreign -Wall -Werror subdir-objects])

AC_ARG_ENABLE(
  [mpi],
  AC_HELP_STRING([--enable-mpi],[compile using MPI for supported codes [default=no]]),
  [ case "${enableval}" in
      yes) mpi=true;;
      no)  mpi=false;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-mpi) ;;
    esac
  ], [ mpi=false ]
)

AC_PROG_F77
AC_PROG_FC
AC_PROG_CC
AC_PROG_CXX

m4_ifdef([AM_PROG_AR],[AM_PROG_AR])

AM_PROG_CC_C_O

LT_INIT

# set silent build rules
AM_SILENT_RULES([yes])

# check for MPI compiler
MPI_ENABLED="DISABLED"
AC_LANG([Fortran])
if test "x$mpi" = "xtrue"; then
  AX_MPI([AC_SUBST([MPILIBS_FC],[${MPILIBS}])],[mpi=false])
fi
AC_LANG([C])
if test "x$mpi" = "xtrue"; then
  AX_MPI([AC_SUBST([MPILIBS_C],[${MPILIBS}])],[mpi=false])
fi
AC_LANG([C++])
if test "x$mpi" = "xtrue"; then
  AX_MPI([AC_SUBST([MPILIBS_CXX],[${MPILIBS}])],[mpi=false])
fi
if test "x$mpi" = "xtrue"; then
  MPI_ENABLED="ENABLED"
  FC=${MPIFC}
  CC=${MPICC}
  CXX=${MPICXX}
fi
AM_CONDITIONAL([HAVE_MPI],[test "${mpi}" = "true"])

# check for CBLAS library and header
AC_LANG([C])
AX_CBLAS([
  LIBS="$CBLAS_LIBS $LIBS"
],[
  AC_MSG_ERROR([could not find CBLAS library])
])
AC_CHECK_HEADERS([cblas.h],,[AC_MSG_ERROR([could not find the cblas.h header])])

# check  for LAPACK library
LAPACK_ENABLED="DISABLED"
lapack=false
AX_LAPACK([
  LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
  LAPACK_ENABLED="ENABLED"
  lapack=true
])
AM_CONDITIONAL([HAVE_LAPACK],[${lapack}])

AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([\
	Makefile \
	gnuscripts/Makefile \
	src/Makefile \
  example_eggbox_C/Makefile \
  example_eggbox_C++/Makefile \
  example_gauss_shell/Makefile \
  bambi-1.2.pc
])

AC_OUTPUT

echo "
================================================
BAMBI v1.2 has been successfully configured:

* LAPACK library support is ${LAPACK_ENABLED}
* MPI compilation is ${MPI_ENABLED}

and will be installed under the directory:

${prefix}

Now run 'make' to build BAMBI,
and run 'make install' to install BAMBI.
================================================
"
