AM_FCFLAGS = -O3 -ffree-line-length-none -I../src
AM_CPPFLAGS = -O3 -I../src
LDADD = -L../src -lbambi-1.2

if HAVE_MPI
AM_CPPFLAGS += -DPARALLEL
AM_FCFLAGS += -DMPI
LIBS += ${MPILIBS_FC} ${MPILIBS_CXX}
endif

EXTRA_DIST = gauss_shell_net.inp

if HAVE_LAPACK

bin_PROGRAMS = gauss_shell

gauss_shell_SOURCES = bambi.cc \
					  params.f90 \
					  like.f90 \
					  nestwrap.f90 \
					  main.f90

endif
