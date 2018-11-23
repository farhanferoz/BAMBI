AM_CPPFLAGS = -O3 -I../src
LDADD = -L../src -lbambi-1.2

if HAVE_MPI
AM_CPPFLAGS += -DPARALLEL
LIBS += ${MPILIBS_CC} ${MPILIBS_CXX}
endif

EXTRA_DIST = eggbox_net.inp

if HAVE_LAPACK

bin_PROGRAMS = eggboxC

eggboxC_SOURCES = bambi.cc \
				  bambi.h \
				  eggbox.c \
				  eggbox.h

endif
