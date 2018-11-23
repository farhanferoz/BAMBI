CBLASDIR = /opt/local

#PARALLEL on Discover
#GSLDIR = /usr/local/other/SLES11.1/gsl/1.15/intel-12.1.0.233
#FC = mpif90
#CC = mpicc
#CXX = mpiicpc
#CFLAGS += -I. -I$(GSLDIR)/include/ -O3 -DPARALLEL
#CXXFLAGS += -I. -I$(GSLDIR)/include/ -O3 -DPARALLEL
#FFLAGS += -O0 -g -DMPI
#LAPACKLIB = -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -lmkl_core -lmkl_lapack95_lp64
#LAPACKLIB = -L/usr/local/other/SLES11/lapack/3.3.1/intel-12.1.0.233/lib -llapack -lblas -lcblas

#MAC OSX
CC = gcc-mp-4.7 -m64
CXX = g++-mp-4.7 -m64
FC = gfortran-mp-4.7 -m64
CFLAGS += -I. -O3 -I$(CBLASDIR)/include/
CXXGFLAGS += -I. -O3 -I$(CBLASDIR)/include/
FFLAGS += -w -O3
LAPACKLIB = -L$(CBLASDIR)/lib -llapack -latlas -lcblas -lf77blas

#PARALLEL on Darwin
#CC = mpicc
#CXX = mpicxx
#CFLAGS += -I. -I$(LIB_SN) -O3 -DPARALLEL
#CXXFLAGS +=  -I. -I$(LIB_SN) -O3 -DPARALLEL
#FC = mpif90 -DMPI
#FFLAGS +=  -w -O3
#LAPACKLIB = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

#PARALLEL on Universe
#FC = ifort -lmpi
#CC = icc -lmpi
#CXX = icpc -lmpi
#FFLAGS += -O3 -DMPI
#CFLAGS += -O3 -DPARALLEL
#LAPACKLIB = -llapack -lcblas

BAMBILIBDIR = ./
BAMBIINCDIR = src/

export FC CC CXX FFLAGS CFLAGS CXXFLAGS LAPACKLIB BAMBILIBDIR BAMBIINCDIR

default: libbambi-1.2.a

all: libbambi-1.2.a nnerror eggboxC++ eggboxC gauss_shell

gauss_shell:
	make -C example_gauss_shell

eggboxC++:
	make -C example_eggbox_C++

eggboxC:
	make -C example_eggbox_C

libbambi-1.2.a:
	make -C src libbambi-1.2.a

nnerror:
	make -C src nnerror

clean_all: clean_src clean_gauss_shell clean_eggboxC++ clean_eggboxC

clean_src:
	make -C src clean

clean_exec:
	rm gauss_shell eggboxC++ eggboxC

clean_gauss_shell:
	make -C example_gauss_shell clean

clean_eggboxC++:
	make -C example_eggbox_C++ clean

clean_eggboxC:
	make -C example_eggbox_C clean
