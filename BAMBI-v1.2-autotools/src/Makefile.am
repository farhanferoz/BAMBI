library_includedir=$(includedir)

AM_FCFLAGS = -O3 -ffree-line-length-none
AM_CPPFLAGS = -O3

if HAVE_MPI
AM_CPPFLAGS += -DPARALLEL
AM_FCFLAGS += -DMPI
LIBS += ${MPILIBS_FC} ${MPILIBS_CXX}
endif

lib_LTLIBRARIES = libbambi-1.2.la

libbambi_1_2_la_SOURCES = utils.f90 \
                          utils1.f90 \
                          priors.f90 \
                          kmeans_clstr.f90 \
                          xmeans_clstr.f90 \
                          posterior.F90 \
                          nested.F90 \
                          NNopt.cc \
                          skynet.cc \
                          fnmatrix.cc \
                          myrand.cc \
                          RBMpretrain.cc \
                          linbcg.cc

library_include_HEADERS = NNopt.h \
						  NeuralNetwork.h \
						  TrainingData.h \
						  PredictedData.h \
						  RBMpretrain.h \
						  whiten.h \
						  myrand.h \
						  fnmatrix.h \
              linbcg.h

bin_PROGRAMS = NNerror

NNerror_SOURCES = NNerror.cc
NNerror_LDADD = -L./ -lbambi-1.2
