AR = ar r  
LINKLIB = ld -shared
 
BOBJECTS = utils.o utils1.o priors.o kmeans_clstr.o xmeans_clstr.o posterior.o nested.o \
			NNopt.o RBMpretrain.o fnmatrix.o myrand.o skynet.o linbcg.o

%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $^ 

%.o: %.F90
	$(FC) $(FFLAGS) -c -o $@ $^ 

%.o: %.cc
	$(CXX) $(CFLAGS) -c $*.cc

default: libbambi-1.2.a

all: libbambi-1.2.a nnerror

libbambi-1.2.so: $(BOBJECTS) 
	$(LINKLIB) -o $(LIBS) ../$@ $^ 
 
libbambi-1.2.a: $(BOBJECTS) 
	$(AR) ../$@ $^ 

nnerror: libbambi-1.2.a
	$(CXX) $(CXXFLAGS) -o ../NNerror NNerror.cc $(LAPACKLIB) -L../$(BAMBILIBDIR) -lbambi-1.2 -I../$(BAMBIINCDIR)

clean:
	rm -f ../libbambi-1.2.* ../NNerror *.o *.mod

tidy:
	rm -f *.o *.mod
