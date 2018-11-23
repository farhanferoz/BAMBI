LIBS = -L../$(BAMBILIBDIR) -lbambi-1.2 $(LAPACKLIB) -lstdc++

CFLAGS += -I. -I../$(BAMBIINCDIR)
FFLAGS += -I. -I../$(BAMBIINCDIR) 
 
OBJFILES = bambi.o params.o like.o nestwrap.o main.o 

default: gauss_shell

all: gauss_shell

%.o: %.f90
	$(FC) $(FFLAGS) -c $*.f90

%.o: %.cc
	$(CXX) $(CFLAGS) -c $*.cc
 
gauss_shell: $(OBJFILES)
	$(FC) -o ../gauss_shell $(OBJFILES) $(FFLAGS) $(LIBS)

clean:
	rm -f *.o *.mod ../gauss_shell

tidy:
	rm -f *.o *.mod
