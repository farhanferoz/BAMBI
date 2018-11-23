LIBS =  -L../$(BAMBILIBDIR) -lbambi-1.2 $(LAPACKLIB) -lstdc++

CFLAGS += -I../$(BAMBIINCDIR)
 
OBJFILES = bambi.o eggbox.o

default: eggbox

all: eggbox 

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c

%.o: %.cc
	$(CXX) $(CFLAGS) -c $*.cc

eggbox: $(OBJFILES)
	$(FC) $(FFLAGS) -o ../eggboxC $(OBJFILES) $(LIBS)

clean:
	rm -f *.o ../eggboxC

tidy:
	rm -f *.o
