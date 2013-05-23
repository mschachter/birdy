CC=g++
CFLAGS= -fpic
FILES = physical_oscillator.o
INCLUDE=
LIBS= -lgsl -lcblas -lm
LIBS_DIR=

all: liboscillator.a

liboscillator.a: $(FILES)
	ar rus liboscillator.a $(FILES)

physical_oscillator.o: physical_oscillator.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS_DIR) -c -o physical_oscillator.o physical_oscillator.cpp $(LIBS)

clean:
	rm *.o *.a
