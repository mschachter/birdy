CC=g++
CFLAGS= -fpic
FILES = physical_oscillator.o
INCLUDE=
LIBS= -lgsl -lcblas -lm
LIBS_DIR=-L./ -L/usr/lib

#all: liboscillator.a

physical_oscillator.o: physical_oscillator.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIBS_DIR) -o physical_oscillator physical_oscillator.cpp $(LIBS)

clean:
	rm *.o
