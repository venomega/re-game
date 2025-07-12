CC=gcc
CFLAGS=-fPIC -O2 -Wall -I.
LDFLAGS=-shared

all: libvjoy.so libuhid.so

libvjoy.so: vjoy.c vjoy.h
	$(CC) $(CFLAGS) vjoy.c -o libvjoy.so $(LDFLAGS)

libuhid.so: uhid.c uhid.h
	$(CC) $(CFLAGS) uhid.c -o libuhid.so $(LDFLAGS)

clean:
	rm -f libvjoy.so libuhid.so 