CC = gcc
CFLAGS = -Wall -Iinclude
SRC = $(wildcard src/*.c)
OBJ = $(SRC:.c=.o)

main: main.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

test: test.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

test.o: test.c
	$(CC) $(CFLAGS) -c test.c

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o src/*.o main test