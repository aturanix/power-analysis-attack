.PHONY: compile clean

CFLAGS = -g -fanalyzer -fsanitize=address,undefined -Wall -Wextra -pedantic -std=c99
LDFLAGS =
LDLIBS =
SRCFILES = $(wildcard *.c)
OBJFILES = $(SRCFILES:.c=.o)
TARGET = program

compile: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJFILES) $(LDLIBS)

clean:
	rm $(TARGET) $(OBJFILES)
