TARGET = main

SOURCE = main.c

rebuild: clean all

all: $(TARGET)

$(TARGET) : $(SOURCE)
	gcc $(SOURCE) -o $(TARGET) -lpthread -lm

clean:
	rm -f $(TARGET)

