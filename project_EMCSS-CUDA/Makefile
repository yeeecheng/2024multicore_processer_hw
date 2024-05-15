NVCC = nvcc

TARGET = main

CUDA_SOURCE = main.cu

rebuild: clean all

all: $(TARGET)

$(TARGET) : $(CUDA_SOURCE)
	$(NVCC) $(CUDA_SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

