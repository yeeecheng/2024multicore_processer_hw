NVCC = nvcc

TARGET = main2

CUDA_SOURCE = main2.cu

rebuild: clean all

all: $(TARGET)

$(TARGET) : $(CUDA_SOURCE)
	$(NVCC) $(CUDA_SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

