NVCC = nvcc

TARGET = second_cuda4

CUDA_SOURCE = second_cuda4.cu

rebuild: clean all

all: $(TARGET)

$(TARGET) : $(CUDA_SOURCE)
	$(NVCC) $(CUDA_SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

