NVCC = nvcc 

TARGET = vec_add

CUDA_SOURCE = vec_add.cu

rebuild: clean all

all: $(TARGET)

$(TARGET) : $(CUDA_SOURCE)
	$(NVCC) $(CUDA_SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)
