NVCC = nvcc

TARGET = main1

CUDA_SOURCE = main1.cu

rebuild: clean all

all: $(TARGET)

$(TARGET) : $(CUDA_SOURCE)
	$(NVCC) $(CUDA_SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

