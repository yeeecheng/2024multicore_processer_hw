// Second CUDA program
// Ping-Che Chen


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256
#define BLOCK_SIZE 32

__global__ static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
{
	extern __shared__ float data[];
	const int tid = threadIdx.x;
	const int row = blockIdx.x;
	int i,j;
	
	for(i = tid; i < n; i += blockDim.x){
		data[i] = a[row * lda + i];
	}
	__syncthreads();

    for(j = tid; j < n; j += blockDim.x)
	{
		float t=0;
		float y=0;
		for(i = 0; i < n; i++)
		{
			float r;
			y -= data[i] * b[i * ldb + j];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		c[row * ldc + j] = t;
	}
}

clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	float *ac, *bc, *cc;
	clock_t start, end;
	size_t pitch_a, pitch_b, pitch_c;
	
	start = clock();
	cudaMallocPitch((void**) &ac, &pitch_a, sizeof(float) * n, n);
	cudaMallocPitch((void**) &bc, &pitch_b, sizeof(float) * n, n);
	cudaMallocPitch((void**) &cc, &pitch_c, sizeof(float) * n, n);

	cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda, sizeof(float) * n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb, sizeof(float) * n, n, cudaMemcpyHostToDevice);
	matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>
        (ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float),
        cc, pitch_c / sizeof(float), n);

	//int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
	//matMultCUDA<<<blocks * n, NUM_THREADS>>> (ac, n, bc, n, cc, n, n);
	//matMultCUDA<<<n, NUM_THREADS, sizeof(float) * n>>>(ac, n, bc, n, cc, n, n);

	//cudaMemcpy2D(c, sizeof(float)*ldc, cc, sizeof(float) * n,  sizeof(float) * n, n, cudaMemcpyDeviceToHost);
	//cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n, sizeof(float) *n, n, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c,
        sizeof(float) * n, n, cudaMemcpyDeviceToHost);

	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);
	end = clock();
	return end-start;
}


void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	int i, j, k;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			double t = 0;
			for(k = 0; k < n; k++) {
				t += a[i * lda + k] * b[k * ldb + j];
			}
			c[i * ldc + j] = t;
		}
	}
}


void matgen(float* a, int lda, int n)
{
	int i, j;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) 
		{
			a[i * lda + j] = (float) rand() / RAND_MAX + (float) rand() / (RAND_MAX * RAND_MAX);
		}
	}
}


void compare_mat(const float* a, int lda, const float* b, int ldb, int n)
{
	float max_err = 0;
	float average_err = 0;
	int i, j;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			if(b[i * ldb + j] != 0) {
				float err = fabs((a[i * lda + j] - b[i * ldb + j]) / b[i * ldb + j]);
				if(max_err < err) max_err = err;
				average_err += err;
			}
		}
	}

	printf("Max error: %g  Average error: %g\n", max_err, average_err / (n * n));
}


bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}


int main()
{
	float *a, *b, *c, *d;
	int n = 1000;

	if(!InitCUDA()) {
		return 0;
	}

	a = (float*) malloc(sizeof(float) * n * n);
	b = (float*) malloc(sizeof(float) * n * n);
	c = (float*) malloc(sizeof(float) * n * n);
	d = (float*) malloc(sizeof(float) * n * n);

	srand(0);

	matgen(a, n, n);
	matgen(b, n, n);

	clock_t time = matmultCUDA(a, n, b, n, c, n, n);

	matmult(a, n, b, n, d, n, n);
	compare_mat(c, n, d, n, n);

	double sec = (double) time / CLOCKS_PER_SEC;
	printf("Time used: %.4lf   (%.2lf GFLOPS)\n", sec, 2.0 * n * n * n / (sec * 1E9));

	free(a);
	free(b);
	free(c);
	free(d);

	return 0;
}
