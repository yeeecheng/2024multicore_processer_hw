#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 32
float** arr;

int* ChkMatSize(char* str, int& n, int& m){
    char* substr = strtok(str, "_");
    int cnt = 0;
    while(substr != NULL){
	if(cnt == 1){
	  n = atoi(substr);
	}
	else if(cnt == 2){
	  substr = strtok(substr, ".");
	  m = atoi(substr);
	  break;
	}
    	substr = strtok(NULL, "_");
    	cnt ++;
    }
    printf(" mat size: %d %d\n", n , m);
    int* mat = (int*)malloc(n * m * sizeof(int));
   
    return mat;
}

FILE* ReadFile(char* file){
    FILE* f;
    f = fopen(file, "r");
    if(f == NULL){
    	printf("cannot opne file\n");
    }
    return f;
}



bool InitCUDA(){
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

__global__ static void CalPCC_CUDA(int s_r, int s_c, const int* target, int t_r, int t_c, float* mat_std_x, float sum_s_sqr){
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    
    if(threadId >= t_r * t_c){
        return;
    }
    float sum_t = 0;
    for (int i = 0; i < s_r; i++){
        for (int j = 0; j < s_c; j++){
	    sum_t += target[threadId + j + t_c * i];
        }
    }
    float avg_t = sum_t / (s_c * s_r), sum_t_sqr = 0, sum_xy = 0;
    for (int i = 0; i < s_r; i++){
        for (int j = 0; j < s_c; j++){
            float num = target[threadId + j + t_c * i] - avg_t;
            sum_t_sqr += num * num;
            sum_xy += num * mat_std_x[i * s_c + j];
        }
    }

    float res = sum_xy / (sqrt(sum_t_sqr) * sqrt(sum_s_sqr));
    if(fabs(res - 1.0) < 0.000001){
        printf(" (%d, %d)\n", threadId / t_c, threadId % t_c);
    }
} 


void CalSource_CUDA(const int* source, int s_r, int s_c, float*& mat_std_s, float& sum_s_sqr){
    
    float sum_s = 0;
    for(int i = 0; i < s_r; i++){
        for(int j = 0; j < s_c; j++){
	     sum_s += source[i * s_c + j];
	}
    }
    float avg_s = sum_s / (s_r * s_c);
    for(int i = 0; i < s_r; i++){
    	for(int j = 0; j < s_c; j++){
	    int ii = i * s_c + j;
	    float num = source[ii] - avg_s;
	    sum_s_sqr += num * num;
	    mat_std_s[ii] = num;
	}
    }
}
int padding(int n){
    return ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
}

clock_t PCC_CUDA(const int* source, int s_r, int s_c, const int* target, int t_r, int t_c, float* mat_std_s){
    int *source_c, *target_c;
    float *mat_std_s_c;

    clock_t start, end;
    size_t pitch_s, pitch_t, pitch_std_s;
    cudaError_t R; 
    start = clock();
    
    
    R = cudaMalloc((void **)&target_c, sizeof(int) * t_c * t_r);
    printf(" Malloc target_c : %s\n",cudaGetErrorString(R));
    
    R = cudaMalloc((void **)&mat_std_s_c, sizeof(float) * s_c * s_r);
    printf(" Malloc mat_std_s_c : %s\n",cudaGetErrorString(R));
    
    R = cudaMemcpy(target_c, target, sizeof(int) * t_c * t_r, cudaMemcpyHostToDevice);
    printf(" Memcpy target_c : %s\n",cudaGetErrorString(R));
    

    float sum_s_sqr = 0;
    int bx = (t_c + BLOCK_SIZE - 1) / BLOCK_SIZE, by = (t_r + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(bx, by);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    printf(" bx: %d, by: %d, BLOCK_SIZE: %d\n", bx, by, BLOCK_SIZE);
    CalSource_CUDA(source, s_r, s_c, mat_std_s, sum_s_sqr);
    R = cudaMemcpy(mat_std_s_c, mat_std_s, sizeof(float) * s_c * s_r, cudaMemcpyHostToDevice);
    printf(" Memcpy mat_std_s_c : %s\n", cudaGetErrorString(R));

    CalPCC_CUDA<<<blocks, threads>>>(s_r, s_c, target_c, t_r, t_c, mat_std_s_c, sum_s_sqr);
    
    end = clock();
    //cudaFree(source_c);
    cudaFree(target_c);
    cudaFree(mat_std_s_c);
    return end - start;

}

float PCC_CPU(float* mat_std_s, int* mat_y, int s_r, int s_c, int t_r, int t_c, int offset_x, int offset_y, float& sum_x_sqr){
    float sum_y = 0;
    for(int i = 0; i < s_r; i++){
    	for(int j = 0; j < s_c; j++){
	    int ii = offset_y * t_c + offset_x + i * t_c + j;
	    sum_y += mat_y[ii];
	}
    }
    
    float avg_y = sum_y / (s_r * s_c);
    float sum_xy = 0, sum_y_sqr = 0;
    for(int i = 0; i < s_r; i++){
        for(int j = 0; j < s_c; j++){ 
	    int ii_x = i * s_c + j, ii_y = offset_y * t_c + offset_x + i * t_c + j;
	    sum_xy += mat_std_s[ii_x]  * (mat_y[ii_y] - avg_y);
	    float num = mat_y[ii_y] - avg_y;
	    sum_y_sqr += num * num;
	}
    }
    return sum_xy / (sqrt(sum_x_sqr) * sqrt(sum_y_sqr));

}                             

float* CalSource_CPU(int* source, int s_r, int s_c, float& sum_s_sqr){
    
    float sum_s = 0;
    for(int i = 0; i < s_r; i++){
        for(int j = 0; j < s_c; j++){
	    sum_s += source[i * s_c + j];
	}
    }

    float* mat = (float*)malloc(s_r * s_c * sizeof(float));
    float avg_s = sum_s / (s_r * s_c);
    for(int i = 0; i < s_r; i++){
        for(int j = 0; j < s_c; j++){
	    float num = source[i * s_c + j] - avg_s;
	    mat[i * s_c + j] = num;
	    sum_s_sqr += num * num;
	} 
    }
    return mat;
}

int main(int argc, char *argv[]){
    
    
    if(!InitCUDA()){
      return 0;
    }
    char* source_file = argv[1], * target_file = argv[2];

    // read file
    int s_n, s_m, t_n, t_m;
 
    FILE* s_f = ReadFile(source_file);
    FILE* t_f = ReadFile(target_file);
    int* s_mat = ChkMatSize(source_file, s_n, s_m);
    int* t_mat = ChkMatSize(target_file, t_n, t_m);
   
    int num, idx = 0;
    while(fscanf(s_f, "%d, ", &num) != EOF){
    	s_mat[idx] = num; 
    	idx++;
    }
    idx = 0;
    while(fscanf(t_f, "%d, ", &num) != EOF){
    	t_mat[idx] = num;
	idx++;
    }

    fclose(s_f);
    fclose(t_f);
    clock_t start, end;
    start = clock();
    float sum_s_sqr = 0;
    float* mat_std_s = CalSource_CPU(s_mat, s_n, s_m, sum_s_sqr);
    for(int i = 0; i < t_n - s_n + 1; i++){
    	for(int j = 0; j < t_m - s_m + 1; j++){
	    float res = PCC_CPU(mat_std_s, t_mat, s_n, s_m, t_n, t_m, j, i, sum_s_sqr);
 	    if(fabs(res - 1.0) < 0.000001){    
	        printf(" (%d, %d)\n", i, j); 
	    }
	}
    }
    end = clock();
    double sec = (double)(end - start) / CLOCKS_PER_SEC;
    printf(" CPU Time used: %.41f\n", sec);
    
    float* mat_std_s_cuda = (float*) malloc(sizeof(float) * t_n * t_m);
    clock_t time = PCC_CUDA(s_mat, s_n, s_m, t_mat, t_n , t_m, mat_std_s_cuda);
    
    sec = (double) time / CLOCKS_PER_SEC;
    printf(" CUDA Time used: %.41f\n", sec);
    
    free(mat_std_s_cuda);
    free(mat_std_s);
    free(s_mat);
    free(t_mat);
    return 0;



}
