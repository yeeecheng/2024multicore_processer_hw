#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>

#define BLOCK_SIZE 32

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

__global__ static void CalPCC_CUDA(int s_r, int s_c, const int* target, int t_r, int t_c, float* mat_std_x, float* sum_s_sqr){
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    
    if(threadId >= t_r * t_c){
        return;
    }
    float sum_t = 0;
    for (int i = 0; i < s_r; i++){
        for (int j = 0; j < s_c; j++){
            if(threadId % t_c + j >= t_c || threadId + t_c * i + j >= t_r * t_c){
                return;
            }
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

    float res = sum_xy / (sqrt(sum_t_sqr) * sqrt(*sum_s_sqr));
    if(fabs(res - 1.0) < 0.000001){
        printf(" (%d, %d)\n", threadId / t_c, threadId % t_c);
    }
} 


__global__ static void CalSource_CUDA(const int* source, int s_r, int s_c, float* mat_std_s, float* sum_s_sqr){
    extern __shared__ char shared_data[];

    int* source_shr = (int*)shared_data;
    float* sum_sqr = (float*)&source_shr[s_r * s_c + 1];

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if(tid > s_r * s_c || blockId != 0){
        return;
    }
    if(tid == s_r * s_c){
        source_shr[tid] = 0;
	return;
    }
    source_shr[tid] = source[tid];
    __syncthreads();
    int d = (s_r * s_c + 1) / 2;
    for (int i = d; i > 1;  i = (i + 1) / 2){
	if(tid < i){
            source_shr[tid] += source_shr[tid + i];
        }
        else{
            source_shr[tid] = 0;
        }
	__syncthreads();
    }
    if(tid == 0){
    	source_shr[0] += source_shr[1];
    }
    __syncthreads();
    __shared__ float avg_s;
    avg_s = source_shr[0] /(float)(s_r * s_c);
    float num = source[tid] - avg_s;
    mat_std_s[tid] = num;
    sum_sqr[tid] = num * num;
    __syncthreads();

    for (int i = d; i > 1;  i = (i + 1) / 2){
        if(tid < i){
            sum_sqr[tid] += sum_sqr[tid + i];
        }
        else{
            sum_sqr[tid] = 0;
        }
	__syncthreads();
    }
    if(tid == 0){
        *sum_s_sqr = sum_sqr[0] + sum_sqr[1];
    }
    __syncthreads();
}

    
 

int padding(int n){
    return ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
}

clock_t PCC_CUDA(const int* source, int s_r, int s_c, const int* target, int t_r, int t_c, float* mat_std_s){
    
    int *source_c, *target_c;
    float *mat_std_s_c;

    clock_t start, end;
    cudaError_t R; 
    float* sum_s_sqr;
    start = clock();
    
    R = cudaMalloc((void**)&source_c, sizeof(int) * s_c * s_r);
    printf(" Malloc source_c: %s\n",cudaGetErrorString(R));

    R = cudaMalloc((void **)&target_c, sizeof(int) * t_c * t_r);
    printf(" Malloc target_c : %s\n",cudaGetErrorString(R));
    
    R = cudaMalloc((void **)&mat_std_s_c, sizeof(float) * s_c * s_r);
    printf(" Malloc mat_std_s_c : %s\n",cudaGetErrorString(R));
    
    R = cudaMalloc((void**)&sum_s_sqr, sizeof(float));
    printf(" Malloc sum_s_sqr : %s\n",cudaGetErrorString(R));

    R = cudaMemcpy(source_c, source, sizeof(int) * s_c * s_r, cudaMemcpyHostToDevice);
    printf(" Memcpy source_c: %s\n", cudaGetErrorString(R));

    R = cudaMemcpy(target_c, target, sizeof(int) * t_c * t_r, cudaMemcpyHostToDevice);
    printf(" Memcpy target_c : %s\n",cudaGetErrorString(R));
    
    R = cudaMemcpy(mat_std_s_c, mat_std_s, sizeof(float) * s_c * s_r, cudaMemcpyHostToDevice);
    printf(" Memcpy mat_std_s_c : %s\n", cudaGetErrorString(R));
    
    float h_sum = 0.0f;
    R = cudaMemcpy(sum_s_sqr,&h_sum, sizeof(float), cudaMemcpyHostToDevice);
    printf(" Memcpy sum_s_sqr : %s\n", cudaGetErrorString(R));
    
    int bx = (t_c + BLOCK_SIZE - 1) / BLOCK_SIZE, by = (t_r + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(bx, by);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    printf(" bx: %d, by: %d, BLOCK_SIZE: %d\n", bx, by, BLOCK_SIZE);
    int dynamic_size = (s_r * s_c + 1) * sizeof(int) + (s_r * s_c + 1) * sizeof(float);
    CalSource_CUDA<<<blocks, threads, (s_r * s_c + 1) * sizeof(int) + (s_r * s_c + 1) * sizeof(float)>>>(source_c, s_r, s_c, mat_std_s_c, sum_s_sqr);
   
    CalPCC_CUDA<<<blocks, threads>>>(s_r, s_c, target_c, t_r, t_c, mat_std_s_c, sum_s_sqr);
    
    end = clock();
    
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

clock_t PCC_CPU(int* source, int s_r, int s_c, int* target, int t_r, int t_c){
    
    clock_t start, end;
    start = clock();
    
    float sum_s_sqr = 0;
    float* mat_std_s = CalSource_CPU(source, s_r, s_c, sum_s_sqr);
    for(int i = 0; i < t_r - s_r + 1; i++){
        for(int j = 0; j < t_c - s_c + 1; j++){
	   float res = PCC_CPU(mat_std_s, target, s_r, s_c, t_r, t_c, j, i, sum_s_sqr);
	   if(fabs(res - 1.0) < 0.000001){
	       printf(" (%d, %d)\n", i, j);
	   }
	}
    }

    end = clock();
    
    free(mat_std_s);

    return end - start;
}

__global__ static void CalSSD_CUDA(const int* source, int s_r, int s_c, const int* target, int t_r, int t_c){
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    if(threadId >= t_r * t_c){
        return;
    }

    float sum_xy = 0;
    for(int i = 0; i < s_r; i++){
        for(int j = 0; j < s_c; j++){
            if(threadId % t_c + j >= t_c || threadId + t_c * i + j >= t_r * t_c){
                return;
            }
	        float num = source[i * s_c + j] - target[threadId + t_c * i + j];
	        sum_xy += num * num;
	    }
    }
    //if( threadId / t_c >= 45){
     //	printf("%d %d\n", threadId / t_c, threadId % t_c);
    //}
    if(sum_xy == 0){
        printf(" (%d, %d)\n", threadId / t_c, threadId % t_c);
    }
}

clock_t SSD_CUDA(int* source, int s_r, int s_c, int* target, int t_r, int t_c){

    int *source_c, *target_c;
    clock_t start, end;
    cudaError_t R;
    start = clock();

    R = cudaMalloc((void**)&source_c, sizeof(int) * s_c * s_r);
    printf(" Malloc source_c: %s\n", cudaGetErrorString(R));

    R = cudaMalloc((void**)&target_c, sizeof(int) * t_c * t_r);
    printf(" Malloc target_c: %s\n", cudaGetErrorString(R));
    
    R = cudaMemcpy(source_c, source, sizeof(int) * s_c * s_r, cudaMemcpyHostToDevice);
    printf(" Memcpy source_c: %s\n", cudaGetErrorString(R));

    R = cudaMemcpy(target_c, target, sizeof(int) * t_c * t_r, cudaMemcpyHostToDevice);
    printf(" Memcpy target_c: %s\n", cudaGetErrorString(R));

    int bx = (t_c + BLOCK_SIZE - 1) / BLOCK_SIZE, by = (t_r + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(bx, by);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    printf(" bx: %d, by: %d, BLOCK_SIZE: %d\n", bx, by, BLOCK_SIZE);
    CalSSD_CUDA<<<blocks, threads>>>(source_c, s_r, s_c, target_c, t_r, t_c);
    
    end = clock();
    
    cudaFree(target_c);
    cudaFree(source_c);

    return end - start;
}

clock_t SSD_CPU(int* source, int s_r, int s_c, int* target, int t_r, int t_c){
    
    clock_t start, end;

    start = clock();
    float sum_xy = 0;
    for(int i = 0; i < t_r - s_r + 1; i++){
        for(int j = 0; j < t_c - s_c + 1; j++){
	    sum_xy = 0;
	    for(int ii = 0; ii < s_r; ii++){
	        for(int jj = 0; jj < s_c; jj++){
		     float num = source[ii * s_c + jj] - target[i * t_c + j + ii * t_c + jj];
		     sum_xy += (num * num);
		}
	    }

	    if(sum_xy == 0){
	        printf(" (%d, %d)\n", i, j);
	    } 
	}
    }

    end = clock();
    return end - start;
   
}

int main(int argc, char *argv[]){
    
    printf(" Block size: %d\n", BLOCK_SIZE); 
    if(!InitCUDA()){
    //cudaFree(source_c);
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

    clock_t time;
    double sec;
    
    // CPU PCC
    printf("\n//// CPU PCC ////\n");
    time = PCC_CPU(s_mat, s_n, s_m, t_mat, t_n, t_m);
    sec = (double)time / CLOCKS_PER_SEC;
    printf(" CPU PCC Time used: %.41f\n\n", sec);
    

    // CUDA PCC
    printf("//// CUDA PCC ////\n");
    float* mat_std_s_cuda = (float*) malloc(sizeof(float) * t_n * t_m);
    time = PCC_CUDA(s_mat, s_n, s_m, t_mat, t_n , t_m, mat_std_s_cuda);
    
    sec = (double) time / CLOCKS_PER_SEC;
    printf(" CUDA PCC Time used: %.41f\n\n", sec);
    
    free(mat_std_s_cuda);
    
    // CPU SSD
    printf("//// CPU SSD ////\n");
    time = SSD_CPU(s_mat, s_n, s_m, t_mat, t_n, t_m);
    sec = (double) time / CLOCKS_PER_SEC;
    printf(" CPU SSD Time used: %.41f\n\n", sec);
    

    // CUDA SSD
    printf("//// CUDA SSD ////\n");
    time = SSD_CUDA(s_mat, s_n, s_m, t_mat, t_n, t_m);
    sec = (double) time / CLOCKS_PER_SEC;
    printf(" CUDA SSD Time used: %.4lf\n\n", sec);

    free(s_mat);
    free(t_mat);
    return 0;

}
