#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct Param {
    
    int* s_mat;
    int* t_mat;
    int t_r;
    int t_c;
    int s_r;
    int s_c;
    int num_thread;
    int tid;

} Param;

FILE* ReadFile(char*);
int* ChkMatSize(char*, int* , int*);


void Thread_Start(Param*, char*,  int);
void* Thread_PCC(void*);
void* Thread_SSD(void*);


int main(int argc, char* argv[]){
    
    char* source_file = argv[1], *target_file = argv[2], *mode = argv[3];
    int num_thread = atoi(argv[4]);
    int s_r, s_c, t_r, t_c;

    FILE* s_f = ReadFile(source_file);
    FILE* t_f = ReadFile(target_file);
    int* s_mat = ChkMatSize(source_file, &s_r, &s_c);
    int* t_mat = ChkMatSize(target_file, &t_r, &t_c);

    int num, idx = 0;
    while(fscanf(s_f, "%d, " , &num) != EOF){
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
    
    printf("\n");

    Param param;
    param.s_mat = s_mat;
    param.t_mat = t_mat;
    param.t_r = t_r;
    param.t_c = t_c;
    param.s_r = s_r;
    param.s_c = s_c;  

    // calculation
    
    Thread_Start(&param, mode, num_thread);
    
    free(s_mat);
    free(t_mat);
    return 0;

}

FILE* ReadFile(char* file){
    FILE* f;
    f = fopen(file, "r");
    if(f == NULL){
        printf("cannot open file\n");
    }
    return f;
}

int* ChkMatSize(char* str, int* n, int* m){
    char* substr = strtok(str, "_");
    int cnt = 0;
    while(substr != NULL){
        if(cnt == 1){
	        *n = atoi(substr);
        }
        else if(cnt == 2){
            substr = strtok(substr, ".");
	        *m = atoi(substr);
            break;
        }
        substr = strtok(NULL, "_");
        cnt ++;
    }
    printf("mat size: %d %d\n", *n , *m);
    int* mat = (int*)malloc((*n) * (*m) * sizeof(int));

    return mat;
}


void Thread_Start(Param* param, char* mode,  int num_thread){
    
    clock_t start, end;
    start = clock();
    pthread_t thread[num_thread];
    param->num_thread = num_thread;
    int rc;
    for(int i = 0; i < num_thread; i++){
        Param* param_copy = malloc(sizeof(Param));
        param_copy->s_mat = param->s_mat;
        param_copy->t_mat = param->t_mat;
        param_copy->s_r = param->s_r;
        param_copy->s_c = param->s_c;
        param_copy->t_r = param->t_r;
        param_copy->t_c = param->t_c;
        param_copy->num_thread = num_thread;
        param_copy->tid = i;
        if(mode == "PCC"){
                rc = pthread_create(&thread[i], NULL, Thread_PCC, (void*) param_copy);
        } 
        else{
            rc = pthread_create(&thread[i], NULL, Thread_SSD, (void*) param_copy);	
        }

        if(rc){
            printf("Error, %d\n", rc);
            exit(-1);
        }
    }

    for(int i = 0; i < num_thread; i++){
        pthread_join(thread[i], NULL);
    }
    end = clock();
    printf("In %s, Use %d Thread Time used: %.41f\n\n",mode, num_thread, (double)(end - start) / CLOCKS_PER_SEC);

}

void* Thread_PCC(void* arg){
    Param tmp = *(struct  Param*)arg;
    int* s_mat = tmp.s_mat, * t_mat = tmp.t_mat;
    int s_r = tmp.s_r, s_c = tmp.s_c, t_r = tmp.t_r, t_c = tmp.t_c;
    int num_thread = tmp.num_thread, tid = tmp.tid;
    
    int x_sum = 0;
    for(int i = 0; i < s_r; i++){
        for(int j = 0; j < s_c; j++){
            x_sum += s_mat[i * s_c + j];
        }
    }
    for(int ti = tid; ti < t_r * t_c; ti += num_thread){
        int i = ti / t_c, j = ti % t_c;
        if(i >= t_r - s_r - 1 || j >= t_c - s_c -1){
            continue;
        }
        float xy_sum = 0, x_std = 0, y_std = 0;
        int y_sum = 0;
        for(int ii = 0; ii < s_r; ii++){
            for(int jj = 0; jj < s_c; jj++){
                y_sum += t_mat[i * t_c + j + ii * t_c + jj];
            }
        }
        for(int ii = 0; ii < s_r; ii++){
            for(int jj = 0; jj < s_c; jj++){
                int x = ii * s_c + jj;
                int y = i * t_c + j + ii * t_c + jj;
                float x_diff = (float)s_mat[x] - (float)x_sum, y_diff = (float)t_mat[y] - (float)y_sum;
		        xy_sum += x_diff * y_diff;
		        x_std += x_diff * x_diff;
		        y_std += y_diff * y_diff;
            }
        }

        float res = xy_sum / (sqrt(x_std) * sqrt(y_std));
        if((int)res == 1){
            printf("(%d, %d)\n", i, j);
        }
    }
    pthread_exit(NULL);
}


void* Thread_SSD(void* arg){
    clock_t s,e;
    Param tmp = *(struct Param*)arg;
    int* s_mat = tmp.s_mat, *t_mat = tmp.t_mat;
    int s_r = tmp.s_r, s_c = tmp.s_c, t_r = tmp.t_r, t_c = tmp.t_c;
    int num_thread = tmp.num_thread, tid = tmp.tid;
    for(int ti = tid; ti < t_r * t_c; ti += num_thread){
        int i = ti / t_c, j = ti % t_c;
        if( i >= t_r - s_r + 1 || j >= t_c - s_c +1){
            continue;
        }
        float xy_sum = 0;
        for(int ii = 0; ii < s_r; ii++){
            for(int jj = 0; jj < s_c; jj++){
                int x = ii * s_c + jj;
		        int y = i * t_c + j + ii * t_c + jj;
                float diff = (float)s_mat[x] - (float)t_mat[y];
                xy_sum += diff * diff;
            }
        }
        if((int)xy_sum == 0){
            printf("(%d, %d)\n", i, j);
        }
    }
    
    pthread_exit(NULL);
}

