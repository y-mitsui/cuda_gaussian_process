#include <stdio.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cusolverDn.h>
#include <assert.h>
#include "gaussian_process.h"
#include <execinfo.h>

#define d_cudaMalloc(ptr, size) debug_cudaMalloc(ptr, size, __FILE__, __LINE__)

void debug_cudaMalloc(void **ptr, size_t size, const char *file, int line_no){
    cudaError_t status;
    status = cudaMalloc(ptr, size);
    if (status != cudaSuccess){
        fprintf(stderr, "cudaMalloc error %s:%d\n", file, line_no);
        exit(1);
    }
}
void cuGaussianProcessFree(cuGaussianProcess *ctx){
    cudaFree(ctx->coefficient_device);
    cudaFree(ctx->sample_X_device);
    cudaFree(ctx->cholesky_lower);
    free(ctx);
}

__global__ void kernelRbf(float *result, float *x, int n_dimention, int n_sample, float gamma, float regularization, int max_iter){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < max_iter){
        int idx = tid + 1;
        int i = (int) ceil((0.5 + n_sample) - sqrt( (0.5 + n_sample) * (0.5 + n_sample) - 2.0 * idx));
        int est_n_l = -0.5 * i * i + (0.5 + n_sample) * i;
        int j = (int) (n_sample - i + 1) - (est_n_l - idx) + i - 1;
        i--;
        j--;

        float *x1 = &x[i * n_dimention];
        float *x2 = &x[j * n_dimention];
        
        float norm=0.;
        
        int k;
        for(k=0; k < n_dimention; k++){
            norm += (x1[k] - x2[k]) * (x1[k] - x2[k]);
        }
        result[i * n_sample + j] = result[j * n_sample + i] = expf(-gamma * norm);
        
        if (i == j) {
            result[i * n_sample + i] += regularization;
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void kernelRbf2(float *result, const float *sample_X, const float *sample_Y, int n_sample_X,int n_sample_Y, int n_dimention, float gamma, int max_iter){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < max_iter){
        int i = tid / n_sample_Y;
        int j = tid % n_sample_Y;
        
        const float *x1 = &sample_X[i * n_dimention];
        const float *x2 = &sample_Y[j * n_dimention];
        
        float norm=0.;
        
        int k;
        for(k=0; k < n_dimention; k++){
            norm += (x1[k] - x2[k]) * (x1[k] - x2[k]);
        }
        result[j * n_sample_X + i] = expf(-gamma * norm);
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void m_trans(float *result, float *sample, int n_row, int n_cal){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < (n_row * n_cal)){
        int i = tid / n_cal;
        int j = tid % n_cal;

        result[j * n_row + i] = sample[i * n_cal + j];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void m_sub(float *result, float *m_A, float *m_B, int n_row, int n_cal){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < (n_row * n_cal)){
        int i = tid / n_cal;
        int j = tid % n_cal;

        result[i * n_cal + j] = m_A[i * n_cal + j] - m_B[i * n_cal + j];
        tid += blockDim.x * gridDim.x;
    }
}

float *getGramMatrix2(const float *sample_X,const float *sample_y_device, int n_sample_X, int n_sample_Y, int n_dimention, float gamma){
    float *m_gram_device;
    float *sample_x_device;
    d_cudaMalloc((void**)&sample_x_device, sizeof(float) * n_sample_X * n_dimention);
    cudaMemcpy(sample_x_device, sample_X, sizeof(float) * n_sample_X * n_dimention, cudaMemcpyHostToDevice);
    
    /*float *sample_y_device;
    d_cudaMalloc((void**)&sample_y_device, sizeof(float) * n_sample_Y * n_dimention);
    cudaMemcpy(sample_y_device, sample_Y, sizeof(float) * n_sample_Y * n_dimention, cudaMemcpyHostToDevice);*/
    
    int n_iter = n_sample_X * n_sample_Y;
    d_cudaMalloc((void**)&m_gram_device, sizeof(float) * n_sample_X * n_sample_Y);

    int n_thread = 512;
    int n_block = (n_iter + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;

    kernelRbf2<<<n_block, n_thread>>>(m_gram_device, sample_x_device, sample_y_device, n_sample_X, n_sample_Y, n_dimention, gamma, n_iter);
    cudaFree(sample_x_device);
    //cudaFree(sample_y_device);
    
    return m_gram_device;
}
float *getGramMatrix(const float *sample_X, int n_sample, int n_dimention, float gamma, float regularization, float **r_sample_X_device){
    float *m_gram_device;
    float *sample_x_device;
    d_cudaMalloc((void**)&sample_x_device, sizeof(float) * n_sample * n_dimention);
    cudaMemcpy(sample_x_device, sample_X, sizeof(float) * n_sample * n_dimention, cudaMemcpyHostToDevice);
    int n_iter = (int)(((float)n_sample + 1.) * ((float)n_sample / 2.));
    d_cudaMalloc((void**)&m_gram_device, sizeof(float) * n_sample * n_sample);

    int n_thread = 512;
    int n_block = (n_iter + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;

    kernelRbf<<<n_block, n_thread>>>(m_gram_device, sample_x_device, n_dimention, n_sample, gamma, regularization, n_iter);
    if (r_sample_X_device) {
        *r_sample_X_device = sample_x_device;
    } else {
        cudaFree(sample_x_device);
    }
    
    return m_gram_device;
}

void cuGaussianProcessPredict(cuGaussianProcess *ctx, const float *sample_X, int n_sample, float *sample_y, float *covar){
    cublasStatus_t blas_status;
    cublasHandle_t blas_handle;
    blas_status = cublasCreate(&blas_handle);
    int n_thread, n_block;
    cusolverStatus_t status;
    cusolverDnHandle_t solver_handle;
    
    status = cusolverDnCreate(&solver_handle);
    
    float *m_gram_device = getGramMatrix2(sample_X, ctx->sample_X_device, n_sample, ctx->n_sample, ctx->n_dimention, ctx->gamma);
    
    float *est_y;
    d_cudaMalloc((void**)&est_y, sizeof(float) * n_sample);
    
    /*float *coefficient_device;
    d_cudaMalloc((void**)&coefficient_device, sizeof(float) * ctx->n_sample);
    cudaMemcpy(coefficient_device, ctx->coefficient, sizeof(float) * ctx->n_sample, cudaMemcpyHostToDevice);*/
    
    float alpha = 1.0f;
    float beta  = 0.0f;
   
    n_thread = 512;
    n_block = (n_sample * ctx->n_sample + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;
    blas_status = cublasSgemv(blas_handle, CUBLAS_OP_N, n_sample, ctx->n_sample, &alpha, m_gram_device, n_sample, ctx->coefficient_device, 1, &beta, est_y, 1);
    assert( blas_status == CUBLAS_STATUS_SUCCESS);
    
    cudaMemcpy(sample_y, est_y, sizeof(float) * n_sample, cudaMemcpyDeviceToHost);
    
    float *m_gram_device_t;
    d_cudaMalloc((void**)&m_gram_device_t, sizeof(float) * n_sample * ctx->n_sample);

    n_thread = 512;
    n_block = (n_sample * ctx->n_sample + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;
    m_trans<<<n_block, n_thread>>>(m_gram_device_t, m_gram_device, n_sample, ctx->n_sample);
    
    int *devInfo;
    // 計算結果に関する情報
    d_cudaMalloc((void**)&devInfo, sizeof(int) * 1);
    
    status = cusolverDnSpotrs(solver_handle, CUBLAS_FILL_MODE_LOWER, ctx->n_sample, ctx->n_sample, ctx->cholesky_lower,  ctx->n_sample, m_gram_device_t,  ctx->n_sample, devInfo);
    assert( status == CUSOLVER_STATUS_SUCCESS );
    float *m_solv_v = m_gram_device_t;
    
    float *matC_device;
    d_cudaMalloc((void**)&matC_device, sizeof(float) * n_sample * n_sample);

    blas_status = cublasSgemm(blas_handle, CUBLAS_OP_N,  CUBLAS_OP_N, n_sample, n_sample, ctx->n_sample, &alpha, m_gram_device, n_sample, m_solv_v, ctx->n_sample, &beta, matC_device, n_sample);
    assert( blas_status == CUBLAS_STATUS_SUCCESS);
    float *m_gram_device2 = getGramMatrix(sample_X, n_sample, ctx->n_dimention, ctx->gamma, 0.0, NULL);
    
    n_thread = 512;
    n_block = (n_sample * ctx->n_sample + n_thread - 1) / n_thread;
    if (n_block > 60000) n_block = 60000;
    m_sub<<<n_block, n_thread>>>(m_gram_device2, m_gram_device2, matC_device, n_sample, n_sample);
    cudaMemcpy(covar, m_gram_device2, sizeof(float) * n_sample * n_sample, cudaMemcpyDeviceToHost);
    
    cublasDestroy(blas_handle);
    cusolverDnDestroy(solver_handle);
    cudaFree(m_gram_device_t);
    //cudaFree(coefficient_device);
    cudaFree(devInfo);
    cudaFree(m_solv_v);
    cudaFree(matC_device);
    cudaFree(est_y);
    cudaFree(m_gram_device);
    cudaFree(m_gram_device2);
}


void gramMatrixSolve(float *m_gram_device, int n_sample, float *m_target_device, int target_col){
    cusolverStatus_t status;
    cusolverDnHandle_t handle;
    status = cusolverDnCreate(&handle);
    
    int worksize;
    status = cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n_sample, m_gram_device, n_sample, &worksize);
    assert( status == CUSOLVER_STATUS_SUCCESS );

    float* workspace ;
    d_cudaMalloc((void**)&workspace, sizeof(float)*worksize);
    
    int *devInfo;
    // 計算結果に関する情報
    d_cudaMalloc((void**)&devInfo, sizeof(int) * 1);

    status = cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n_sample, m_gram_device, n_sample, workspace, worksize, devInfo);
    assert( status == CUSOLVER_STATUS_SUCCESS );
    status = cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, n_sample, target_col, m_gram_device, n_sample, m_target_device, n_sample, devInfo);
    assert( status == CUSOLVER_STATUS_SUCCESS );
    
    //cudaMemcpy(coefficient, m_target_device, sizeof(float) * n_sample * target_col, cudaMemcpyDeviceToHost);

    cudaFree(workspace);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);
}

cuGaussianProcess* cuGaussianProcessSolve(const float *sample_X,const float *sample_y, int n_sample, int n_dimention, float gamma, float regularization){
    float *sample_x_device;
    float *m_gram_device = getGramMatrix(sample_X, n_sample, n_dimention, gamma, regularization, &sample_x_device);
    float *sample_y_device;
    
    d_cudaMalloc((void**)&sample_y_device, sizeof(float) * n_sample);
    cudaMemcpy(sample_y_device, sample_y, sizeof(float) * n_sample, cudaMemcpyHostToDevice);
    gramMatrixSolve(m_gram_device, n_sample, sample_y_device, 1);
    
    cuGaussianProcess *result;
    result = (cuGaussianProcess*)malloc(sizeof(cuGaussianProcess));
    result->coefficient_device = sample_y_device;
    result->sample_X_device = sample_x_device;
    result->n_dimention = n_dimention;
    result->n_sample = n_sample;
    result->cholesky_lower = m_gram_device;
    result->gamma = gamma;
    
    return result;
    
}
/*
int main(void){
    float sample_X[]={-4,-2, 1, 2};
    float sample_Y[]={-10,-5, 1, 5};
    float test_X[]={-4.0, 1.0};
    float est_y[2];
    float covar[2*2];
    int n_sample=3;
    int n_test = 2;
    int n_dimention = 1, i;
    cuGaussianProcess *ctx = cuGaussianProcessSolve(sample_X, sample_Y, n_sample, n_dimention, 0.5, 1e-4);
    cuGaussianProcessPredict(ctx, test_X, n_test, est_y, covar);
    cuGaussianProcessFree(ctx);
    for(i=0;i<n_test;i++){
        printf("final est_y[%d]:%f\n",i, est_y[i]);
        printf("final covar[%d]:%f\n",i, covar[i * n_test + i]);
    }
}
*/
