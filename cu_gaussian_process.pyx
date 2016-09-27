import numpy as np
import sys
import warnings

cdef extern from "gaussian_process.h":
    ctypedef struct cuGaussianProcess:
        pass
    cuGaussianProcess* cuGaussianProcessSolve(const float *sample_X,const float *sample_y, int n_sample, int n_dimention, float gamma, float regularization)
    void cuGaussianProcessPredict(cuGaussianProcess *ctx, const float *sample_X, int n_sample, float *sample_y, float *covar)
    void cuGaussianProcessFree(cuGaussianProcess *ctx)
    
cdef extern from "stdlib.h":
    void *malloc(size_t size)
    void free(void *)

cdef class GaussianProcess:
    cdef cuGaussianProcess *ctx
    cdef float gamma
    cdef float regularization
    cdef object X_mean
    cdef object X_std
    cdef object y_mean
    cdef object y_std
    
    def __init__(self, gamma, regularization):
        """
        Computes the nonzero componentwise L1 cross-distances between the vectors
        in X.
        Parameters
        ----------
        gamma: float
            The regression weight vector for RBF kernel
            exp(-gamma * |x1 - x2|^2)
            
        regularization: float
            Value added to the diagonal of the kernel matrix during fitting.
        """
        if regularization < 1e-6:
            warnings.warn("Too small regularization:%.10f"%(regularization))
        
        self.gamma = gamma
        self.regularization = regularization
        self.ctx = NULL
        
    def __dealloc__(self):
        if self.ctx != NULL:
            cuGaussianProcessFree(self.ctx);
        
    def _standard(self, sample_X, sample_y):
        self.X_mean = np.average(sample_X, 0)
        self.X_std = np.std(sample_X, 0)

        if 0. in self.X_std:
            raise Exception('contein 0 standard variable')
        
        sample_X = (sample_X -  self.X_mean) / self.X_std

        self.y_mean = np.average(sample_y, 0)
        self.y_std = np.std(sample_y, 0)
        sample_y = (sample_y -  self.y_mean) / self.y_std
        return sample_X, sample_y
        
        
    def fit(self, sample_X, sample_y):
        """Fit Gaussian process regression model
        Parameters
        ----------
        sample_X : array-like, shape = (n_samples, n_features)
            Training data
        sample_y : array-like, shape = (n_samples, )
            Target values
            
        """
        sample_X, sample_y = self._standard(sample_X, sample_y)
        
        cdef int n_sample = sample_X.shape[0]
        cdef int n_dimention = sample_X.shape[1]
        cdef int i, j
        cdef float *c_sample_X = <float*>malloc(sizeof(float) * n_sample * n_dimention)
        cdef float *c_sample_y = <float*>malloc(sizeof(float) * n_sample)
        
        for i in range(n_sample):
            for j in range(n_dimention):
                c_sample_X[i * n_dimention + j] = <float>sample_X[i, j]
            c_sample_y[i] = <float>sample_y[i]
            
        self.ctx = cuGaussianProcessSolve(c_sample_X, c_sample_y, n_sample, n_dimention, self.gamma, self.regularization)
        free(c_sample_X)
        free(c_sample_y)
        
    def predict(self, sample_X):
        """Predict using the Gaussian process regression model
        Parameters
        ----------
        sample_X : array-like, shape = (n_samples, n_features)
            Training data
            
        """
        sample_X = (sample_X -  self.X_mean) / self.X_std
        
        cdef int i, j
        cdef int n_sample = sample_X.shape[0]
        cdef int n_dimention = sample_X.shape[1]
        cdef float *c_sample_X = <float*>malloc(sizeof(float) * n_sample * n_dimention)
        cdef float *c_sample_y = <float*>malloc(sizeof(float) * n_sample)
        cdef float *c_sample_covar = <float*>malloc(sizeof(float) * n_sample * n_sample)
        
        for i in range(n_sample):
            for j in range(n_dimention):
                c_sample_X[i * n_dimention + j] = <float>sample_X[i,j]
        cuGaussianProcessPredict(self.ctx, c_sample_X, n_sample, c_sample_y, c_sample_covar)
        
        result = []
        result_covar = []
        for i in range(n_sample):
            tmp = []
            for j in range(n_sample):
                tmp.append(c_sample_covar[i * n_sample + j])
            result_covar.append(tmp)
            result.append(c_sample_y[i])
            
        free(c_sample_X)
        free(c_sample_y)
        return np.array(result) * self.y_std + self.y_mean,  np.array(result_covar)



