
typedef struct{
    float *coefficient_device;
    float *sample_X_device;
    int n_dimention;
    int n_sample;
    float gamma;
    float *cholesky_lower;
}cuGaussianProcess;

cuGaussianProcess* cuGaussianProcessSolve(const float *sample_X,const float *sample_y, int n_sample, int n_dimention, float gamma, float regularization);
void cuGaussianProcessPredict(cuGaussianProcess *ctx, const float *sample_X, int n_sample, float *sample_y, float *covar);
void cuGaussianProcessFree(cuGaussianProcess *ctx);

