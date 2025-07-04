#include <stdio.h>

__global__
void vec_add_kernel(float *A, float *B, float *C, int n) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n) C[i] = A[i] + B[i];
}

void vec_add(float *A_h, float *B_h, float *C_h, int n) {
	size_t size = n * sizeof(float);
	float *A_d, *B_d, *C_d;

	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	vec_add_kernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

int main() {
	int n = 3;
	float *A = (float *)malloc(sizeof(float) * n);
	float *B = (float *)malloc(sizeof(float) * n);
	float *C = (float *)malloc(sizeof(float) * n);

	A[0] = 1; A[1] = 2; A[2] = 3;
	B[0] = 1; B[1] = 1; B[2] = 1;

	vec_add(A, B, C, n);

	for (int i = 0; i < n; i++) {
		printf("%.2f\n", C[i]);
	}

	return 0;
}
