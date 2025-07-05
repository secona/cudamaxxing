#include <stdio.h>

__global__
void vector_mul_kernel(float *M1, size_t n1, size_t m1,
					   float *M2, size_t n2, size_t m2,
					   float *M3, size_t n3, size_t m3) {
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if ((row < n3) && (col < m3)) {
		float p = 0;
		for (int i = 0; i < m1; i++) {
			p += M1[row * m1 + i] * M2[i * m2 + col];
		}
		M3[row * m3 + col] = p;
	}
}

void vector_mul(float *M1_h, size_t n1, size_t m1,
					   float *M2_h, size_t n2, size_t m2,
					   float *M3_h, size_t n3, size_t m3) {
	float *M1_d, *M2_d, *M3_d;

	cudaMalloc((void **) &M1_d, n1 * m1 * sizeof(float));
	cudaMalloc((void **) &M2_d, n2 * m2 * sizeof(float));
	cudaMalloc((void **) &M3_d, n3 * m3 * sizeof(float));

	cudaMemcpy(M1_d, M1_h, n1 * m1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(M2_d, M2_h, n2 * m2 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 gridDim(1);
	dim3 blockDim(n3, m3);
	vector_mul_kernel<<<gridDim, blockDim>>>(M1_d, n1, m1,
										     M2_d, n2, m2,
										     M3_d, n3, m3);

	cudaMemcpy(M3_h, M3_d, n3 * m3 * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(M1_d);
	cudaFree(M2_d);
	cudaFree(M3_d);
}

void print_matrix(float *A, size_t n, size_t m) {
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < m; k++) {
			printf("%.2f ", A[i * m + k]);
		}
		printf("\n");
	}
}

int main() {
	size_t n1 = 2, m1 = 4;
	float M1[n1 * m1] = { 1, 2, 3, 4, 5, 6, 7, 8 };

	size_t n2 = 4, m2 = 2;
	float M2[n2 * m2] = { 1, 2, 3, 4, 5, 6, 7, 8 };

	size_t n3 = n1, m3 = m2;
	float *M3 = (float *)malloc(n3 * m3 * sizeof(float));
	vector_mul(M1, n1, m1, M2, n2, m2, M3, n3, m3);

	print_matrix(M1, n1, m1); printf("\n*\n\n");
	print_matrix(M2, n2, m2); printf("\n=\n\n");
	print_matrix(M3, n3, m3);
}
