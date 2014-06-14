#include<cuComplex.h>
#include<iostream>
#include<math.h>
#include<math_constants.h>
#include<stdio.h>
#include <time.h>

//Pode ir

using namespace std;

//Faz exponencial complexa. O parametro Ã o coeficiente complexo do expoente
__host__ __device__ cuDoubleComplex complexp(double exp) {
  double a = cos(exp);
  double bi = sin(exp);
  return make_cuDoubleComplex(a, bi);
}

__global__ void fft(cuDoubleComplex* A, long int m) {
  //Paraleliza a partir do segundo for
  unsigned int th = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int k = th / (m / 2);
  unsigned int j = th % (m / 2);
  //Assume que a propriedade e^x*e^y = e^(x+y) Ã© valida para 
  //exponencial de numeros complexos
  cuDoubleComplex w = complexp(((2 * CUDART_PI) / m)*j);
  cuDoubleComplex t = cuCmul(w, A[k + j + m / 2]);
  cuDoubleComplex u = A[k + j];
  A[k + j] = cuCadd(u, t);
  A[k + j + m / 2] = cuCsub(u, t);
}

//Faz a reversÃo de bits do Ãndice
__global__ void bit_reverse_copy(cuDoubleComplex* A, long int size, cuDoubleComplex* R) {
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  if ( n > size ) return;
  int s = (int)log2((double)size);
  int revn = 0;
  for ( int i = 0; i<s; i++ ) {
    revn += ((n >> i) & 1) << ((s - 1) - i);
  }
  cuDoubleComplex aux = A[n];
  R[revn] = aux;
}

int main() {
  int p;
  cin >> p;

  long int n = (long int)pow(2, p);
  size_t size = n*sizeof(cuDoubleComplex);
  cuDoubleComplex* A = (cuDoubleComplex*)malloc(size);


  for ( long int k = 0; k < n; k++ ) {
    /*A[k].x = sin(k * (2 * CUDART_PI_HI / n) );
    A[k].y = 0;*/
    if ( k < n / 2 ) {
      A[k].x = 1;
      A[k].y = 0;
    } else {
      A[k].x = 0;
      A[k].y = 0;
    }
  }

  cudaEvent_t stt, stp, stt2, stp2;
  cudaEventCreate(&stt);
  cudaEventCreate(&stp);
  cudaEventCreate(&stt2);
  cudaEventCreate(&stp2);

  cudaEventRecord(stt);


  cuDoubleComplex* A_d, *B_d;
  cudaMalloc(&A_d, size);
  cudaMalloc(&B_d, size);
  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);


  cudaEventRecord(stt2);


  unsigned int t = (n) > 512 ? 512 : (n);
  unsigned int bt = (unsigned int)((n) / t);
  //bt = (bt == 0) ? 1 : 0; //trata o caso de N < 512. Os outros casos a divisão da sempre exata devido a entrada ser potencia de 2
  dim3 g(t);
  dim3 b(bt);


  bit_reverse_copy << <g, b >> >(A_d, n, B_d);

  long int m = 2;
  for ( int i = 1; i <= log2((double)n); i++ ) {
    unsigned int x = ((n / 2) < 512) ? (n / 2) : 512;
    unsigned int bx = ((n / 2) / x);
    //bx = (bx == 0) ? 1 : bx;
    dim3 grid(x);
    dim3 blocks(bx);
    fft << <grid, blocks >> >(B_d, m);
    m *= 2;
  }
  cudaEventRecord(stp2);
  cudaMemcpy(A, B_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);

  cudaEventRecord(stp);

  cudaEventSynchronize(stp);
  cudaEventSynchronize(stp2);

  float milliseconds = 0;
  float milliseconds2 = 0;
  cudaEventElapsedTime(&milliseconds, stt, stp);
  cudaEventElapsedTime(&milliseconds2, stt2, stp2);

  cout << milliseconds << " seconds elapsed! (With copy)" << endl;
  cout << milliseconds2 << " seconds elapsed! (Without copy)" << endl;


  FILE * out;
  out = fopen("teste_cuda.txt", "a+");
  if ( out ) {
    fprintf(out, "%lf  -  %lf\n", milliseconds, milliseconds2);
    fclose(out);
  }
  free(A);

  cout << "Acabei o procedimento..." << endl;
  int d;
  cin >> d;
}



