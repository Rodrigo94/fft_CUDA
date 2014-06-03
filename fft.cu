#include<cuComplex.h>
#include<iostream>
#include<math.h>
#include<math_constants.h>

using namespace std;

//Faz exponencial complexa. O parametro Ã o coeficiente complexo do expoente
__host__ __device__ cuDoubleComplex complexp(double exp) {
  double a = cos(exp);
  double bi = sin(exp);
  return make_cuDoubleComplex(a, bi);
}

__global__ void fft(cuDoubleComplex* A, int m) {
  //Paraleliza a partir do segundo for
  int k = (blockIdx.y*blockDim.y + threadIdx.y)*m;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  //Assume que a propriedade e^x*e^y = e^(x+y) Ã© valida para 
  //exponencial de numeros complexos
  cuDoubleComplex w = complexp(((2 * CUDART_PI) / m)*j);
  cuDoubleComplex t = cuCmul(w, A[k + j + m / 2]);
  cuDoubleComplex u = A[k + j];
  A[k + j] = cuCadd(u, t);
  A[k + j + m / 2] = cuCsub(u, t);
}

//Faz a reversÃo de bits do Ãndice
__global__ void bit_reverse_copy(cuDoubleComplex* A, int size) {
  int n = blockIdx.x*blockDim.x + threadIdx.x;
  if ( n > size ) return;
  int s = (int)log2((double)size);
  int revn = 0;
  for ( int i = 0; i<s; i++ ) {
    revn += ((n >> i) & 1) << ((s - 1) - i);
  }
  cuDoubleComplex aux = A[n];
  A[n] = A[revn];
  A[revn] = aux;
}

int main() {
  int n = pow(2, 5);
  int size = n*sizeof(cuDoubleComplex);
  cuDoubleComplex* A = (cuDoubleComplex*)malloc(size);


  for ( int k = 0; k < 32; k++ ) {
    if ( k < 16 ) {
      A[k].x = 0;
      A[k].y = 0;
    } else {
      A[k].x = 1;
      A[k].y = 0;
    }
  }
  cuDoubleComplex* A_d;
  cudaMalloc(&A_d, size);
  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

  int t = n > 512 ? 512 : n;
  dim3 g(t);
  dim3 b(n / t);


  bit_reverse_copy << <g, b >> >(A_d, n);

  /*cuDoubleComplex* B = (cuDoubleComplex*)malloc(size);
  cudaMemcpy(B, A_d, size, cudaMemcpyDeviceToHost);

  for ( int k = 0; k < 32; k++ ) {
  cout <<
  B[k].x
  <<
  " "
  <<
  B[k].y
  <<
  endl;
  }*/

  int m = 2;
  for ( int i = 1; i <= log2((double)n); i++ ) {
    //Divide o trabalho proporcionalmente
    int nk = n / m;
    int nj = m / 2;
    int num = nj* nk;
    double prop = ((double)nk) / ((double)num);
    int threads = num > 512 ? 512 : num;
    int py = (int)(threads*prop);
    //Trata o caso de a proporÃÃo de trabalho de k ser 
    //tÃ£o pequena que d menos que uma thread por bloco
    int y = (py >= 1) ? py : 1;
    int x = threads / y;
    int by = nk / y;
    int bx = nj / x;
    cout << "Elas sao: " << x << " " << y << " " << bx << " " << by << endl;
    dim3 grid(x, y);
    dim3 blocks(bx, by);
    //cout << "Chamei fft:" << endl;
    //fft << < grid, blocks >> >(A_d, m);
    cout << i << " " << cudaGetErrorString(cudaGetLastError()) << endl;
    //cout << "Sai do fft:" << endl;
    m *= 2;
  }
  cudaMemcpy(A, A_d, size, cudaMemcpyDeviceToHost);

  for ( int k = 0; k < 32; k++ ) {
    cout <<
      A[k].x
      <<
      " "
      <<
      A[k].y
      <<
      endl;
  }

  cout << "Blablbebea" << endl;
  int d;
  cin >> d;
}



