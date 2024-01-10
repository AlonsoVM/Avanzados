#include <stdio.h>

#define N 500

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int i = threadIdx.x;
    DC[i] = DA[i] + DB[i];
}

int main()
{ int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;
  int i; int size = N*sizeof(int);
  float ms, msa, msb, msc;
  cudaEvent_t startEvent, stopEvent, totalStart, totalEnd;
  checkCuda( cudaEventCreate(&startEvent));
  checkCuda( cudaEventCreate(&totalStart));
  checkCuda( cudaEventCreate(&stopEvent));
  checkCuda( cudaEventCreate(&totalEnd));
  
  checkCuda( cudaEventRecord(totalStart, 0));

  // reservamos espacio en la memoria global del device
  checkCuda ( cudaEventRecord(startEvent, 0));
  cudaMalloc((void**)&DA, size);
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("cudaMallod DA :  (ms) %f\n", ms);

  checkCuda ( cudaEventRecord(startEvent, 0));
  cudaMalloc((void**)&DB, size);
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("cudaMallod DB :  (ms) %f\n", ms);

  checkCuda ( cudaEventRecord(startEvent, 0));
  cudaMalloc((void**)&DC, size);
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("cudaMallod DC :  (ms) %f\n", ms);
  
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  checkCuda ( cudaEventRecord(startEvent, 0));
  cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&msa, startEvent, stopEvent));
  printf("cudaMemCPY HA:  (ms) %f\n", msa);

  checkCuda ( cudaEventRecord(startEvent, 0));
  cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&msb, startEvent, stopEvent));
  printf("cudaMemCPY HB:  (ms) %f\n", msb);
  
  // llamamos al kernel (1 bloque de N hilos)
  checkCuda ( cudaEventRecord(startEvent, 0));
  VecAdd <<<1, N>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&msc, startEvent, stopEvent));
  printf("Kernel execution time:  (ms) %f\n", msc);
  

  checkCuda ( cudaEventRecord(startEvent, 0));
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  checkCuda( cudaEventRecord(stopEvent, 0));
  checkCuda( cudaEventSynchronize(stopEvent));
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("cudaMemCPY HC:  (ms) %f\n", ms);
  
  // liberamos la memoria reservada en el device
  cudaFree(DA); cudaFree(DB); cudaFree(DC);  
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}

  checkCuda( cudaEventRecord( totalEnd, 0));
  checkCuda( cudaEventSynchronize(totalEnd));
  checkCuda( cudaEventElapsedTime(&ms, totalStart, totalEnd));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);

  printf("Bandwith in the transfer DA <-- HA %f GBs\t\n", size/(msa/1000)/1000000000);
  printf("Bandwith in the transfer DB <-- HB %f GBs\t\n", size/(msb/1000)/1000000000);
  printf("Bandwith in the transfer DC <-- HC %f GBs\t\n", size/(msc/1000)/1000000000);

  printf("Finished\t\n");
  return 0;
} 
