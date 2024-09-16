#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

#define CHECK(value) {\
  cudaError_t _m_cudaStat = (value);\
  if (_m_cudaStat != cudaSuccess) {\
    std::cout << "Error:" << cudaGetErrorString(_m_cudaStat)\
      << " at line " << __LINE__ << " in file " << __FILE__ << "\n";\
    exit(1);\
  }\
}

__host__ __device__ unsigned char adjustContrast(unsigned char _pixelValue, float alpha) {
  int pixelValue = _pixelValue;
  pixelValue = static_cast<int>(alpha * (pixelValue - 128) + 128);
  pixelValue = min(max(pixelValue, 0), 255);  // Clamping to [0, 255]
  return static_cast<unsigned char>(pixelValue);
}

__global__ void adjustContrast(unsigned char* image_data, float alpha, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    image_data[i] = adjustContrast(image_data[i], alpha);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path> <alpha>" << std::endl;
    return -1;
  }
  cv::Mat image = cv::imread(argv[1]);
  if (image.data == NULL) {
    std::cerr << "Can't open file" << std::endl;
    return -1;
  }
  int N = image.rows * image.cols * image.channels();
  float alpha = std::stof(argv[2]);
  unsigned char* image_host_data = image.data;
  unsigned char* image_result_data = new unsigned char[N];

  cudaEvent_t startCUDA, stopCUDA;
  clock_t startCPU;
  float elapsedTimeCUDA, elapsedTimeCPU;

  cudaEventCreate(&startCUDA);
  cudaEventCreate(&stopCUDA);

  unsigned char* image_device_data;

  CHECK( cudaMalloc(&image_device_data, N * sizeof(unsigned char)) );
  CHECK( cudaMemcpy(image_device_data, image_host_data, N * sizeof(unsigned char), cudaMemcpyHostToDevice) );

  startCPU = clock();
  for (int i = 0; i < N; i++) {
    image_host_data[i] = adjustContrast(image_host_data[i], alpha);
  }
  elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
  std::cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
  std::cout << "CPU memory throughput = " << N * sizeof(unsigned char)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";

  cudaEventRecord(startCUDA, 0);
  adjustContrast<<<(N+511)/512, 512>>>(image_device_data, alpha, N);
  cudaEventRecord(stopCUDA, 0);
  cudaEventSynchronize(stopCUDA);
  CHECK( cudaGetLastError() );

  cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

  std::cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
  std::cout << "CUDA memory throughput = " << N * sizeof(unsigned char) / elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";

  CHECK( cudaMemcpy(image_result_data, image_device_data, N * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

  for (int i = 0; i < N; i++) {
    if (image_host_data[i] != image_result_data[i]) {
      std::cout << "Error in element N " << i << ": image_host_data[i] = " << image_host_data[i]
        << " image_result_data[i] = " << image_result_data[i] << "\n";
      // exit(1);
    }
  }

  CHECK( cudaFree(image_device_data) );

  memcpy(image.data, image_host_data, N * sizeof(unsigned char));
  cv::imwrite("img_cpu.png", image);
  memcpy(image.data, image_result_data, N * sizeof(unsigned char));
  cv::imwrite("img_gpu.png", image);

  return 0;
}