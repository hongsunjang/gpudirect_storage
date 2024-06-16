#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include "cufile.h"
#include <cstring>
#include <cerrno>
//#include "cufile_sample_utils.h"

#define GB(x) ((x)*1024L*1024L*1024L)
#define MB(x) ((x)*1024L*1024L)
#define KB(x) ((x)*1024L)

// POSIX
template<class T,
	typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
	status = std::abs(status);
	return IS_CUFILE_ERR(status) ?
		std::string(CUFILE_ERRSTR(status)) : std::string(std::strerror(status));
}

// CUfileError_t
template<class T,
	typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
	std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
	if (IS_CUDA_ERR(status))
		errStr.append(".").append(GetCuErrorString(status.cu_err));
	return errStr;
}

template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

__global__ void hello(char *str) {
	printf("Hello World!\n");
	printf("buf: %s\n", str);
}

int main(int argc, char *argv[])
{
	int fd;
	int ret;

	char *gpumem_buf;

	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;
	CUfileError_t status;
	status = cuFileDriverOpen();
	if (status.err != CU_FILE_SUCCESS) {
			std::cerr << "cufile driver open error: "<<std::endl;
			return -1;
	}

	fd = open(argv[1], O_RDWR | O_DIRECT);
	if (fd == -1) {
        perror("open");
        return 1;
    }
	struct stat st;
    if (fstat(fd, &st) == -1) {
        perror("fstat");
        close(fd);
        return 1;
    }	
	

	int blksize = 512;
	//uint file_size_in_bytes = ((st.st_size -1)/blksize + 1) *blksize ;
	uint file_size_in_bytes = GB(1) ;
	
	// Print the file size in bytes
    std::cout << "File size: " << (double)(file_size_in_bytes) / 1024 / 1024 << " MB" << std::endl;
	
	std::cout << "Done\n";
	cudaMalloc(&gpumem_buf, file_size_in_bytes);
	
	off_t file_offset = 0;
	off_t mem_offset = 0;
	
	memset((void*)&cf_desc, 0, sizeof(CUfileDescr_t));
	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	status = cuFileHandleRegister(&cf_handle, &cf_desc);
	cuFileBufRegister((char*)gpumem_buf, file_size_in_bytes, 0);

	std::cout << "Read starts..." << std::endl;	
	std::chrono::high_resolution_clock::time_point read_start = std::chrono::high_resolution_clock::now();

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, file_size_in_bytes, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
        close(fd);
        return 1;
	}
	close(fd);
	
	std::chrono::high_resolution_clock::time_point read_end = std::chrono::high_resolution_clock::now();
	ulong read_time = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
	
	std::cout << "Read ends...\n" << std::endl;	

    double read_microsec_duration = (double) read_time;
	double read_millisec_duration = read_microsec_duration / 1e3;
	
	std::cout << "CPU: " << read_millisec_duration << " ms"
		<< std::setprecision(6) << std::fixed << "\n";
		
	std::cout << "Throughput: " << (double)(file_size_in_bytes) / GB(1)/ (read_millisec_duration*1e3) << " GB/s"
		<< std::setprecision(6) << std::fixed << "\n";


	//strrev<<<1,1>>>(gpumem_buf, gpu_len);
	std::vector<char, aligned_allocator<char>> system_buf(file_size_in_bytes);
	cudaMemcpy(&system_buf[0], gpumem_buf, file_size_in_bytes, cudaMemcpyDeviceToHost);
	std::cout << "Done\n";
	fd = open(argv[2], O_RDWR | O_DIRECT| O_CREAT, 0644);
	if (fd == -1) {
        perror("open");
        return 1;
    }

	

	std::cout << "Write starts..." << std::endl;	
	std::chrono::high_resolution_clock::time_point write_start = std::chrono::high_resolution_clock::now();

	ret = pwrite(fd, (void*)&system_buf[0], file_size_in_bytes, 0);
	if (ret == -1) {
		std::cout << "P2P: write() failed, err: " << ret << ", "<< strerror(errno) << ", line: " << __LINE__ << std::endl;
		return EXIT_FAILURE;
	}	

	/*
	cf_desc.handle.fd = fd;
	cuFileHandleRegister(&cf_handle, &cf_desc);

	ret = cuFileWrite(cf_handle, (char*)gpumem_buf, file_size_in_bytes, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileWrite failed : %d\n", ret); 
		close(fd);
        return 1;
	}
	*/	
	std::chrono::high_resolution_clock::time_point write_end = std::chrono::high_resolution_clock::now();
	ulong write_time = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count();
	
	std::cout << "Write ends...\n" << std::endl;	

    double write_microsec_duration = (double) write_time;
	double write_millisec_duration = write_microsec_duration / 1e3;
	
	std::cout << "CPU: " << write_millisec_duration << " ms"
		<< std::setprecision(6) << std::fixed << "\n";
		
	std::cout << "Throughput: " << (double)(file_size_in_bytes) / GB(1) / (write_millisec_duration*1000) << " GB/s"
		<< std::setprecision(6) << std::fixed << "\n";

	//printf("%s\n", system_buf);
	printf("See also %s\n", argv[2]);

	cuFileBufDeregister((char*)gpumem_buf);

	cudaFree(gpumem_buf);

	cuFileDriverClose();
}
