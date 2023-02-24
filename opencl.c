#include <malloc.h>
#include <stdio.h>
#include <CL/opencl.h>


#define MAX_SOURCE_SIZE (0x100000)

float f_a[50000], f_b[50000], f_c[50000], cl_c[50000];

void fill_data() {
  for(int i=0; i<50000; i++) {
    f_a[i] = (7.7f * (float)i) / 17.7f;
    f_b[i] = (8.7f * (float)i) / 18.7f;
  }
}

void func_cpu() {
  for(int i=0; i<50000; i++) {
    f_c[i] = f_a[i] * f_b[i];
  }  
}


int main() {
  fill_data();
  func_cpu();
  
  FILE *fp;
  char *source_str;
  
  fp = fopen("kernel.cl", "r");
  if(!fp) {
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  size_t source_size = fread(source_str, sizeof(char), MAX_SOURCE_SIZE, fp);
 
  cl_uint cl_func_result, ret_num_platforms, ret_num_devices;
  cl_platform_id *platform_ids;

  cl_func_result = clGetPlatformIDs(0, 0, &ret_num_platforms);
  platform_ids = (cl_platform_id*)malloc(sizeof(cl_platform_id) * ret_num_platforms);
  cl_func_result = clGetPlatformIDs(ret_num_platforms, platform_ids, NULL);
  
  cl_device_id device_id;
  cl_context context;
  cl_int errcode;
  cl_kernel kernel;
  cl_command_queue command_queue;
  cl_program program;
  cl_mem mem_a, mem_b, mem_c;
  
  for(int i=0; i<ret_num_platforms; i++) {
    cl_func_result = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 0, 0, &ret_num_devices);
    //or CPU
    
    //get devices
    cl_func_result = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    
    //can get device info from clGetDeviceInfo functions
    //https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/
    
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode);
    command_queue = clCreateCommandQueue(context, device_id, 0, NULL);
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &errcode);
    
    mem_a = clCreateBuffer(context, CL_MEM_READ_ONLY, 50000 * sizeof(float), NULL, &errcode);
    mem_b = clCreateBuffer(context, CL_MEM_READ_ONLY, 50000 * sizeof(float), NULL, &errcode);
    mem_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 50000 * sizeof(float), NULL, &errcode);
    cl_func_result = clEnqueueWriteBuffer(command_queue, mem_a, CL_TRUE, 0, 50000 * sizeof(float), f_a, 0, 0, 0);
    cl_func_result = clEnqueueWriteBuffer(command_queue, mem_b, CL_TRUE, 0, 50000 * sizeof(float), f_b, 0, 0, 0);
    
    kernel = clCreateKernel(program, "func_opencl", &errcode);
    cl_func_result = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_a);
    cl_func_result = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_b);
    cl_func_result = clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_c);
    //cl_func_result = clSetKernelArg(kernel, 3, sizeof(size_t), &width);
    
    size_t globalworksize = 50000;
    size_t localworksize = 1;
    
    cl_event event;
    cl_func_result = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalworksize, &localworksize, 0, 0, &event);
    //clFinish(command_queue); //blocking
    cl_func_result = clWaitForEvents(1, &event);
    cl_func_result = clEnqueueReadBuffer(command_queue, mem_c, CL_TRUE, 0, 50000 * sizeof(float), cl_c, 0, 0, 0);

    printf("%f, %f\n", f_c[1234], cl_c[1234]);
    
    cl_func_result = clReleaseKernel(kernel);    
    cl_func_result = clReleaseProgram(program);  
    cl_func_result = clReleaseMemObject(mem_a);
    cl_func_result = clReleaseMemObject(mem_b);
    cl_func_result = clReleaseMemObject(mem_c);
    cl_func_result = clReleaseCommandQueue(command_queue);
    cl_func_result = clReleaseContext(context);
  }
  free(platform_ids);
  free(source_str);
  fclose(fp);
  
  return 0;
}
