__kernel void func_opencl(__global const float *a, __global const float *b, __global float *c) {
  int id;
  id = get_global_id(0);
  
  c[id] = a[id] * b[id];  
}
