//a simple OpenCL kernel which copies all pixels from A to B
__kernel void identity(__global const uchar4* A, __global uchar4* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

//simple 2D identity kernel
__kernel void identity2D(__global const uchar4* A, __global uchar4* B) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0); //width in pixels
	int id = x + y*width;
	B[id] = A[id];
}

//2D averaging filter
__kernel void avg_filter2D(__global const uchar4* A, __global uchar4* B) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0); //width in pixels
	int id = x + y*width;

	uint4 result = (uint4)(0);//zero all 4 components

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++) 
		result += convert_uint4(A[i + j*width]); //convert pixel values to uint4 so the sum can be larger than 255

	result /= (uint4)(9); //normalise all components (the result is a sum of 9 values) 

	B[id] = convert_uchar4(result); //convert back to uchar4 
}

//2D 3x3 convolution kernel
__kernel void convolution2D(__global const uchar4* A, __global uchar4* B, __constant float* mask) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0); //width in pixels
	int id = x + y*width;

	float4 result = (float4)(0.0f,0.0f,0.0f,0.0f);//zero all 4 components

	for (int i = (x-1); i <= (x+1); i++)
	for (int j = (y-1); j <= (y+1); j++)
		result += convert_float4(A[i + j*width])*(float4)(mask[i-(x-1) + (j-(y-1))*3]);//convert pixel and mask values to float4

	B[id] = convert_uchar4(result); //convert back to uchar4
}


 __kernel void average(__global const int* A, __global int* B, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	
	scratch[lid] = A[id]; // All value witin the vector go from gloabl to local memory into the scratch

	// Is all this one work group??
	printf("Conversion\n");
	barrier(CLK_LOCAL_MEM_FENCE); // wait for each local thread to copy over. so yeah elements are run in parallel

	int temp;

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) { 
			printf("%d\n",scratch[lid]); // All values within the vector are operated upon at once, is this only one work group?
			if(scratch[lid] > scratch[lid + i]){ // So before it was if(scratch[lid] > A[lid + i]) which didnt work is searches in A where the 1200 value has not been swaped with 1 in the i = 1 (first) iteration
				temp = scratch[lid];
				scratch[lid] = scratch[lid + i];
				printf("Result Out : %d has been replaced with %d\n",temp,scratch[lid]);
			}
		}

		printf("Before barrier %d\n", i);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
		atomic_min(&B[0],scratch[lid]);   // Everything added to the first element in the global memory 
	}
}



__kernel void minVec(__global const int* A, __global int* B, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	
	scratch[lid] = A[id]; // All value witin the vector go from gloabl to local memory into the scratch

	// Is all this one work group??
	printf("Conversion\n");
	barrier(CLK_LOCAL_MEM_FENCE); // wait for each local thread to copy over. so yeah elements are run in parallel

	int temp;

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) { 
			printf("%d\n",scratch[lid]); // All values within the vector are operated upon at once, is this only one work group?
			if(scratch[lid] > scratch[lid + i]){ // So before it was if(scratch[lid] > A[lid + i]) which didnt work is searches in A where the 1200 value has not been swaped with 1 in the i = 1 (first) iteration
				temp = scratch[lid];
				scratch[lid] = scratch[lid + i];
				printf("Result Out : %d has been replaced with %d\n",temp,scratch[lid]);
			}
		}
		printf("Before barrier %d\n", i);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (!lid) {
		atomic_min(&B[0],scratch[lid]);   // Everything added to the first element in the global memory 
	}
}

