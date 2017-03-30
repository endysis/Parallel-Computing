//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];
	
	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
			//printf("%d\n",scratch[lid]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//printf("break\n");

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);   // Everything added to the first element in the global memory 
	}
}

__kernel void addVec(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		//printf("Array Element %f\n", scratch[lid]);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//printf("break\n");
	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array

	if(!lid){ 
	B[gid+lid] = scratch[lid];
	}
}


__kernel void minVec(__global const float* A, __global float* B, __local float* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);
	
	scratch[lid] = A[id]; // All value witin the vector go from gloabl to local memory into the scratch

	// Is all this one work group??
	//printf("Conversion\n");
	barrier(CLK_LOCAL_MEM_FENCE); // wait for each local thread to copy over. so yeah elements are run in parallel

	float temp;
	 
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) { 
			//printf("%d\n",scratch[lid]); // All values within the vector are operated upon at once, is this only one work group?
			if(scratch[lid] > scratch[lid + i]){ // So before it was if(scratch[lid] > A[lid + i]) which didnt work is searches in A where the 1200 value has not been swaped with 1 in the i = 1 (first) iteration
				temp = scratch[lid];
				scratch[lid] = scratch[lid + i];
				//printf("Result Out : %d has been replaced with %d\n",temp,scratch[lid]);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(!lid){ 
	B[gid+lid] = scratch[lid];
	}
}








__kernel void maxVec(__global const int* A, __global int* B, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int gid = get_group_id(0);
	
	scratch[lid] = A[id]; // All value witin the vector go from gloabl to local memory into the scratch

	// Is all this one work group??
	//printf("Conversion\n");
	barrier(CLK_LOCAL_MEM_FENCE); // wait for each local thread to copy over. so yeah elements are run in parallel

	float temp;

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) { 
			//printf("%d\n",scratch[lid]); // All values within the vector are operated upon at once, is this only one work group?
			if(scratch[lid] < scratch[lid + i]){ // So before it was if(scratch[lid] > A[lid + i]) which didnt work is searches in A where the 1200 value has not been swaped with 1 in the i = 1 (first) iteration
				temp = scratch[lid];
				scratch[lid] = scratch[lid + i];
				//printf("Result Out : %d has been replaced with %d\n",temp,scratch[lid]);
			}
		}
	}
	if(!lid){ 
	B[gid+lid] = scratch[lid];
	}
}
 

  
															// Am i fine to just pass integer values here ? 
__kernel void standDev(__global const float* A, __global float* B, float mean, local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	
	//scratch[lid] = A[id];
	//barrier(CLK_LOCAL_MEM_FENCE);
	//printf("ID = %f - mean %f , Local id = %d \n", scratch[lid], mean,lid);
	//scratch[lid] -= mean;
	//printf(" ");
	//barrier(CLK_LOCAL_MEM_FENCE);
	//printf("ID %f * ID %f,  Local id = %d \n",scratch[lid],scratch[lid],lid);
	//scratch[lid] *= scratch[lid];

	scratch[lid] = (A[id] - mean) * (A[id] - mean);
	//printf("mean = %f\n",mean);
	//barrier(CLK_LOCAL_MEM_FENCE);

	B[id] = scratch[lid];
}

 


// Try to do without atomic function, by pasting the sum of each work group into a global vector and applying reduce again to the vector.
// To get float number working I have to get the above to work


// When 10 elements there is only one workgorup.


 

__kernel void ParallelBitonic_Local(__global const float * A, __global float * B, __local float * aux)
{
	int i = get_local_id(0); // index in workgroup
	int wg = get_local_size(0); // workgroup size = block size, power of 2

								// Move IN, OUT to block start
	int offset = get_group_id(0) * wg;
	A += offset; B += offset;

	// Load block in AUX[WG]
	aux[i] = A[i];
	barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

								  // Loop on sorted sequence length
	for (int length = 1; length < wg; length <<= 1)
	{
		bool direction = ((i & (length << 1)) != 0); // direction of sort: 0=asc, 1=desc
													 // Loop on comparison distance (between keys)
		for (int inc = length; inc>0; inc >>= 1)
		{
			int j = i ^ inc; // sibling to compare
			float iData = aux[i];
			float iKey = iData;
			float jData = aux[j];
			float jKey = jData;
			bool smaller = (jKey < iKey) || (jKey == iKey && j < i);
			bool swap = smaller ^ (j < i) ^ direction;
			barrier(CLK_LOCAL_MEM_FENCE);
			aux[i] = (swap) ? jData : iData;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	// Write output
	B[i] = aux[i];
}


