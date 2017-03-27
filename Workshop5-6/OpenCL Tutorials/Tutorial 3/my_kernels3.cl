//fixed 4 step reduce
__kernel void reduce_add_1(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
	 
	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}
 

//flexible step reduce 
__kernel void reduce_add_2(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];
			
		barrier(CLK_GLOBAL_MEM_FENCE);
	} 
}



//reduce using local memory (so called privatisation)
__kernel void reduce_add_3(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//copy the cache to output array
	B[id] = scratch[lid];
}




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
			printf("%d\n",scratch[lid]);
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





__kernel void maxVec(__global const int* A, __global int* B, __local int* scratch){ 
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
			if(scratch[lid] < scratch[lid + i]){ // So before it was if(scratch[lid] > A[lid + i]) which didnt work is searches in A where the 1200 value has not been swaped with 1 in the i = 1 (first) iteration
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


//a very simple histogram implementation
__kernel void hist_simple(__global const int* A, __global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
// Step complxity logn 
//Work complexity nlogn


// So in using the Hills Inclusive, it pastes the sum in the last element of the array (not the first like in reduce).

// I need to paste the Result of each work group into a new array of the size of the number of work groups from the previous run.


__kernel void scan_add(__global const int* A, __global int* B, __local int* scratch_1, __local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	__local int *scratch_3; //used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];
	 
	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local memory
	printf("lid is : %d\n",lid);
	for (int i = 1; i < N; i *= 2){  
		if (lid >= i){
		printf("%d : is bigger or equal to :  %d\n",lid,i); 
		printf("Into for loop %d\n", scratch_1[lid]);
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
			printf("%d : is added to : %d\n",scratch_1[lid], scratch_1[lid - i]);
		}
		else {
			scratch_2[lid] = scratch_1[lid];
			}
		printf("Before Barrier : %d\n", i);
		barrier(CLK_LOCAL_MEM_FENCE);

		// I need to understand the swaping here...
		//buffer swap 
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;

		// Swaping the pointers
		// The reason is you save the tempory data from the last iteration in scratch 2, because scrtach two will be altered again,
		// and you need to store its buffer to read from, otherwise you would read from scrtach one ""Which is still on the first iteration of the buffer"
	
	}
	 
	//copy the cache to output array
	B[id] = scratch_1[lid];
}
 

  

//calculates the block sums
__kernel void block_sum(__global const int* A, __global int* B, int local_size) {
	int id = get_global_id(0);
	printf("B[id] = [(id+1)*local_size-1]\n");
	printf("%d = %d\n",id, (id + 1)*local_size - 1); // So all this is doing is getting the last element in the array we need the last element of each work group  
	 // Still dont get why they times the id by the local size, why not just give the local size?
	
	B[id] = A[(id+1)*local_size-1];
}




__kernel void block_sumReduce(__global const int* A, __global int* B, int inputLocal_size ,int outputLocal_size) {
	int id = get_global_id(0);
	int lid = get_local_id(0);

	int gid = get_group_id(0);
	// He said paste something like add this to the local id to get the output
	
	int G = get_global_size(0);
	int N = get_local_size(0);

	int stepNum = inputLocal_size/outputLocal_size;

//	printf("%d , %d\n",G,N);
// 

	//printf("%d\n",stepNum);

	/*for(int i = (inputLocal_size - 1); i >= 0; i - (stepDownNum - 1)){ 
	//	B[id] = A[i];
		printf("StepDownNum = %d",stepDownNum);
	}*/ 
	 
	

	//printf("%d the local size is\n", A[(lid+1)*8-1]);

	printf("%d sent size is %d\n",inputLocal_size);

	printf("local id %d\n",lid);

	printf("global id %d\n",id);

	printf("group id %d\n",gid);


	//for(int i = 1; i < N; i++){ 
	B[lid + gid] = A[(id + 1) * (8-1)];
	//}

	/*for(int i = 7; i <= 15; i+=8){ 
		//printf("%d\n",stepNum);
		printf("Id in A : %d\n",A[i]);
		printf("%d\n",i);
		B[0] = A[i]; 
	}*/
	// Cant seem to get another element in the second position of the output array
	// Probs because of a race condition
} 
 

// So It breaks the A array into two groups/sections (or the number of workgroups designed for the kernel) and
// get the last element of each section  - I wasnt under standing parrael



// So I think you cant use this function with elements which have more than one work gorup
// In this example it will only get the end element
// However we want to get two elements from two separte work groups
// So in this case element 8 and 16

// We need to divide the size of A by the new local size to get the number of iterations we want to reduce 
// the vector by

// if there was a workgroup size of 5 and we had 5 groups
// The elements we want in the block are the 5th 10th 15th 20th 25th
// so to get each one we need to -5 each time
// so we could use a reduce function which minus 5 each time
 







//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
__kernel void scan_add_atomic(__global int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}





//adjust the values stored in partial scans by adding block sums to corresponding blocks
__kernel void scan_add_adjust(__global int* A, __global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}








