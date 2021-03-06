#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
} 
 
int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	 
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);
		cl::Event prof_event;
		 
		cout << "Working" << endl; 
		  
		//build and debug the kernel code
		try {
			program.build(); 

			cout << "Finishes bulid" << endl;
		} 
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}   
		                        
		typedef int mytype;
		//Part 4 - memory allocation
		//host - input
		std::vector<mytype> A = {20,50,30,1,6,8,13,2,5,6,3,4,6,7,3,9};//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
		  
		  std::vector<mytype> C = {99,99};

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 8;
		
		size_t padding_size = A.size() % local_size;  
		     
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end()); 
		}  
		                     
		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;
		 
		cout << nr_groups << " da group number" << endl;    
		  
		//host - output
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size()*sizeof(mytype);//size in bytes
		cout << "Finishes output host size" << endl;


		size_t output_sizeC = C.size() * sizeof(mytype);//size in bytes

		
		  
		      
		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);

		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		 
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_sizeC); //  Final output vectio

		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_sizeC); //  Final output vectio
		 
		cout << "Makes buffers" << endl;		 
		    
		//Part 5 - device operations
		   
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_sizeC);//zero B buffer on device memory
		      
		cout << "Queues buffers" << endl;      
		     
		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4W");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));//
		//kernel_1.setArg(3, cl::Local(local_size * sizeof(mytype)));//
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL,&prof_event);

		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		std::cout << " Before : A = " << A << std::endl;
		std::cout << "Before : B = " << B << std::endl; 
		int groups = nr_groups;
		int loc_size = local_size;
		int c = 0;
		for(int i = 0; i < A.size(); i+=local_size) {
			C[c] = B[i];
			c++;
		} // Am I okay to do this
	
		cout << "C : " << C << endl;

		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, output_sizeC, &C[0]);
		
		

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_add_4W");
		kernel_2.setArg(0, buffer_C);
		kernel_2.setArg(1, buffer_D);
		kernel_2.setArg(2, cl::Local(C.size() * sizeof(mytype)));
																  
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(nr_groups), cl::NullRange, NULL, &prof_event);

		

		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_sizeC, &C[0]);
		cout << "New C : " << C << endl;

		

		/*for (int i = (local_size - 1); i < A.size(); i += local_size) {
			C[c] = B[i];
			c++;
		} // Am I okay to do this - this is for the scan

		
		// So what we are trying to do is understand how block sums work,
		// we have solved one erro, but block sums is meant to only output two length array
		// because of the two workgroup size and we get the second element 70 (from B) in pos [0] an 1 in pos[1]

		 
		/*queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(groups), cl::NullRange, NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0,nr_groups, &C[0]);		*/
		
		 
	


							   // So for the ten elements the array length would only be one?
	
	



		/*cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; //get device
		cout << "Working" << endl;
		cerr << "Prefered Size : " << kernel_1.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE> (device) << endl; //get info*/
		                                                                                                                                                                                                                      		/*std::cout << "Kernel execution time[ns]:"<<prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;*/
	}
	 
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	    
	return 0; 
}



 
// This will only work for one work group?






/* 
Question 1 Block scan, why does it not work because of two threads trying to access the same array?
2 Efficientcy on the computers next door
3 Passing whole int values to kernel by value or by reference
4 Structure of the assignment



*/










 