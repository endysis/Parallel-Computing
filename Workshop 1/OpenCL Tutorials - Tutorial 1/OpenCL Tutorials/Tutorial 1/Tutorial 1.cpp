#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#include <CL/cl.hpp>
#include "Utils.h"

using namespace std;

void print_help() {
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

	
		//display the selected device
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		//display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - memory allocation
		//host - input
		vector<double> A(100000); //C++11 allows this type of initialisation
		vector<double> B(100000);
		
		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(double);//size in bytes

		 

		//host - output
		vector<double> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//5.2 Setup and execute the kernel (i.e. device code)
		
		cl::Event prof_event;

		

	/*	cl::Kernel kernel_mult = cl::Kernel(program, "mult");

		kernel_mult.setArg(0, buffer_A);
		kernel_mult.setArg(1, buffer_B);
		kernel_mult.setArg(2, buffer_C); */
		

		cl::Kernel kernel_addf = cl::Kernel(program, "add");
		kernel_addf.setArg(0, buffer_C);
		kernel_addf.setArg(1, buffer_B);
		kernel_addf.setArg(2, buffer_C);

		//queue.enqueueNDRangeKernel(kernel_mult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange,NULL,&prof_event);		
		queue.enqueueNDRangeKernel(kernel_addf, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange,NULL,&prof_event);
		
		/*

		Doing more complex caluclation in the device is slower than using a serial method in this host code why?
		- Becuase CPU can access memory straight away.

		Changing to float take long because float is different calulartions double is 64 bit and float 32 and int is 16

		 
		  
		*/


		
		/*

		
		cl::Kernel kernel_addMult = cl::Kernel(program, "addMult");
		kernel_addMult.setArg(0, buffer_A);
		kernel_addMult.setArg(1, buffer_B);
		kernel_addMult.setArg(2, buffer_C);
		
		queue.enqueueNDRangeKernel(kernel_addMult, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange,NULL,&prof_event);
	    
		
		*/

		 


		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		/*cout << "A = " << A << endl;
		cout << "B = " << B << endl;
		cout << "C = " << C << endl; */

		cout << "Kernel execution time [ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;


		cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
}