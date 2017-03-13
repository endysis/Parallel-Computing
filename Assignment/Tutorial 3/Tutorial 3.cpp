#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include "WeatherData.h"


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

	for (int i = 1; i < argc; i++)	{
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

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		WeatherData wI;
		vector<WeatherData> weatherList;

		//typedef int mytype;
		//vector<mytype> weatherTemper; // Testing array


		typedef float mytype;
		//Part 4 - memory allocation
		//host - input
		//std::vector<mytype> A; //= { 3,4,8,3,7,9,3,2 };//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!

		std::vector<mytype> A = {2.5,5,4,7,4,5,2,2,4,6};


									 // Should we read the file in parallel ?
		ifstream inputFile("temp_lincolnshire_short.txt");
		string weatherStation1;
		int weatherYear;
		int weatherMonth;
		int weatherDay;
		int weatherTime;
		float weatherTemp;

		//int c = 0;

		while (inputFile >> weatherStation1 >> weatherYear >> weatherMonth >> weatherDay >> weatherTime >> weatherTemp) {
			//cout << "Variable " << weatherStation1 << endl;
			wI.setWeatherStation(weatherStation1);
			wI.setYearCollected(weatherYear);
			wI.setMonth(weatherMonth);
			wI.setDay(weatherDay);
			wI.setTime(weatherTime);
			wI.setTemp(weatherTemp);
			weatherList.push_back(wI);
			//A.push_back(weatherTemp); // Test Vector

			//string s = weatherList[c].getWeatherStation();
			//cout << "Vector Element " << s << endl;
			//c++
		}
		 
		  

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient

		size_t local_size = 10; // So i have to loop through the input vector to determine the locl size?
		 
		size_t padding_size = A.size() % local_size;
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		 
		if(padding_size) {
			//create an extra vector with neutral values
			std::vector<float> A_ext(local_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}
		       
		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		std::vector<mytype> B(1);  // Size of the B vector
		size_t output_size = B.size()*sizeof(mytype);//size in bytes

		 
		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer bufferAverage(context, CL_MEM_READ_ONLY, input_size);
		 
		//Part 5 - device operations
		 
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		 

		// Average kernal buffer
		queue.enqueueWriteBuffer(bufferAverage, CL_TRUE, 0, input_size, &A[0]);
		  
		 
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory 
		  
		     
		    
		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "minVec");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size
		  
		int s = A.size();
		 
		std::vector<int> sizeA = { s };

		printf("%d\n",A.size());

		  
		cl::Kernel kernel_AV = cl::Kernel(program, "average");
		kernel_AV.setArg(0, bufferAverage);
		kernel_AV.setArg(1, buffer_B);
		kernel_AV.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size 
		 

		//call all kernels in a sequence
		//queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size),NULL,&prof_event);


		//Average Kernel Call
		queue.enqueueNDRangeKernel(kernel_AV, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);


		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		B[0] = B[0]/A.size();

		//std::cout << "A = " << weatherTemper << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "Kernel execution time[ns]:"<<prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
	}
	
	
	
	
	
	
	
	
	
	
	
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	  
	return 0;
}
 