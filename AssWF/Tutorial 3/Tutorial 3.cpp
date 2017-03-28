#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <math.h>
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

		typedef float mytype;
		vector<mytype> weatherTemper; // Testing array


		typedef float mytype;
		//Part 4 - memory allocation
		//host - input
		//std::vector<mytype> A = { 3.5,7,8,3,7,9,3,2}; //allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!

		std::vector<mytype> A;


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
			A.push_back(weatherTemp); // Test Vector

									  //string s = weatherList[c].getWeatherStation();
									  //cout << "Vector Element " << s << endl;
									  //c++
		}
		 
		 
		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient

		size_t local_size = 20; // So i have to loop through the input vector to determine the locl size?
		 
		size_t padding_size = A.size() % local_size;
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected

		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<float> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;


		//host - output
		std::vector<mytype> B(input_elements);  // Size of the B vector
		std::vector<mytype> C(nr_groups);
		
		size_t output_size = B.size() * sizeof(mytype);//size in bytes
		size_t output_sizeC = C.size() * sizeof(mytype);//size in bytes
		 
		 
													    //device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_sizeC);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_sizeC); //  Final output vectio
		

		

		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_sizeC);//zero B buffer on device memory 

		// ___AVERAGE KERNEL___
		cl::Kernel kernel_AV = cl::Kernel(program, "reduce_add_4W");
		kernel_AV.setArg(0, buffer_A);
		kernel_AV.setArg(1, buffer_B);
		kernel_AV.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size 
		queue.enqueueNDRangeKernel(kernel_AV, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		//std::cout << "A = " << A << std::endl;
		//std::cout << "Average Number B = " << B << std::endl;
		//std::cout << "Average Number B = " << B[0] << std::endl;
		int mean = B[0]/input_elements;
		

		int c = 0;
		for (int i = 0; i < A.size(); i += local_size) {
			C[c] = B[i];
			c++;
		} // Am I okay  
		
		//cout << "C : " << C << endl;
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, output_sizeC, &C[0]);

		cl::Kernel kernel_2 = cl::Kernel(program, "reduce_add_4W");
		kernel_2.setArg(0, buffer_C);
		kernel_2.setArg(1, buffer_D);
		kernel_2.setArg(2, cl::Local(C.size() * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(nr_groups), cl::NullRange, NULL, &prof_event);



		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_sizeC, &C[0]);
		cout << "New C Total : " << C[0] << endl;




		//5.3 Copy the result from device to host
		


		//std::cout << "A = " << weatherTemper << std::endl;
		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}