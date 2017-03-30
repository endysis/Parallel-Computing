#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
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

	int profAVG = 0;
	int profAVGr = 0;






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
		//std::vector<mytype> A = {-3.5,-7,8,3,7,9,3,2,1.2f,-400,9.3f,5,94,-6.5}; 
		//std::vector<mytype> A {1,2,3,4,5,6,7,8};
		std::vector<mytype> A;
		ifstream inputFile("temp_lincolnshire.txt");
		string weatherStation1;
		int weatherYear;
		int weatherMonth;
		int weatherDay;
		int weatherTime;
		float weatherTemp;
		int count = 0;
		    
		while (inputFile >> weatherStation1 >> weatherYear >> weatherMonth >> weatherDay >> weatherTime >> weatherTemp) {
			wI.setWeatherStation(weatherStation1);
			wI.setYearCollected(weatherYear);
			wI.setMonth(weatherMonth);
			wI.setDay(weatherDay);
			wI.setTime(weatherTime);
			wI.setTemp(weatherTemp);
			//weatherList.push_back(wI);
			A.push_back(weatherTemp);
			count++; 
		} 
		 //count = 10;
		//printf("Count %d\n", count);

		size_t local_size = 64; 
		 // 892 for short
		 // 694 for long 
		 
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
		std::vector<mytype> C(input_elements);
		
		size_t output_size = B.size() * sizeof(mytype);//size in bytes
		size_t output_sizeC = C.size() * sizeof(mytype);//size in bytes
													    //device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_sizeC); //  Final output vectio
		
		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B,0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_size);//zero B buffer on device memory
		int workGroupCount = nr_groups;


		float firstQuart;
		float thirdQuart;
		vector<float> sortVec;


		// ___SORT KERNEL___
		cl::Kernel kernel_S = cl::Kernel(program, "ParallelBitonic_Local");
		kernel_S.setArg(0, buffer_A);
		kernel_S.setArg(1, buffer_B);
		kernel_S.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_S, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		//cout << B << endl;
		for (int i = (local_size / 2) - 1; i < B.size(); i += local_size) {
			sortVec.push_back((B[i] + B[i + 1]) / 2);
		}
		 
		//cout << "Padding Size: " << padding_size << endl;
		sort(sortVec.begin(), sortVec.end());
		printf("\n");
		float middle = (sortVec.size() + padding_size) / 2;
		//int middle = floor(middle);
		//cout << "Debug : " << middle << endl;
		
		firstQuart = (sortVec.size() + padding_size) / 4;
		thirdQuart = middle + firstQuart;


		cout << "First Quartile = " << sortVec.at(firstQuart) << endl;
		cout << "_______________" << endl;
		cout << "Middle Value = " << sortVec.at(middle) << endl;
		cout << "_______________" << endl;
		cout << "Third Quartile = " << sortVec.at(thirdQuart) << endl;
		cout << "_______________" << endl;

	




		//RESET//	
		int vector_elements = input_elements;
		int output_size1 = output_size;
		int vectorSize = 0;
		workGroupCount = nr_groups;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &C[0]);
		B = C;
		///////


		// MEAN KERNEL
		  cl::Event prof_eventMean;
		  cl::Event prof_eventMeanRe;
		cl::Kernel kernel_ADD = cl::Kernel(program, "addVec");
		while (workGroupCount >= 1) {
			kernel_ADD.setArg(0, buffer_A);
			kernel_ADD.setArg(1, buffer_B);
			kernel_ADD.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size 
			queue.enqueueNDRangeKernel(kernel_ADD, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &prof_eventMean);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size1, &B[0],NULL,&prof_eventMeanRe);
		vector<float>::const_iterator first = B.begin() + 0;
		vector<float>::const_iterator last = B.begin() + workGroupCount;
		vector<float> newInput (first,last);
		if(workGroupCount == 1){
			B = newInput;
			break;
		} 
		size_t padding_size = newInput.size() % local_size;
		if (padding_size) {
			std::vector<float> newInput_ext(local_size - padding_size, 0);
			newInput.insert(newInput.end(), newInput_ext.begin(),newInput_ext.end());
			}
		vector<float> newOutput(newInput.size(),0);
		vectorSize = newInput.size() * sizeof(float);
		queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,vectorSize,&newInput[0]);
		queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,vectorSize,&newOutput[0]);
		workGroupCount = newInput.size()/local_size;
		vector_elements = newInput.size();
		output_size1 = workGroupCount * sizeof(float);
		} 

		float mean = B[0]/count;
		printf("Mean Value : %f\n",mean);
		cout << "_______________" << endl;
		std::cout << "Kernel execution time [ns]:" << prof_eventMean.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventMean.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Read Time [ns]:" << prof_eventMeanRe.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventMeanRe.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "_______________" << endl;

		//RESET//		
		vector_elements = input_elements;
		output_size1 = output_size;
		vectorSize = 0;
		workGroupCount = nr_groups;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &C[0]);
		B = C; 
		///////


		//STANDARD DEVIAITION
		cl::Event prof_eventSD;
		cl::Event prof_eventSDR;
		cl::Kernel kernel_SD = cl::Kernel(program, "standDev");
		kernel_SD.setArg(0, buffer_A); // Set Arg Buffers
		kernel_SD.setArg(1, buffer_B);
		kernel_SD.setArg(2, mean);
		kernel_SD.setArg(3, cl::Local(local_size * sizeof(mytype)));//local memory size 
		queue.enqueueNDRangeKernel(kernel_SD, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_eventSD); // Enqueue Kernel
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &B[0]);
		cl::Kernel kernel_SD1 = cl::Kernel(program, "addVec");
		while (workGroupCount >= 1) { // Start while loop 
		kernel_SD1.setArg(0, buffer_A);
		kernel_SD1.setArg(1, buffer_B);
		kernel_SD1.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size
		queue.enqueueNDRangeKernel(kernel_SD1, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &prof_eventSDR);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size1, &B[0]);
		vector<float>::const_iterator first = B.begin() + 0;
		vector<float>::const_iterator last = B.begin() + workGroupCount;
		vector<float> newInput(first, last); // set new array to go into input A
		if (workGroupCount == 1) { // Break If here is only one workgroup left
		B = newInput;
		break;
		}
		size_t padding_size = newInput.size() % local_size;
		if (padding_size) {
		std::vector<float> newInput_ext(local_size - padding_size, 0);
		newInput.insert(newInput.end(), newInput_ext.begin(), newInput_ext.end());
		}
		vector<float> newOutput(newInput.size(), 0);
		vectorSize = newInput.size() * sizeof(float);
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vectorSize, &newInput[0]); // Place the new input into buffer_A
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vectorSize, &newOutput[0]); // Place new size empty vector
		workGroupCount = newInput.size() / local_size;
		vector_elements = newInput.size();
		output_size1 = workGroupCount * sizeof(float);
		}

		float RES = B[0] / count;
		float SD = sqrt(RES);
		printf("Standard Deviation : %f\n", SD);
		cout << "_______________" << endl;
		std::cout << "Kernel execution time [ns]:" << prof_eventSD.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventSD.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Read Time [ns]:" << prof_eventSDR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventSDR.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "_______________" << endl;

		///RESET///
		vector_elements = input_elements;
		output_size1 = output_size;
		vectorSize = 0;
		workGroupCount = nr_groups;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &C[0]);
		B = C; 
		//////////



		//MIN KERNEL //
		cl::Event prof_eventMIN;
		cl::Event prof_eventMINR;
		cl::Kernel kernel_MIN = cl::Kernel(program, "minVec");
		while (workGroupCount >= 1) {
			kernel_MIN.setArg(0, buffer_A);
			kernel_MIN.setArg(1, buffer_B);
			kernel_MIN.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size 
			queue.enqueueNDRangeKernel(kernel_MIN, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &prof_eventMIN);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size1, &B[0],NULL,&prof_eventMINR);
			vector<float>::const_iterator first = B.begin() + 0;
			vector<float>::const_iterator last = B.begin() + workGroupCount;
			vector<float> newInput(first, last);
			if (workGroupCount == 1) {
				B = newInput;
				break;
			}
			size_t padding_size = newInput.size() % local_size;
			if (padding_size) {
				std::vector<float> newInput_ext(local_size - padding_size, 0);
				newInput.insert(newInput.end(), newInput_ext.begin(), newInput_ext.end());
			}
			vector<float> newOutput(newInput.size(), 0);
			vectorSize = newInput.size() * sizeof(float);
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vectorSize, &newInput[0]);
			queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vectorSize, &newOutput[0]);
			workGroupCount = newInput.size() / local_size;
			vector_elements = newInput.size();
			output_size1 = workGroupCount * sizeof(float);
		}
		printf("Min Value : %f\n", B[0]);
		cout << "_______________" << endl;
		std::cout << "Kernel execution time [ns]:" << prof_eventMIN.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventMIN.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Read Time [ns]:" << prof_eventMINR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventMINR.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "_______________" << endl;

		 

		//RESET//		
		vector_elements = input_elements;
		output_size1 = output_size;
		vectorSize = 0;
		workGroupCount = nr_groups;
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, input_size, &C[0]);
		B = C;
		///////

		// MAX KERNEL
		cl::Event prof_eventMAX;
		cl::Event prof_eventMAXR;
		cl::Kernel kernel_MAX = cl::Kernel(program, "maxVec");
		while (workGroupCount >= 1) {
			kernel_MAX.setArg(0, buffer_A);
			kernel_MAX.setArg(1, buffer_B);
			kernel_MAX.setArg(2, cl::Local(local_size * sizeof(mytype)));//local memory size 
			queue.enqueueNDRangeKernel(kernel_MAX, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &prof_eventMAX);
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size1, &B[0],NULL,&prof_eventMAXR);
			vector<float>::const_iterator first = B.begin() + 0;
			vector<float>::const_iterator last = B.begin() + workGroupCount;
			vector<float> newInput(first, last);
			if (workGroupCount == 1) {
				B = newInput;
				break;
			}
			size_t padding_size = newInput.size() % local_size;
			if (padding_size) {
				std::vector<float> newInput_ext(local_size - padding_size, 0);
				newInput.insert(newInput.end(), newInput_ext.begin(), newInput_ext.end());
			}
			vector<float> newOutput(newInput.size(), 0);
			vectorSize = newInput.size() * sizeof(float);
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vectorSize, &newInput[0]);
			queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vectorSize, &newOutput[0]);
			workGroupCount = newInput.size() / local_size;
			vector_elements = newInput.size();
			output_size1 = workGroupCount * sizeof(float);
		}
		printf("Max Value : %f\n", B[0]);
		cout << "_______________" << endl;
		std::cout << "Kernel execution time [ns]:" << prof_eventMAX.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventMAX.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Read Time [ns]:" << prof_eventMAXR.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_eventMAXR.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "_______________" << endl;


	}

	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}


 


