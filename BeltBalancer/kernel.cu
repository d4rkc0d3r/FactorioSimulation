
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "BeltEntity.cuh"
#include "BlueprintStringReader.cuh"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <time.h>

using namespace std;

#ifdef __unix__ 

void printAndMoveCursorBack(string str)
{
	// not implemented for linux
}

#elif defined(_WIN32) || defined(WIN32)

#include <windows.h>

void printAndMoveCursorBack(string str)
{
	cout << str;
	COORD pos;
	pos.X = 0;
	CONSOLE_SCREEN_BUFFER_INFO nfo;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &nfo);
	pos.Y = nfo.dwCursorPosition.Y;
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);
}

#endif

bool useCPU = true;
bool testInputBalance = true;
bool testOuputBalance = true;
bool testFullLoadTroughput = true;
bool testAllTwoBeltTroughputCombinations = false;
int threads = 256;

bool updateEntities(BeltEntity* entities, size_t size, unsigned int iterations)
{
	return useCPU ? updateOnCPU(entities, size, iterations) : updateOnGPU(entities, size, iterations, threads);
}

void displayEntities(BeltEntity* entities, size_t size)
{
	for (unsigned int i = 0; i < size; i++)
	{
		char t = '?';
		switch (entities[i].type)
		{
		case TYPE_BELT:
			t = 'b';
			break;
		case TYPE_LEFT_SPLITTER:
		case TYPE_RIGHT_SPLITTER:
			t = 's';
			break;
		case TYPE_SPAWN:
			t = '+';
			break;
		case TYPE_VOID:
			t = '-';
			break;
		case TYPE_UNDERGROUND_ENTRANCE:
		case TYPE_UNDERGROUND_EXIT:
			t = 'u';
			break;
		case TYPE_BLOCK:
			t = 'X';
			break;
		}

#ifndef _DEBUG
		if (t == '+' || t == '-')
#endif
			cout << "(" << t << ", " << entities[i].buffer << ", " << entities[i].lastTroughput << ", " << i - 1 << ", " << entities[i].next << ")" << endl;
	}
	cout << endl;
}

string loadBlueprintFile(string fileName)
{
	string output;
	ifstream t(fileName);
	stringstream ss;
	ss << t.rdbuf();
	t.close();
	output = ss.str();
	if (output == "")
	{
		ss.clear();
		t.open(getenv("APPDATA") + string("\\factorio\\script-output\\blueprint-string\\") + fileName);
		ss << t.rdbuf();
		t.close();
		output = ss.str();
	}
	return output;
}

struct IdTroughputHelper
{
	int id;
	float troughput;
};

void testBalance(BeltEntity* entities, size_t size, int iterations)
{
	vector<IdTroughputHelper> spawnBelts;
	vector<IdTroughputHelper> voidBelts;

	for (unsigned int i = 0; i < size; i++)
	{
		IdTroughputHelper t;
		t.id = i;
		t.troughput = entities[i].maxTroughput;
		if (entities[i].type == TYPE_SPAWN)
		{
			spawnBelts.push_back(t);
		}
		else if (entities[i].type == TYPE_VOID)
		{
			voidBelts.push_back(t);
		}
	}

	cout << "Testing a " << spawnBelts.size() << " to " << voidBelts.size() << " balancer" << endl;

	BeltEntity* workingCopy = new BeltEntity[size];

	if(testOuputBalance)
	{
		int passedInputs = 0;
		int troughputLimitedInputs = 0;

		for (unsigned int i = 0; i < spawnBelts.size(); i++)
		{
			printAndMoveCursorBack("Progress: " + to_string(passedInputs) + "(" + to_string(i) + ") / " + to_string(spawnBelts.size()));

			memcpy(workingCopy, entities, size * sizeof(BeltEntity));

			for (unsigned int j = 0; j < spawnBelts.size(); j++)
			{
				if (i != j)
				{
					workingCopy[spawnBelts[j].id].maxTroughput = 0;
				}
			}

			updateEntities(workingCopy, size, iterations);

			float expectedResult = workingCopy[voidBelts[0].id].lastTroughput;
			int passedOutputs = 1;

			for (unsigned int j = 1; j < voidBelts.size(); j++)
			{
				float v = workingCopy[voidBelts[j].id].lastTroughput;
				if (fabsf(expectedResult - v) / expectedResult < 0.001)
				{
					passedOutputs++;
				}
				else
				{
#ifdef _DEBUG
					cout << "Output is " << v << " while expected is " << expectedResult << " (" << (fabsf(expectedResult - v) / expectedResult) << ")" << endl;
#endif
				}
			}

			if (workingCopy[spawnBelts[i].id].lastTroughput - workingCopy[spawnBelts[i].id].maxTroughput < -0.001)
			{
				troughputLimitedInputs++;
			}

			if (passedOutputs == voidBelts.size())
			{
				passedInputs++;
			}
		}

		cout << "Output balance: " << passedInputs << "/" << spawnBelts.size();
		if (troughputLimitedInputs > 0)
		{
			cout << "  (" << troughputLimitedInputs << " input" << ((troughputLimitedInputs == 1) ? " is" : "s are") << " troughput limited)" << endl;
		}
		else
		{
			cout << "              " << endl;
		}
	}

	if (testInputBalance)
	{
		int passedOutputs = 0;
		int troughputLimitedOutputs = 0;

		for (unsigned int i = 0; i < voidBelts.size(); i++)
		{
			printAndMoveCursorBack("Progress: " + to_string(passedOutputs) + "(" + to_string(i) + ") / " + to_string(voidBelts.size()));

			memcpy(workingCopy, entities, size * sizeof(BeltEntity));

			for (unsigned int j = 0; j < voidBelts.size(); j++)
			{
				if (i != j)
				{
					workingCopy[voidBelts[j].id].maxTroughput = 0;
				}
			}

			updateEntities(workingCopy, size, iterations);

			float expectedResult = workingCopy[spawnBelts[0].id].lastTroughput;
			int passedInputs = 1;

			for (unsigned int j = 1; j < spawnBelts.size(); j++)
			{
				float s = workingCopy[spawnBelts[j].id].lastTroughput;
				if (fabsf(expectedResult - s) / expectedResult < 0.001)
				{
					passedInputs++;
				}
				else
				{
#ifdef _DEBUG
					cout << "Output is " << s << " while expected is " << expectedResult << " (" << (fabsf(expectedResult - s) / expectedResult) << ")" << endl;
#endif
				}
			}

			if (workingCopy[voidBelts[i].id].lastTroughput - workingCopy[voidBelts[i].id].maxTroughput < -0.001)
			{
				troughputLimitedOutputs++;
			}

			if (passedInputs == spawnBelts.size())
			{
				passedOutputs++;
			}
		}

		cout << "Input balance: " << passedOutputs << "/" << voidBelts.size();
		if (troughputLimitedOutputs > 0)
		{
			cout << "   (" << troughputLimitedOutputs << " output" << ((troughputLimitedOutputs == 1) ? " is" : "s are") << " troughput limited)" << endl;
		}
		else
		{
			cout << "              " << endl;
		}
	}

	if (testFullLoadTroughput)
	{
		memcpy(workingCopy, entities, size * sizeof(BeltEntity));

		updateEntities(workingCopy, size, iterations);

		double maxInput = 0;
		double maxOutput = 0;

		for (unsigned int i = 0; i < size; i++)
		{
			if (entities[i].type == TYPE_SPAWN)
			{
				maxInput += entities[i].spawnAmount;
			}
			else if (entities[i].type == TYPE_VOID)
			{
				maxOutput += entities[i].voidAmount;
			}
		}

		maxOutput = min(maxInput, maxOutput);

		double actualOutput = 0;

		for (unsigned int i = 0; i < size; i++)
		{
			if (workingCopy[i].type == TYPE_VOID)
			{
				actualOutput += workingCopy[i].lastTroughput;
			}
		}

		double troughputPercentage = ((int)(actualOutput / maxOutput * 1000)) / 10.0;

		cout << "Troughput under full load: " << troughputPercentage << "%" << endl;
	}

	if (testAllTwoBeltTroughputCombinations)
	{
		BeltEntity* allBlocked = new BeltEntity[size];
		memcpy(allBlocked, entities, size * sizeof(BeltEntity));
		for (unsigned int i = 0; i < spawnBelts.size(); i++)
		{
			allBlocked[spawnBelts[i].id].spawnAmount = 0;
		}
		for (unsigned int i = 0; i < voidBelts.size(); i++)
		{
			allBlocked[voidBelts[i].id].voidAmount = 0;
		}

		double minTroughput = 100;
		int tested = 0;
		double lastProgress = -1;
		int toTest = ((spawnBelts.size() - 1) * (spawnBelts.size()) / 2) * ((voidBelts.size() - 1) * (voidBelts.size()) / 2);

		for (unsigned int i1 = 0; i1 < spawnBelts.size() - 1; i1++) for (unsigned int i2 = i1 + 1; i2 < spawnBelts.size(); i2++)
		{
			for (unsigned int o1 = 0; o1 < voidBelts.size() - 1; o1++) for (unsigned int o2 = o1 + 1; o2 < voidBelts.size(); o2++)
			{
				double progress = ((int)((tested++ / (double)toTest) * 1000)) / 10.0;
				if (progress != lastProgress)
				{
					stringstream ss;
					ss << "Min troughput: " << minTroughput << "%  Progress: " << progress << ((progress - ((int)progress) == 0) ? ".0%   " : "%   ");
					printAndMoveCursorBack(ss.str());
					lastProgress = progress;
				}

				memcpy(workingCopy, allBlocked, size * sizeof(BeltEntity));
				workingCopy[spawnBelts[i1].id].spawnAmount = spawnBelts[i1].troughput;
				workingCopy[spawnBelts[i2].id].spawnAmount = spawnBelts[i2].troughput;
				workingCopy[voidBelts[o1].id].voidAmount = voidBelts[o1].troughput;
				workingCopy[voidBelts[o2].id].voidAmount = voidBelts[o2].troughput;

				updateEntities(workingCopy, size, iterations);

				double maxInput = 0;
				double maxOutput = 0;

				maxInput += spawnBelts[i1].troughput;
				maxInput += spawnBelts[i2].troughput;
				maxOutput += voidBelts[o1].troughput;
				maxOutput += voidBelts[o2].troughput;

				maxOutput = min(maxInput, maxOutput);

				double actualOutput = 0;

				actualOutput += workingCopy[voidBelts[o1].id].lastTroughput;
				actualOutput += workingCopy[voidBelts[o2].id].lastTroughput;

				double troughputPercentage = ((int)(actualOutput / maxOutput * 1000)) / 10.0;

				if (troughputPercentage < minTroughput)
				{
					minTroughput = troughputPercentage;
				}
			}
		}

		cout << "Min troughput with two belts: " << minTroughput << "%                          " << endl;
	}

	cout << endl;
	delete[] workingCopy;
}

void printHelp()
{
	cout << "beltbalancer.exe -f=YOUR_BALANCER_FILE.txt ([-cpu]|[-gpu]|[-cudadev=N]) [-t] [-i=N] [-benchmark] [-o]" << endl;
}

int main(int argc, char** argv)
{
	cudaError_t cudaStatus;

	int iterations = -1;
	string file = "DUMB_ASS";
	int cudaDeviceId = 0;
	bool timeIt = false;
	bool optimize = false;

	for (int i = 1; i < argc; i++)
	{
		string arg = argv[i];
		if (arg.compare(0, 3, "-f=") == 0)
		{
			file = arg.substr(3, arg.length() - 3);
		}
		else if (arg.compare("-o") == 0)
		{
			optimize = true;
		}
		else if (arg.compare("-t") == 0)
		{
			testAllTwoBeltTroughputCombinations = true;
		}
		else if (arg.compare("-cpu") == 0)
		{
			useCPU = true;
		}
		else if (arg.compare("-gpu") == 0)
		{
			useCPU = false;
		}
		else if (arg.compare(0, 9, "-threads=") == 0)
		{
			threads = stoi(arg.substr(9));
			useCPU = false;
		}
		else if (arg.compare(0, 9, "-cudadev=") == 0)
		{
			cudaDeviceId = stoi(arg.substr(9));
			useCPU = false;
		}
		else if (arg.compare(0, 3, "-i=") == 0)
		{
			iterations = stoi(arg.substr(3));
		}
		else if (arg.compare("-benchmark") == 0)
		{
			timeIt = true;
		}
		else if (arg.compare("-h") == 0 || arg.compare("-?"))
		{
			printHelp();
			return 0;
		}
	}

	if (file.compare("DUMB_ASS") == 0)
	{
		printHelp();
		return 0;
	}

	if (!useCPU)
	{
		// Choose which GPU to run on
		cudaStatus = cudaSetDevice(cudaDeviceId);
		if (cudaStatus != cudaSuccess)
		{
			cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
			return 1;
		}
	}

	size_t size = 0;
	BeltEntity* belts = parseBlueprintString(loadBlueprintFile(file), &size, optimize);

	if (iterations == -1)
	{
		iterations = size * 5;
	}

	if (size == 0)
	{
		cerr << "Loading blueprint failed" << endl;
		return 1;
	}

	if (!timeIt)
	{
		testBalance(belts, size, iterations);
	}
	else
	{
		clock_t start;
		clock_t end;

		start = clock();

		updateEntities(belts, size, iterations);

		end = clock();

		double timeTaken = (end - start) / (double)CLOCKS_PER_SEC;

		cout << "Simulating " << size << " belt parts for " << iterations << " iterations took " << timeTaken << " seconds." << endl << endl;
	}

	delete[] belts;

	if (!useCPU)
	{
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}

    return 0;
}
