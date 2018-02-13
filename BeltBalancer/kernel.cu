
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
#include <regex>

using namespace std;

#ifdef __unix__ 

void printAndMoveCursorBack(string str)
{
	printf("\33[2K\r");
	printf(str);
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
bool printProgress = true;
bool adaptiveIterationCount = false;
bool isSorted = true;
bool testInputBalance = true;
bool testOuputBalance = true;
bool testFullLoadThroughput = true;
bool testAllTwoBeltThroughputCombinations = false;
bool testAllThroughputCombinationsGPU = false;
bool testAllThroughputCombinationsCPU = false;
bool testRandomThroughputCombinations = false;
bool testLocalThroughputCombinations = false;
int threads = 256;
int cpuThreads = 1;

int minIterations = 1 << 30;
int maxIterations = 0;

int updateEntities(BeltEntity* entities, size_t size, unsigned int iterations)
{
	int iter;
	if (useCPU)
	{
		if (adaptiveIterationCount)
		{
			if (isSorted)
			{
				iter = updateOnCPUSorted(entities, size, iterations, 0.0001);
			}
			else
			{
				iter = updateOnCPU(entities, size, iterations, 0.0001);
			}
		}
		else
		{
			iter = updateOnCPU(entities, size, iterations);
		}
	}
	else
	{
		iter = updateOnGPU(entities, size, iterations, threads);
	}
	minIterations = min(minIterations, iter);
	maxIterations = max(maxIterations, iter);
	return iter;
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
			cout << "(" << t << ", " << entities[i].buffer << ", " << entities[i].lastThroughput << ", " << i - 1 << ", " << entities[i].next << ")" << endl;
	}
	cout << endl;
}

string getClipboard()
{
	if (!OpenClipboard(nullptr))
	{
		return "";
	}
	HANDLE hData = GetClipboardData(CF_TEXT);
	if (hData == nullptr)
	{
		return "";
	}
	char * pszText = static_cast<char*>(GlobalLock(hData));
	if (pszText == nullptr)
	{
		return "";
	}
	string text(pszText);
	GlobalUnlock(hData);
	CloseClipboard();
	return text;
}

string loadBlueprintFile(string fileName)
{
	string output;
	if (fileName.compare("CLIPBOARD") == 0)
	{
		output = getClipboard();
		if (!output.empty())
		{
			return output;
		}
	}
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

struct IdThroughputHelper
{
	int id;
	float Throughput;
};

void testBalance(BeltEntity* entities, size_t size, int iterations)
{
	vector<IdThroughputHelper> spawnBelts;
	vector<IdThroughputHelper> voidBelts;

	for (unsigned int i = 0; i < size; i++)
	{
		IdThroughputHelper t;
		t.id = i;
		t.Throughput = entities[i].maxThroughput;
		if (entities[i].type == TYPE_SPAWN)
		{
			spawnBelts.push_back(t);
		}
		else if (entities[i].type == TYPE_VOID)
		{
			voidBelts.push_back(t);
		}
	}

	BeltEntity* workingCopy = new BeltEntity[size];

	if(testOuputBalance)
	{
		int passedInputs = 0;
		int ThroughputLimitedInputs = 0;

		for (unsigned int i = 0; i < spawnBelts.size(); i++)
		{
			if (printProgress)
			{
				printAndMoveCursorBack("Progress: " + to_string(passedInputs) + "(" + to_string(i) + ") / " + to_string(spawnBelts.size()));
			}

			memcpy(workingCopy, entities, size * sizeof(BeltEntity));

			for (unsigned int j = 0; j < spawnBelts.size(); j++)
			{
				if (i != j)
				{
					workingCopy[spawnBelts[j].id].maxThroughput = 0;
				}
			}

			updateEntities(workingCopy, size, iterations);

			float expectedResult = workingCopy[voidBelts[0].id].lastThroughput;
			int passedOutputs = 1;

			for (unsigned int j = 1; j < voidBelts.size(); j++)
			{
				float v = workingCopy[voidBelts[j].id].lastThroughput;
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

			if (workingCopy[spawnBelts[i].id].lastThroughput - workingCopy[spawnBelts[i].id].maxThroughput < -0.001)
			{
				ThroughputLimitedInputs++;
			}

			if (passedOutputs == voidBelts.size())
			{
				passedInputs++;
			}
		}

		cout << "Output balance: " << passedInputs << "/" << spawnBelts.size();
		if (ThroughputLimitedInputs > 0)
		{
			cout << "  (" << ThroughputLimitedInputs << " input" << ((ThroughputLimitedInputs == 1) ? " is" : "s are") << " Throughput limited)" << endl;
		}
		else if (printProgress)
		{
			cout << "              " << endl;
		}
		else
		{
			cout << endl;
		}
	}

	if (testInputBalance)
	{
		int passedOutputs = 0;
		int ThroughputLimitedOutputs = 0;

		for (unsigned int i = 0; i < voidBelts.size(); i++)
		{
			if (printProgress)
			{
				printAndMoveCursorBack("Progress: " + to_string(passedOutputs) + "(" + to_string(i) + ") / " + to_string(voidBelts.size()));
			}

			memcpy(workingCopy, entities, size * sizeof(BeltEntity));

			for (unsigned int j = 0; j < voidBelts.size(); j++)
			{
				if (i != j)
				{
					workingCopy[voidBelts[j].id].maxThroughput = 0;
				}
			}

			updateEntities(workingCopy, size, iterations);

			float expectedResult = workingCopy[spawnBelts[0].id].lastThroughput;
			int passedInputs = 1;

			for (unsigned int j = 1; j < spawnBelts.size(); j++)
			{
				float s = workingCopy[spawnBelts[j].id].lastThroughput;
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

			if (workingCopy[voidBelts[i].id].lastThroughput - workingCopy[voidBelts[i].id].maxThroughput < -0.001)
			{
				ThroughputLimitedOutputs++;
			}

			if (passedInputs == spawnBelts.size())
			{
				passedOutputs++;
			}
		}

		cout << "Input balance: " << passedOutputs << "/" << voidBelts.size();
		if (ThroughputLimitedOutputs > 0)
		{
			cout << "   (" << ThroughputLimitedOutputs << " output" << ((ThroughputLimitedOutputs == 1) ? " is" : "s are") << " Throughput limited)" << endl;
		}
		else if (printProgress)
		{
			cout << "              " << endl;
		}
		else
		{
			cout << endl;
		}
	}

	if (testFullLoadThroughput)
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
				actualOutput += workingCopy[i].lastThroughput;
			}
		}

		double ThroughputPercentage = (round(actualOutput / maxOutput * 1000)) / 10.0;

		cout << "Throughput under full load: " << ThroughputPercentage << "%" << endl;
	}

	if (testAllTwoBeltThroughputCombinations)
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

		double minThroughput = 100;
		int tested = 0;
		double lastProgress = -1;
		int toTest = ((spawnBelts.size() - 1) * (spawnBelts.size()) / 2) * ((voidBelts.size() - 1) * (voidBelts.size()) / 2);

		for (unsigned int i1 = 0; i1 < spawnBelts.size() - 1; i1++) for (unsigned int i2 = i1 + 1; i2 < spawnBelts.size(); i2++)
		{
			for (unsigned int o1 = 0; o1 < voidBelts.size() - 1; o1++) for (unsigned int o2 = o1 + 1; o2 < voidBelts.size(); o2++)
			{
				if (printProgress)
				{
					double progress = ((int)((tested++ / (double)toTest) * 1000)) / 10.0;
					if (progress != lastProgress)
					{
						stringstream ss;
						ss << "Min Throughput: " << minThroughput << "%  Progress: " << progress << ((progress - ((int)progress) == 0) ? ".0%   " : "%   ");
						printAndMoveCursorBack(ss.str());
						lastProgress = progress;
					}
				}

				memcpy(workingCopy, allBlocked, size * sizeof(BeltEntity));
				workingCopy[spawnBelts[i1].id].spawnAmount = spawnBelts[i1].Throughput;
				workingCopy[spawnBelts[i2].id].spawnAmount = spawnBelts[i2].Throughput;
				workingCopy[voidBelts[o1].id].voidAmount = voidBelts[o1].Throughput;
				workingCopy[voidBelts[o2].id].voidAmount = voidBelts[o2].Throughput;

				updateEntities(workingCopy, size, iterations);

				double maxInput = 0;
				double maxOutput = 0;

				maxInput += spawnBelts[i1].Throughput;
				maxInput += spawnBelts[i2].Throughput;
				maxOutput += voidBelts[o1].Throughput;
				maxOutput += voidBelts[o2].Throughput;

				maxOutput = min(maxInput, maxOutput);

				double actualOutput = 0;

				actualOutput += workingCopy[voidBelts[o1].id].lastThroughput;
				actualOutput += workingCopy[voidBelts[o2].id].lastThroughput;

				double throughputPercentage = (round(actualOutput / maxOutput * 1000)) / 10.0;

				if (throughputPercentage < minThroughput)
				{
					minThroughput = throughputPercentage;
				}
			}
		}

		cout << "Min Throughput with two belts: " << minThroughput << ((printProgress) ? "%                          " : "%") << endl;
	}

	if (testLocalThroughputCombinations)
	{
		double minThroughput = round(testThroughputCombinationsLocally(entities, size, iterations, cpuThreads, printProgress) * 1000) / 10;

		cout << "Min Throughput with local combinations: " << minThroughput << "%" << endl;
	}

	if (testAllThroughputCombinationsCPU)
	{
		double minThroughput = round(testThroughputCombinationsOnCPU(entities, size, iterations, (spawnBelts.size() + voidBelts.size() <= 16) ? 1 : 2, 16, cpuThreads, printProgress) * 1000) / 10;

		cout << "Min Throughput with all combinations: " << minThroughput << "%" << endl;
	}

	if (testAllThroughputCombinationsGPU)
	{
		double minThroughput = round(testThroughputCombinationsOnGPU(entities, size, iterations, (spawnBelts.size() + voidBelts.size() <= 16) ? 1 : 2, 16) * 1000) / 10;

		cout << "Min Throughput with all combinations: " << minThroughput << "%" << endl;
	}

	if (testRandomThroughputCombinations)
	{
		testThroughputCombinationsRandomly(entities, size, iterations, cpuThreads);
	}

	delete[] workingCopy;
}

void sortEntities(BeltEntity* entities, size_t size)
{
	BeltEntity* sortedEntities = new BeltEntity[size];
	unsigned int* newId = new unsigned int[size];
	for (unsigned int i = 0; i < size; i++)
	{
		newId[i] = 0;
	}
	unsigned int currentIndex = 0;
	for (unsigned int typeId = 0; typeId <= TYPE_RIGHT_SPLITTER; typeId++)
	{
		for (unsigned int i = 0; i < size; i++)
		{
			if (entities[i].type == typeId)
			{
				sortedEntities[currentIndex] = entities[i];
				newId[i] = currentIndex - 1;
				currentIndex++;
			}
		}
	}
	for (unsigned int i = 0; i < size; i++)
	{
		sortedEntities[i].next = newId[sortedEntities[i].next + 1];
		sortedEntities[i].otherSplitterPart = newId[sortedEntities[i].otherSplitterPart + 1];
	}
	memcpy(entities, sortedEntities, size * sizeof(BeltEntity));
	delete[] sortedEntities;
	delete[] newId;
}

void printHelp()
{
	cout << "beltbalancer.exe [-f=YOUR_BALANCER_FILE.txt] ([-cpu]|[-gpu]|[-cudadev=N]) [-t2]" << endl;
	cout << "                 [-tall(cpu[N]|gpu)] [-trandom[N]] [-i=N] [-benchmark] [-time]" << endl;
	cout << "                 [-s] [-a]" << endl;
	cout << "             " << endl;
	cout << "  -f=FILE    loads the blueprint string file FILE, if not found tries again" << endl;
	cout << "             with %APPDATA%\\factorio\\script-output\\blueprint-string\\FILE" << endl;
	cout << "             if not specified, it loads the string from clipboard" << endl;
	cout << "  -t2        tests all throughput combinations where exactly two inputs and" << endl;
	cout << "             outputs are used" << endl;
	cout << "  -tall      tests all throughput combinations where more or equal to two" << endl;
	cout << "             inputs and outputs are used. N specifies how many threads will" << endl;
	cout << "             be launched and is 1 by default" << endl;
	cout << "  -trandom   tests random throughput combinations. N specifies how many threads" << endl;
	cout << "             will be launched and is 1 by default" << endl;
	cout << "  -i=N       specifies the number of iterations the simulation should run" << endl;
	cout << "             default is 2 * (2 * nSplitter + nInputs + nOutputs + 1) or 5" << endl;
	cout << "             if -a is also selected" << endl;
	cout << "  -time      times the complete testing time needed excluding loading and" << endl;
	cout << "             parsing the blueprint file" << endl;
	cout << "  -s         does suppress the ongoing progress display" << endl;
	cout << "             useful if you pipe the output to a file" << endl;
	cout << "  -a         does run the simulation until it converges instead of running" << endl;
	cout << "             it for a fixed amount of steps. Massive speed gains but does" << endl;
	cout << "             not converge for a minority of blueprints" << endl;
	cout << "  -benchmark times only the simulation time needed to run the specified amount" << endl;
	cout << "             of iterations" << endl;
}

int main(int argc, char** argv)
{
	cudaError_t cudaStatus;

	int iterations = -1;
	string file = "CLIPBOARD";
	int cudaDeviceId = -1;
	bool doBenchmark = false;
	bool timeIt = false;
	bool optimize = true;
	bool preSort = true;
	printProgress = true;

	for (int i = 1; i < argc; i++)
	{
		string arg = argv[i];
		if (arg.compare(0, 3, "-f=") == 0)
		{
			file = arg.substr(3, arg.length() - 3);
		}
		else if (arg.compare("-no") == 0)
		{
			optimize = false;
		}
		else if (arg.compare("-s") == 0)
		{
			printProgress = false;
		}
		else if (arg.compare("-t2") == 0)
		{
			testAllTwoBeltThroughputCombinations = true;
		}
		else if (regex_match(arg, regex("-tallcpu(\\d*)")))
		{
			testAllThroughputCombinationsCPU = true;
			if (arg.length() > 8)
			{
				cpuThreads = stoi(arg.substr(8));
			}
		}
		else if (regex_match(arg, regex("-tall(\\d*)")))
		{
			testAllThroughputCombinationsCPU = true;
			if (arg.length() > 5)
			{
				cpuThreads = stoi(arg.substr(5));
			}
		}
		else if (regex_match(arg, regex("-trandom(\\d*)")))
		{
			testRandomThroughputCombinations = true;
			if (arg.length() > 8)
			{
				cpuThreads = stoi(arg.substr(8));
			}
		}
		else if (regex_match(arg, regex("-tlocal(\\d*)")))
		{
			testLocalThroughputCombinations = true;
			if (arg.length() > 7)
			{
				cpuThreads = stoi(arg.substr(7));
			}
		}
		else if (arg.compare("-tallgpu") == 0)
		{
			testAllThroughputCombinationsGPU = true;
		}
		else if (arg.compare("-time") == 0)
		{
			timeIt = true;
		}
		else if (arg.compare("-cpu") == 0)
		{
			useCPU = true;
		}
		else if (arg.compare("-gpu") == 0)
		{
			useCPU = false;
			if (cudaDeviceId == -1)
			{
				cudaDeviceId = 0;
			}
		}
		else if (arg.compare(0, 9, "-threads=") == 0)
		{
			threads = stoi(arg.substr(9));
			useCPU = false;
		}
		else if (arg.compare(0, 9, "-cudadev=") == 0)
		{
			cudaDeviceId = stoi(arg.substr(9));
		}
		else if (arg.compare(0, 3, "-i=") == 0)
		{
			iterations = stoi(arg.substr(3));
		}
		else if (arg.compare("-benchmark") == 0)
		{
			doBenchmark = true;
		}
		else if (arg.compare("-a") == 0)
		{
			adaptiveIterationCount = true;
		}
		else if (arg.compare("-nosort") == 0)
		{
			preSort = false;
			isSorted = false;
		}
		else if (arg.compare("-h") == 0 || arg.compare("-?") == 0)
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

	if (cudaDeviceId != -1)
	{
		// Choose which GPU to run on
		cudaStatus = cudaSetDevice(cudaDeviceId);
		if (cudaStatus != cudaSuccess)
		{
			cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << endl;
			return 1;
		}
	}

	string fileContent = loadBlueprintFile(file);
	if (fileContent.empty())
	{
		cerr << "File not found" << endl;
		return 1;
	}

	size_t size = 0;
	BeltEntity* belts = parseBlueprintString(fileContent, &size, optimize);

	if (iterations == -1)
	{
		if (adaptiveIterationCount)
		{
			iterations = 5;
		}
		else
		{
			iterations = size * 2;
		}
	}
	

	if (size == 0)
	{
		cerr << "Loading blueprint failed" << endl;
		return 1;
	}

	clock_t start;
	clock_t end;

	start = clock();

	if (preSort)
	{
		// sorting the array helps the branch predictor in UpdateOnCPU and thus improves performance by ~3.5%
		sortEntities(belts, size);
	}

	if (!doBenchmark)
	{
		testBalance(belts, size, iterations);
	}
	else
	{
		updateEntities(belts, size, iterations);
	}

	end = clock();

	double timeTaken = (end - start) / (double)CLOCKS_PER_SEC;

	if (doBenchmark)
	{
		cout << "Simulating " << size << " belt parts for " << iterations << " iterations took " << timeTaken << " seconds." << endl;
	}
	else if (timeIt)
	{
		cout << "Test took " << timeTaken << " seconds." << endl;
		cout << "Iteration result: " << minIterations << " | " << maxIterations << endl;
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
