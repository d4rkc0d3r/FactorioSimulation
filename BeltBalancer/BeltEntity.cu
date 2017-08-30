#include "BeltEntity.cuh"
#include <vector>
#include <iostream>
#include <thread>
#include <string>
#include <sstream>
#include "windows.h"
#include <time.h>
#include <mmintrin.h>
#include <random>

float inline minss(float a, float b)
{
	_mm_store_ss(&a, _mm_min_ss(_mm_set_ss(a), _mm_set_ss(b)));
	return a;
}

#define MIN(x, y) ((x) < (y) ? x : y)

using namespace std;

#ifdef _DEBUG
#define DEBUG(x) (x)
#else
#define DEBUG(x) ;
#endif

__device__ inline void updatePass1(BeltEntity* entities, int i)
{
	BeltEntity* b = entities + i;

	switch (b->type)
	{
	case TYPE_SPAWN:
		b->buffer = b->spawnAmount;
	case TYPE_BELT:
	case TYPE_UNDERGROUND_ENTRANCE:
	case TYPE_UNDERGROUND_EXIT:
		BeltEntity* next = entities + b->next;
		next->addToBuffer = MIN(b->maxThroughput, b->buffer);
		if (next->addToBuffer + next->buffer > next->maxThroughput * 2)
		{
			next->addToBuffer = next->maxThroughput * 2 - next->buffer;
		}
		b->subtractFromBuffer = next->addToBuffer;
		break;
	case TYPE_VOID:
		b->subtractFromBuffer = MIN(b->buffer, b->voidAmount);
		break;
	case TYPE_LEFT_SPLITTER:
		BeltEntity* r = entities + b->otherSplitterPart;
		BeltEntity* lnext = entities + b->next;
		BeltEntity* rnext = entities + r->next;
		float ldemand = lnext->maxThroughput * 2 - lnext->buffer;
		ldemand = MIN(ldemand, lnext->maxThroughput);
		ldemand = MIN(ldemand, b->maxThroughput);
		float rdemand = rnext->maxThroughput * 2 - rnext->buffer;
		rdemand = MIN(rdemand, rnext->maxThroughput);
		rdemand = MIN(rdemand, r->maxThroughput);
		float lsupply = MIN(b->maxThroughput, b->buffer);
		float rsupply = MIN(r->maxThroughput, r->buffer);
		float demand = ldemand + rdemand;
		float supply = lsupply + rsupply;
		if (demand >= supply)
		{
			float halfSupply = supply / 2;
			if (ldemand < halfSupply)
			{
				lnext->addToBuffer = ldemand;
				rnext->addToBuffer = supply - ldemand;
			}
			else if (rdemand < halfSupply)
			{
				rnext->addToBuffer = rdemand;
				lnext->addToBuffer = supply - rdemand;
			}
			else
			{
				lnext->addToBuffer = halfSupply;
				rnext->addToBuffer = halfSupply;
			}
			b->subtractFromBuffer = lsupply;
			r->subtractFromBuffer = rsupply;
		}
		else
		{
			float halfDemand = demand / 2;
			lnext->addToBuffer = ldemand;
			rnext->addToBuffer = rdemand;
			if (lsupply < halfDemand)
			{
				b->subtractFromBuffer = lsupply;
				r->subtractFromBuffer = demand - lsupply;
			}
			else if (rsupply < halfDemand)
			{
				r->subtractFromBuffer = rsupply;
				b->subtractFromBuffer = demand - rsupply;
			}
			else
			{
				r->subtractFromBuffer = halfDemand;
				b->subtractFromBuffer = halfDemand;
			}
		}
		break;
	case TYPE_RIGHT_SPLITTER: // right splitter part gets updated together with the left part
	case TYPE_BLOCK:
	default:
		break;
	}
}

__device__ inline void updatePass2(BeltEntity* entities, int i)
{
	BeltEntity* b = entities + i;

	b->buffer += b->addToBuffer - b->subtractFromBuffer;
}

__global__ void updateKernel(BeltEntity* entities)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	updatePass1(entities, i);

	__syncthreads();

	updatePass2(entities, i);

	__syncthreads();
}

__global__ void testThroughputKernel(BeltEntity* allEntities, size_t size, unsigned int iterations, int* beltIds, int inputBeltCount,
									 int outputBeltCount, int testCaseCount, float* testCaseData, float* results)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= testCaseCount)
	{
		return;
	}

	BeltEntity* entities = allEntities + index * size + 1;

	int ic = inputBeltCount;
	int oc = outputBeltCount;

	int testCaseOffset = index * (ic + oc);
	int* inputBeltIds = beltIds + testCaseOffset;
	int* outputBeltIds = inputBeltIds + ic;

	for (int i = 0; i < ic; i++)
	{
		entities[inputBeltIds[i]].maxThroughput *= testCaseData[testCaseOffset + i];
	}

	for (int i = 0; i < oc; i++)
	{
		entities[outputBeltIds[i]].maxThroughput *= testCaseData[testCaseOffset + ic + i];
	}

	for (int i = 0; i < iterations; i++)
	{
		for (int j = 0; j < size - 1; j++)
		{
			updatePass1(entities, j);
		}
		for (int j = 0; j < size - 1; j++)
		{
			updatePass2(entities, j);
		}
	}

	float maxInput = 0;
	float maxOutput = 0;

	for (int i = 0; i < ic; i++)
	{
		maxInput += entities[inputBeltIds[i]].maxThroughput;
	}

	for (int i = 0; i < oc; i++)
	{
		maxOutput += entities[outputBeltIds[i]].maxThroughput;
	}

	maxOutput = MIN(maxInput, maxOutput);

	float actualOutput = 0;

	for (int i = 0; i < oc; i++)
	{
		actualOutput += entities[outputBeltIds[i]].lastThroughput;
	}

	results[index] = actualOutput / maxOutput;
}

int countSetBits(unsigned int v)
{
	v = v - ((v >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}

double testThroughputCombinationsOnGPU(BeltEntity* entities, size_t size, unsigned int iterations, int minPopCount, int maxPopCount)
{
	vector<int> inputIds;
	vector<int> outputIds;

	for (int i = 0; i < size; i++)
	{
		if (entities[i].type == TYPE_SPAWN)
		{
			inputIds.push_back(i - 1);
		}
		else if (entities[i].type == TYPE_VOID)
		{
			outputIds.push_back(i - 1);
		}
	}

	DEBUG(cout << "Detected " << inputIds.size() << " inputs and " << outputIds.size() << " outputs" << endl);

	int testCaseCount = 0;
	int inputBeltCount = inputIds.size();
	int outputBeltCount = outputIds.size();

	vector<float> outputCombinations;
	for (int out = 0; out < (1 << outputBeltCount); out++)
	{
		int popCount = countSetBits(out);
		if (popCount < minPopCount || popCount > maxPopCount)
		{
			continue;
		}
		int outCopy = out;
		for (int i = 0; i < outputBeltCount; i++)
		{
			outputCombinations.push_back((float)(outCopy & 1));
			outCopy = outCopy >> 1;
		}
	}
	int outputCombinationsSize = outputCombinations.size() / outputBeltCount;

	vector<float> testCaseDataVector;
	float* inputData = new float[inputBeltCount];
	for (int in = 0; in < (1 << inputBeltCount); in++)
	{
		int popCount = countSetBits(in);
		if (popCount < minPopCount || popCount > maxPopCount)
		{
			continue;
		}
		int inCopy = in;
		for (int i = 0; i < inputBeltCount; i++)
		{
			inputData[i] = (float)(inCopy & 1);
			inCopy = inCopy >> 1;
		}
		for (int out = 0; out < outputCombinationsSize; out++)
		{
			for (int i = 0; i < inputBeltCount; i++)
			{
				testCaseDataVector.push_back(inputData[i]);
			}
			for (int i = 0; i < outputBeltCount; i++)
			{
				testCaseDataVector.push_back(outputCombinations[out * outputBeltCount + i]);
			}
			testCaseCount++;
		}
	}

	DEBUG(cout << "testCaseCount " << testCaseCount << endl);
	DEBUG(cout << "testCaseDataSize " << (sizeof(float)* testCaseDataVector.size()) << endl);

	float* dev_testCaseData = 0;
	cudaMalloc((void**)&dev_testCaseData, sizeof(float)* testCaseDataVector.size());
	cudaMemcpy(dev_testCaseData, &testCaseDataVector[0], sizeof(float)*testCaseDataVector.size(), cudaMemcpyHostToDevice);

	float* results = (float*)malloc(sizeof(float)* testCaseCount);
	for (unsigned int i = 0; i < testCaseCount; i++)
	{
		results[i] = -0.69f;
	}
	float* dev_results = 0;
	cudaMalloc((void**)&dev_results, sizeof(float)* testCaseCount);
	cudaMemcpy(dev_results, results, sizeof(float)* testCaseCount, cudaMemcpyHostToDevice);

	int* dev_beltIds = 0;
	cudaError_t cudaError = cudaMalloc((void**)&dev_beltIds, sizeof(int)* testCaseCount * (inputBeltCount + outputBeltCount));
	if (cudaError != cudaSuccess) {
		cerr << "cudaMalloc dev_beltIds failed: " << cudaError << endl;
	}
	cudaMemcpy(dev_beltIds, &inputIds[0], sizeof(int)* inputBeltCount, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_beltIds + inputBeltCount, &outputIds[0], sizeof(int)* outputBeltCount, cudaMemcpyHostToDevice);

	BeltEntity* dev_entities = 0;
	DEBUG(cout << "cudaMalloc dev_entities with " << (sizeof(BeltEntity)* size * testCaseCount / 1024) << " kbytes" << endl);
	cudaError = cudaMalloc((void**)&dev_entities, sizeof(BeltEntity)* size * testCaseCount);
	if (cudaError != cudaSuccess) {
		cerr << "cudaMalloc entities failed: " << cudaError << endl;
	}
	cudaMemcpy(dev_entities, entities, sizeof(BeltEntity) * size, cudaMemcpyHostToDevice);

	// duplicate arrays for the belt ids and the belt structures themselves
	for (int i = 1; i < testCaseCount; i++)
	{
		cudaMemcpy(dev_entities + size * i, dev_entities, sizeof(BeltEntity)* size, cudaMemcpyDeviceToDevice);
		cudaMemcpy(dev_beltIds + (inputBeltCount + outputBeltCount) * i, dev_beltIds, sizeof(int)* (inputBeltCount + outputBeltCount), cudaMemcpyDeviceToDevice);
	}

	const int threads = 256;
	int blocks = (testCaseCount - 1) / threads + 1;

	DEBUG(cout << "Launching kernel with dimensions: (" << blocks << "," << threads << ")" << endl);

	testThroughputKernel << <blocks, threads >> >(dev_entities, size, iterations, dev_beltIds, inputBeltCount, outputBeltCount, testCaseCount, dev_testCaseData, dev_results);
	cudaDeviceSynchronize();

	DEBUG(cout << "last cuda error: " << cudaGetErrorString(cudaPeekAtLastError()) << endl);
	
	cudaMemcpy(results, dev_results, sizeof(float)* testCaseCount, cudaMemcpyDeviceToHost);
	DEBUG(cout << "Copied results to host memory" << endl);

	float minimum = 1;
	for (unsigned int i = 0; i < testCaseCount; i++)
	{
#ifdef _DEBUG
		if (i < 32) cout << "results[" << i << "] = " << results[i] << endl;
#endif
		if (results[i] < minimum)
		{
			minimum = results[i];
		}
	}

	DEBUG(cout << "Free data" << endl);
	free(results);
	cudaFree(dev_beltIds);
	cudaFree(dev_entities);
	cudaFree(dev_testCaseData);
	cudaFree(dev_results);

	DEBUG(cout << "Return minimum of " << minimum << endl);
	return minimum;
}

int updateEntities(BeltEntity* entities, size_t size, unsigned int iterations);

void testThroughput(BeltEntity* source, size_t size, unsigned int iterations, vector<int>& inputIds, vector<int>& outputIds,
					long long startIndex, long long endIndex, vector<float>& inputData, vector<float>& outputData, float* results, long long* progress)
{
	BeltEntity* entities = new BeltEntity[size];

	for (long long index = startIndex; index < endIndex; index++)
	{
		memcpy(entities, source, size * sizeof(BeltEntity));

		long long inputDataSize = inputData.size() / inputIds.size();
		
		int inputOffset = (index % inputDataSize) * inputIds.size();
		int outputOffset = (index / inputDataSize) * outputIds.size();

		for (int i = 0; i < inputIds.size(); i++)
		{
			entities[inputIds[i]].maxThroughput *= inputData[inputOffset + i];
		}

		for (int i = 0; i < outputIds.size(); i++)
		{
			entities[outputIds[i]].maxThroughput *= outputData[outputOffset + i];
		}

		updateEntities(entities, size, iterations);

		float maxInput = 0;
		float maxOutput = 0;

		for (int i = 0; i < inputIds.size(); i++)
		{
			maxInput += entities[inputIds[i]].maxThroughput;
		}

		for (int i = 0; i < outputIds.size(); i++)
		{
			maxOutput += entities[outputIds[i]].maxThroughput;
		}

		maxOutput = MIN(maxInput, maxOutput);

		float actualOutput = 0;

		for (int i = 0; i < outputIds.size(); i++)
		{
			actualOutput += entities[outputIds[i]].lastThroughput;
		}

		*results = MIN(actualOutput / maxOutput, *results);
		++*progress;
	}

	delete [] entities;
}

void testRandomThroughput(BeltEntity* source, size_t size, unsigned int iterations, int randomSeed, float* results, long long* progress)
{
	vector<int> inputIds;
	vector<int> outputIds;
	for (int i = 0; i < size; i++)
	{
		if (source[i].type == TYPE_SPAWN)
		{
			inputIds.push_back(i);
		}
		else if (source[i].type == TYPE_VOID)
		{
			outputIds.push_back(i);
		}
	}

	int inputBeltCount = inputIds.size();
	int outputBeltCount = outputIds.size();

	vector<float> inputData;
	vector<float> outputData;

	uniform_int_distribution<int> uniformDistribution(0, 1);
	mt19937 rng;
	rng.seed(randomSeed);

	while (true)
	{
		inputData.clear();
		int ic = 0;
		for (int i = 0; i < inputBeltCount; i++)
		{
			int r = uniformDistribution(rng);
			ic += r;
			inputData.push_back((float)r);
		}
		outputData.clear();
		int oc = 0;
		for (int i = 0; i < outputBeltCount; i++)
		{
			int r = uniformDistribution(rng);
			oc += r;
			outputData.push_back((float)r);
		}
		if (ic > 0 && oc > 0)
		{
			testThroughput(source, size, iterations, inputIds, outputIds, 0, 1, inputData, outputData, results, progress);
		}
	}
}

string formatSeconds(long sec)
{
	long day = sec / (60 * 60 * 24);
	sec -= day * 60 * 60 * 24;
	long h = sec / (60 * 60);
	sec -= h * 60 * 60;
	long min = sec / 60;
	sec -= min * 60;
	stringstream ss;
	if (day != 0) ss << day << "d ";
	if (day != 0 || h != 0) ss << h << "h ";
	if (day != 0 || h != 0 || min != 0) ss << min << "m ";
	ss << sec << "s";
	return ss.str();
}

bool checkCorrectlyOrderedSplitterCombinations(int c, const vector<int>& mergedIds)
{
	vector<int> combinations;
	for (int i = 0; i < mergedIds.size(); i++)
	{
		combinations.push_back(c & 1);
		c = c >> 1;
	}
	for (int i = 0; i < mergedIds.size(); i++)
	{
		if (combinations[i] == 0 || mergedIds[i] == -1)
		{
			continue;
		}
		if (combinations[mergedIds[i]] == 0)
		{
			return false;
		}
	}
	return true;
}

double testThroughputCombinationsOnCPU(BeltEntity* entities, size_t size, unsigned int iterations, int minPopCount, int maxPopCount, int threadCount, bool printProgress)
{
	vector<int> inputIds;
	vector<int> outputIds;
	for (int i = 0; i < size; i++)
	{
		if (entities[i].type == TYPE_SPAWN)
		{
			inputIds.push_back(i);
		}
		else if (entities[i].type == TYPE_VOID)
		{
			outputIds.push_back(i);
		}
	}
	
	int inputBeltCount = inputIds.size();
	int outputBeltCount = outputIds.size();

	vector<int> mergedWithIn;
	for (int i = 0; i < inputBeltCount; i++)
	{
		int nextSplitterId = entities[inputIds[i]].next;
		if (entities[nextSplitterId + 1].type != TYPE_LEFT_SPLITTER)
		{
			mergedWithIn.push_back(-1);
			continue;
		}
		int matchId = -1;
		for (int j = 0; j < inputBeltCount; j++)
		{
			if (i == j) continue;
			BeltEntity b = entities[entities[inputIds[j]].next + 1];
			if (b.type == TYPE_RIGHT_SPLITTER && b.otherSplitterPart == nextSplitterId)
			{
				matchId = j;
				break;
			}
		}
		mergedWithIn.push_back(matchId);
#ifdef _DEBUG
		if (matchId != -1)
		{
			cout << "Input " << i << " shares a splitter with " << matchId << endl;
		}
#endif
	}

	vector<int> mergedWithOut;
	for (int i = 0; i < outputBeltCount; i++)
	{
		mergedWithOut.push_back(-1);
	}
	for (int i = 0; i < size; i++)
	{
		if (entities[i].type == TYPE_LEFT_SPLITTER &&
			entities[i].next != -1 &&
			entities[entities[i].next + 1].type == TYPE_VOID &&
			entities[entities[i].otherSplitterPart + 1].next != -1 &&
			entities[entities[entities[i].otherSplitterPart + 1].next + 1].type == TYPE_VOID)
		{
			int j = 0;
			for (; j < outputBeltCount; j++)
			{
				if (outputIds[j] == entities[i].next + 1)
				{
					break;
				}
			}
			int matchId = 0;
			for (; matchId < outputBeltCount; matchId++)
			{
				if (outputIds[matchId] == entities[entities[i].otherSplitterPart + 1].next + 1)
				{
					break;
				}
			}
			mergedWithOut[j] = matchId;
#ifdef _DEBUG
			cout << "Output " << j << " shares a splitter with " << matchId << endl;
#endif
		}
	}

	vector<float> inputCombinations;
	for (int in = 0; in < (1 << inputBeltCount); in++)
	{
		int popCount = countSetBits(in);
		if (popCount < minPopCount || popCount > maxPopCount)
		{
			continue;
		}
		if (!checkCorrectlyOrderedSplitterCombinations(in, mergedWithIn))
		{
			continue;
		}
		int inCopy = in;
		for (int i = 0; i < inputBeltCount; i++)
		{
			inputCombinations.push_back((float)(inCopy & 1));
			inCopy = inCopy >> 1;
		}
	}
	long long inputCombinationsSize = inputCombinations.size() / inputBeltCount;

	vector<float> outputCombinations;
	for (int out = 0; out < (1 << outputBeltCount); out++)
	{
		int popCount = countSetBits(out);
		if (popCount < minPopCount || popCount > maxPopCount)
		{
			continue;
		}
		if (!checkCorrectlyOrderedSplitterCombinations(out, mergedWithOut))
		{
			continue;
		}
		int outCopy = out;
		for (int i = 0; i < outputBeltCount; i++)
		{
			outputCombinations.push_back((float)(outCopy & 1));
			outCopy = outCopy >> 1;
		}
	}
	long long outputCombinationsSize = outputCombinations.size() / outputBeltCount;

	long long testCases = outputCombinationsSize * inputCombinationsSize;
	
	threadCount = MIN(threadCount, testCases);
	
	thread** threads = new thread*[threadCount];
	long long* progress = new long long[threadCount];

	vector<float> result(threadCount, 69.0f);
		
	for (int i = 0; i < threadCount; i++)
	{
		long long startIndex = (testCases / threadCount) * i;
		long long endIndex = (testCases / threadCount) * (i + 1);
		if (i == threadCount - 1)
		{
			endIndex = testCases;
		}
		progress[i] = 0;
		threads[i] = new thread(testThroughput, entities, size, iterations, inputIds, outputIds, startIndex, endIndex, inputCombinations, outputCombinations, &result[i], &progress[i]);
	}

	if (printProgress)
	{
		clock_t start;
		clock_t end;
		start = clock();
		long long prog = 0;
		while (prog < testCases)
		{
			Sleep(100);
			prog = 0;
			for (int i = 0; i < threadCount; i++)
			{
				prog += progress[i];
			}
			end = clock();
			double progPercent = prog / (double)testCases;
			double elapsed = ((end - start) / (double)CLOCKS_PER_SEC);
			long estimatedSeconds = elapsed / progPercent - elapsed;
			double p = round(progPercent * 1000) / 10;
			stringstream ss;
			ss << "Progress: " << p << ((p - ((int)p) == 0) ? ".0%" : "%") << ((p < 10) ? "  " : " ") << "| estimated time left: " << formatSeconds(estimatedSeconds) << "                ";
			printAndMoveCursorBack(ss.str());
		}
		printAndMoveCursorBack("                                                                   ");
	}
		
	for (int i = 0; i < threadCount; i++)
	{
		threads[i]->join();
		delete threads[i];
	}
		
	delete[] threads;
	delete[] progress;

	double minimum = 69;
	
	for (int i = 0; i < result.size(); i++)
	{
		if (result[i] < minimum)
		{
			minimum = result[i];
		}
	}
	
	return minimum;
}

double testThroughputCombinationsLocally(BeltEntity* entities, size_t size, unsigned int iterations, int threadCount, bool printProgress)
{
	vector<int> inputIds;
	vector<int> outputIds;
	for (int i = 0; i < size; i++)
	{
		if (entities[i].type == TYPE_SPAWN)
		{
			inputIds.push_back(i);
		}
		else if (entities[i].type == TYPE_VOID)
		{
			outputIds.push_back(i);
		}
	}

	int inputBeltCount = inputIds.size();
	int outputBeltCount = outputIds.size();

	int maxBelts = MIN(inputBeltCount, outputBeltCount) / 2;
	double minimum = 1;

	for (int beltCount = 2; beltCount <= maxBelts; beltCount++)
	{

		vector<float> inputCombinations;
		for (int off = 0; off <= inputBeltCount - beltCount; off++)
		{
			for (int i = 0; i < off; i++)
			{
				inputCombinations.push_back(0.0f);
			}
			for (int i = 0; i < beltCount; i++)
			{
				inputCombinations.push_back(1.0f);
			}
			for (int i = 0; i < inputBeltCount - beltCount - off; i++)
			{
				inputCombinations.push_back(0.0f);
			}
		}
		long long inputCombinationsSize = inputCombinations.size() / inputBeltCount;

		vector<float> outputCombinations;
		for (int off = 0; off <= outputBeltCount - beltCount; off++)
		{
			for (int i = 0; i < off; i++)
			{
				outputCombinations.push_back(0.0f);
			}
			for (int i = 0; i < beltCount; i++)
			{
				outputCombinations.push_back(1.0f);
			}
			for (int i = 0; i < outputBeltCount - beltCount - off; i++)
			{
				outputCombinations.push_back(0.0f);
			}
		}
		long long outputCombinationsSize = outputCombinations.size() / outputBeltCount;

		long long testCases = outputCombinationsSize * inputCombinationsSize;

		int actualThreadCount = MIN(threadCount, testCases);

		thread** threads = new thread*[actualThreadCount];
		long long* progress = new long long[actualThreadCount];

		vector<float> result;

		for (int i = 0; i < actualThreadCount; i++)
		{
			long long startIndex = (testCases / actualThreadCount) * i;
			long long endIndex = (testCases / actualThreadCount) * (i + 1);
			if (i == actualThreadCount - 1)
			{
				endIndex = testCases;
			}
			result.push_back(1.0f);
			progress[i] = 0;
			threads[i] = new thread(testThroughput, entities, size, iterations, inputIds, outputIds, startIndex, endIndex, inputCombinations, outputCombinations, &result[i], &progress[i]);
		}

		if (printProgress)
		{
			long long prog = 0;
			while (prog < testCases)
			{
				Sleep(100);
				prog = 0;
				for (int i = 0; i < actualThreadCount; i++)
				{
					prog += progress[i];
					minimum = MIN(minimum, result[i]);
				}
				double progPercent = prog / (double)testCases;
				double p = floor(progPercent * 1000) / 10;
				stringstream ss;
				ss << beltCount << "/" << maxBelts << " | subprogress: " << p << ((p - ((int)p) == 0) ? ".0%" : "%") << ((p < 10) ? "  " : " ");
				ss << " | min throughput: " << (round(minimum * 1000) / 10) << "%                ";
				printAndMoveCursorBack(ss.str());
			}
		}

		for (int i = 0; i < actualThreadCount; i++)
		{
			threads[i]->join();
			delete threads[i];
		}

		delete[] threads;
		delete[] progress;

		for (int i = 0; i < result.size(); i++)
		{
			if (result[i] < minimum)
			{
				minimum = result[i];
			}
		}

	}

	if (printProgress)
	{
		printAndMoveCursorBack("                                                                   ");
	}

	return minimum;
}

double testThroughputCombinationsRandomly(BeltEntity* entities, size_t size, unsigned int iterations, int threadCount)
{
	thread** threads = new thread*[threadCount];
	long long* progress = new long long[threadCount];

	vector<float> result(threadCount, 69.0f);

	for (int i = 0; i < threadCount; i++)
	{
		progress[i] = 0;
		threads[i] = new thread(testRandomThroughput, entities, size, iterations, rand(), &result[i], &progress[i]);
	}

	long long prog = 0;
	while (true)
	{
		Sleep(100);
		prog = 0;
		double minThroughput = 1;
		for (int i = 0; i < threadCount; i++)
		{
			prog += progress[i];
			if (minThroughput > result[i])
			{
				minThroughput = result[i];
			}
		}
		stringstream ss;
		ss << "Simulated combinations: " << prog << " | min throughput: " << (round(minThroughput * 1000) / 10) << "%                ";
		printAndMoveCursorBack(ss.str());
	}
}

int updateOnGPU(BeltEntity* entities, size_t size, unsigned int iterations, int threads)
{
	BeltEntity* dev_entities = 0;
	cudaError_t cudaStatus;

	int nSize = size;
	int blocks = (nSize - 1) / threads + 1;
	int fSize = blocks * threads;

	BeltEntity* paddingBlocks = new BeltEntity[fSize - nSize];

	for (int i = 0; i < (fSize - nSize); i++)
	{
		BeltEntity b;
		b.type = TYPE_BLOCK;
		b.maxThroughput = 0;
		b.addToBuffer = 0;
		b.buffer = 0;
		b.subtractFromBuffer = 0;
		b.next = -1;
		b.otherSplitterPart = -1;
		paddingBlocks[i] = b;
	}

	cudaStatus = cudaMalloc((void**)&dev_entities, (fSize)* sizeof(BeltEntity));
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_entities, entities, (nSize)* sizeof(BeltEntity), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_entities + nSize, paddingBlocks, (fSize - nSize)* sizeof(BeltEntity), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	for (unsigned int i = 0; i < iterations; i++)
	{
		updateKernel << <blocks, threads >> >(dev_entities + 1);
	}

	cudaStatus = cudaMemcpy(entities, dev_entities, (nSize) * sizeof(BeltEntity), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

Error:
	cudaFree(dev_entities);

	return (cudaStatus == cudaSuccess) ? iterations : 0;
}

int updateOnCPU(BeltEntity* entities, size_t size, unsigned int iterations)
{
	for (unsigned int j = 0; j < iterations; j++)
	{
		for (int i = 1; i < size; i++)
		{
			BeltEntity* b = entities + i;
			float ldemand = 0;
			float rdemand = 0;
			float lsupply = 0;
			float rsupply = 0;
			float demand = 0;
			float supply = 0;
			BeltEntity* r = 0;
			BeltEntity* lnext = 0;
			BeltEntity* rnext = 0;
			BeltEntity* next = 0;

			switch (b->type)
			{
			case TYPE_SPAWN:
				b->buffer = b->spawnAmount;
			case TYPE_BELT:
			case TYPE_UNDERGROUND_ENTRANCE:
			case TYPE_UNDERGROUND_EXIT:
				next = entities + b->next + 1;
				next->addToBuffer = minss(b->maxThroughput, b->buffer);
				next->addToBuffer = minss(next->addToBuffer, next->maxThroughput * 2 - next->buffer);
				b->subtractFromBuffer = next->addToBuffer;
				break;
			case TYPE_VOID:
				b->subtractFromBuffer = minss(b->buffer, b->voidAmount);
				break;
			case TYPE_LEFT_SPLITTER:
				r = entities + b->otherSplitterPart + 1;
				lnext = entities + b->next + 1;
				rnext = entities + r->next + 1;
				ldemand = minss(lnext->maxThroughput, b->maxThroughput);
				rdemand = minss(rnext->maxThroughput, r->maxThroughput);
				ldemand = minss(ldemand, lnext->maxThroughput + lnext->maxThroughput - lnext->buffer);
				rdemand = minss(rdemand, rnext->maxThroughput + rnext->maxThroughput - rnext->buffer);
				lsupply = minss(b->maxThroughput, b->buffer);
				rsupply = minss(r->maxThroughput, r->buffer);
				demand = ldemand + rdemand;
				supply = lsupply + rsupply;
				if (demand >= supply)
				{
					float halfSupply = supply / 2;
					b->subtractFromBuffer = lsupply;
					r->subtractFromBuffer = rsupply;
					if (ldemand < halfSupply)
					{
						lnext->addToBuffer = ldemand;
						rnext->addToBuffer = supply - ldemand;
					}
					else if (rdemand < halfSupply)
					{
						rnext->addToBuffer = rdemand;
						lnext->addToBuffer = supply - rdemand;
					}
					else
					{
						lnext->addToBuffer = halfSupply;
						rnext->addToBuffer = halfSupply;
					}
				}
				else
				{
					float halfDemand = demand / 2;
					lnext->addToBuffer = ldemand;
					rnext->addToBuffer = rdemand;
					if (lsupply < halfDemand)
					{
						b->subtractFromBuffer = lsupply;
						r->subtractFromBuffer = demand - lsupply;
					}
					else if (rsupply < halfDemand)
					{
						r->subtractFromBuffer = rsupply;
						b->subtractFromBuffer = demand - rsupply;
					}
					else
					{
						r->subtractFromBuffer = halfDemand;
						b->subtractFromBuffer = halfDemand;
					}
				}
				break;
			case TYPE_RIGHT_SPLITTER: // right splitter part gets updated together with the left part
			default:
				break;
			}
		}

		for (int i = 1; i < size; i++)
		{
			BeltEntity* b = entities + i;
			b->buffer += b->addToBuffer - b->subtractFromBuffer;
		}
	}

	return iterations;
}

int updateOnCPU(BeltEntity* entities, size_t size, unsigned int iterations, double throughputThresholdToFinish)
{
	vector<BeltEntity*> spawnBelts;
	vector<BeltEntity*> voidBelts;

	int iterationCount = 0;

	for (int i = 0; i < size; i++)
	{
		if (entities[i].type == TYPE_SPAWN)
		{
			spawnBelts.push_back(entities + i);
		}
		else if (entities[i].type == TYPE_VOID)
		{
			voidBelts.push_back(entities + i);
		}
	}

	double troughputDifference = spawnBelts.size();

	do
	{
		updateOnCPU(entities, size, iterations);

		iterationCount += iterations;
		troughputDifference = 0;

		for (int i = 0; i < spawnBelts.size(); i++)
		{
			troughputDifference += spawnBelts[i]->lastThroughput;
		}
		for (int i = 0; i < voidBelts.size(); i++)
		{
			troughputDifference -= voidBelts[i]->lastThroughput;
		}
	}
	while (abs(troughputDifference) > throughputThresholdToFinish);

	return iterationCount;
}
