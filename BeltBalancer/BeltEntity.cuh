#ifndef BELTENTITY_H
#define BELTENTITY_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>

#define TYPE_BLOCK 0
#define TYPE_SPAWN 1
#define TYPE_VOID 2
#define TYPE_BELT 3
#define TYPE_UNDERGROUND_ENTRANCE 4
#define TYPE_UNDERGROUND_EXIT 5
#define TYPE_LEFT_SPLITTER 6
#define TYPE_RIGHT_SPLITTER 7
#define TYPE_PLS_DELETE 8

struct BeltEntity
{
	int type;
	float buffer;
	float addToBuffer;
	union {
		float subtractFromBuffer;
		float lastThroughput;
	};
	union {
		float maxThroughput;
		float voidAmount;
		float spawnAmount;
	};
	int next;
	int otherSplitterPart;
};

// result is minimum throughput
double testThroughputCombinationsOnGPU(BeltEntity* entities, size_t size, unsigned int iterations, int minPopCount, int maxPopCount);
double testThroughputCombinationsOnCPU(BeltEntity* entities, size_t size, unsigned int iterations, int minPopCount, int maxPopCount, int threadCount, bool printProgress);
double testThroughputCombinationsLocally(BeltEntity* entities, size_t size, unsigned int iterations, int threadCount, bool printProgress);
double testThroughputCombinationsRandomly(BeltEntity* entities, size_t size, unsigned int iterations, int threadCount);

int updateOnGPU(BeltEntity* entities, size_t size, unsigned int iterations, int threads);
int updateOnCPU(BeltEntity* entities, size_t size, unsigned int iterations);
int updateOnCPU(BeltEntity* entities, size_t size, unsigned int iterations, double throughputThresholdToFinish);
int updateOnCPUSorted(BeltEntity* entities, size_t size, unsigned int iterations, double throughputThresholdToFinish);

void printAndMoveCursorBack(std::string str);

#endif // BELTENTITY_H
