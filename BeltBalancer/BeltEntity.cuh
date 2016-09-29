#ifndef BELTENTITY_H
#define BELTENTITY_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TYPE_BELT 0
#define TYPE_LEFT_SPLITTER 1
#define TYPE_RIGHT_SPLITTER 2
#define TYPE_UNDERGROUND_ENTRANCE 3
#define TYPE_UNDERGROUND_EXIT 4
#define TYPE_SPAWN 5
#define TYPE_VOID 6
#define TYPE_BLOCK 7
#define TYPE_PLS_DELETE 8

struct BeltEntity
{
	int type;
	float buffer;
	float addToBuffer;
	union{
		float substractFromBuffer;
		float lastTroughput;
	};
	union {
		float maxTroughput;
		float voidAmount;
		float spawnAmount;
	};
	int next;
	int otherSplitterPart;
};

bool updateOnGPU(BeltEntity* entities, size_t size, unsigned int iterations, int threads);
bool updateOnCPU(BeltEntity* entities, size_t size, unsigned int iterations);

#endif // BELTENTITY_H
