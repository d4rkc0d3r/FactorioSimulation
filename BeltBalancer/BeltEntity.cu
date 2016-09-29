#include "BeltEntity.cuh"

#define MIN(x, y) ((x) < (y) ? x : y)

__global__ void updateKernel(BeltEntity* entities)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	BeltEntity* b = entities + i;

	switch (b->type)
	{
	case TYPE_SPAWN:
		b->buffer = b->spawnAmount;
	case TYPE_BELT:
	case TYPE_UNDERGROUND_ENTRANCE:
	case TYPE_UNDERGROUND_EXIT:
		BeltEntity* next = entities + b->next;
		next->addToBuffer = MIN(b->maxTroughput, b->buffer);
		if (next->addToBuffer + next->buffer > next->maxTroughput * 2)
		{
			next->addToBuffer = next->maxTroughput * 2 - next->buffer;
		}
		b->substractFromBuffer = next->addToBuffer;
		break;
	case TYPE_VOID:
		b->substractFromBuffer = MIN(b->buffer, b->voidAmount);
		break;
	case TYPE_LEFT_SPLITTER:
		BeltEntity* r = entities + b->otherSplitterPart;
		BeltEntity* lnext = entities + b->next;
		BeltEntity* rnext = entities + r->next;
		float ldemand = lnext->maxTroughput * 2 - lnext->buffer;
		ldemand = MIN(ldemand, lnext->maxTroughput);
		ldemand = MIN(ldemand, b->maxTroughput);
		float rdemand = rnext->maxTroughput * 2 - rnext->buffer;
		rdemand = MIN(rdemand, rnext->maxTroughput);
		rdemand = MIN(rdemand, r->maxTroughput);
		float lsupply = MIN(b->maxTroughput, b->buffer);
		float rsupply = MIN(r->maxTroughput, r->buffer);
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
			else if(rdemand < halfSupply)
			{
				rnext->addToBuffer = rdemand;
				lnext->addToBuffer = supply - rdemand;
			}
			else
			{
				lnext->addToBuffer = halfSupply;
				rnext->addToBuffer = halfSupply;
			}
			b->substractFromBuffer = lsupply;
			r->substractFromBuffer = rsupply;
		}
		else
		{
			float halfDemand = demand / 2;
			lnext->addToBuffer = ldemand;
			rnext->addToBuffer = rdemand;
			if (lsupply < halfDemand)
			{
				b->substractFromBuffer = lsupply;
				r->substractFromBuffer = demand - lsupply;
			}
			else if (rsupply < halfDemand)
			{
				r->substractFromBuffer = rsupply;
				b->substractFromBuffer = demand - rsupply;
			}
			else
			{
				r->substractFromBuffer = halfDemand;
				b->substractFromBuffer = halfDemand;
			}
		}
		break;
	case TYPE_RIGHT_SPLITTER: // right splitter part gets updated together with the left part
	case TYPE_BLOCK:
	default:
			break;
	}

	__syncthreads();

	b->buffer += b->addToBuffer - b->substractFromBuffer;

	__syncthreads();
}

bool updateOnGPU(BeltEntity* entities, size_t size, unsigned int iterations, int threads)
{
	BeltEntity* dev_entities = 0;
	cudaError_t cudaStatus;

	int nSize = size - 1;
	int blocks = (nSize - 1) / threads + 1;
	int fSize = blocks * threads;

	BeltEntity* paddingBlocks = new BeltEntity[fSize - nSize];

	for (int i = 0; i < (fSize - nSize); i++)
	{
		BeltEntity b;
		b.type = TYPE_BLOCK;
		b.maxTroughput = 0;
		b.addToBuffer = 0;
		b.buffer = 0;
		b.substractFromBuffer = 0;
		b.next = -1;
		b.otherSplitterPart = -1;
		paddingBlocks[i] = b;
	}

	cudaStatus = cudaMalloc((void**)&dev_entities, (fSize) * sizeof(BeltEntity));
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_entities, entities + 1, (nSize)* sizeof(BeltEntity), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_entities + nSize, paddingBlocks, (fSize - nSize)* sizeof(BeltEntity), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	for (unsigned int i = 0; i < iterations; i++)
	{
		updateKernel << <blocks, threads >> >(dev_entities);
	}

	cudaStatus = cudaMemcpy(entities + 1, dev_entities, (nSize) * sizeof(BeltEntity), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

Error:
	cudaFree(dev_entities);

	return cudaStatus == cudaSuccess;
}

bool updateOnCPU(BeltEntity* entities, size_t size, unsigned int iterations)
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
				next->addToBuffer = MIN(b->maxTroughput, b->buffer);
				if (next->addToBuffer + next->buffer > next->maxTroughput * 2)
				{
					next->addToBuffer = next->maxTroughput * 2 - next->buffer;
				}
				b->substractFromBuffer = next->addToBuffer;
				break;
			case TYPE_VOID:
				b->substractFromBuffer = MIN(b->buffer, b->voidAmount);
				break;
			case TYPE_LEFT_SPLITTER:
				r = entities + b->otherSplitterPart + 1;
				lnext = entities + b->next + 1;
				rnext = entities + r->next + 1;
				ldemand = lnext->maxTroughput * 2 - lnext->buffer;
				ldemand = MIN(ldemand, lnext->maxTroughput);
				ldemand = MIN(ldemand, b->maxTroughput);
				rdemand = rnext->maxTroughput * 2 - rnext->buffer;
				rdemand = MIN(rdemand, rnext->maxTroughput);
				rdemand = MIN(rdemand, r->maxTroughput);
				lsupply = MIN(b->maxTroughput, b->buffer);
				rsupply = MIN(r->maxTroughput, r->buffer);
				demand = ldemand + rdemand;
				supply = lsupply + rsupply;
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
					b->substractFromBuffer = lsupply;
					r->substractFromBuffer = rsupply;
				}
				else
				{
					float halfDemand = demand / 2;
					lnext->addToBuffer = ldemand;
					rnext->addToBuffer = rdemand;
					if (lsupply < halfDemand)
					{
						b->substractFromBuffer = lsupply;
						r->substractFromBuffer = demand - lsupply;
					}
					else if (rsupply < halfDemand)
					{
						r->substractFromBuffer = rsupply;
						b->substractFromBuffer = demand - rsupply;
					}
					else
					{
						r->substractFromBuffer = halfDemand;
						b->substractFromBuffer = halfDemand;
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
			b->buffer += b->addToBuffer - b->substractFromBuffer;
		}
	}

	return true;
}
