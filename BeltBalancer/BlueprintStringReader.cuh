#ifndef BLUEPRINTSTRINGREADER_H
#define BLUEPRINTSTRINGREADER_H

#include "BeltEntity.cuh"
#include <string>

BeltEntity* parseBlueprintString(std::string blueprint, size_t* outputSize, bool optimize);

#endif // !BLUEPRINTSTRINGREADER_H
