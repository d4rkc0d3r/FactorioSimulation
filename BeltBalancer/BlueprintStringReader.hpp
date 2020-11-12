#ifndef BLUEPRINTSTRINGREADER_H
#define BLUEPRINTSTRINGREADER_H

#include "BeltEntity.hpp"
#include <string>

BeltEntity* parseBlueprintString(std::string blueprint, size_t* outputSize, bool optimize);

#endif // !BLUEPRINTSTRINGREADER_H
