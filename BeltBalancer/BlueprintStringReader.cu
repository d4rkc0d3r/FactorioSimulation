#include "BlueprintStringReader.cuh"

#include <vector>
#include "base64.h"
#include "lib\zlib.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <set>
#include "JSONObject.h"

using namespace std;

/// from http://windrealm.org/tutorials/decompress-gzip-stream.php
bool gzipInflate(const std::string& compressedBytes, std::string& uncompressedBytes) {
	if (compressedBytes.size() == 0) {
		uncompressedBytes = compressedBytes;
		return true;
	}

	uncompressedBytes.clear();

	unsigned full_length = compressedBytes.size();
	unsigned half_length = compressedBytes.size() / 2;

	unsigned uncompLength = full_length;
	char* uncomp = (char*)calloc(sizeof(char), uncompLength);

	z_stream strm;
	strm.next_in = (Bytef *)compressedBytes.c_str();
	strm.avail_in = compressedBytes.size();
	strm.total_out = 0;
	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;

	bool done = false;

	if (inflateInit2(&strm, MAX_WBITS + 16) != Z_OK) {
		free(uncomp);
		return false;
	}

	while (!done) {
		// If our output buffer is too small  
		if (strm.total_out >= uncompLength) {
			// Increase size of output buffer  
			char* uncomp2 = (char*)calloc(sizeof(char), uncompLength + half_length);
			memcpy(uncomp2, uncomp, uncompLength);
			uncompLength += half_length;
			free(uncomp);
			uncomp = uncomp2;
		}

		strm.next_out = (Bytef *)(uncomp + strm.total_out);
		strm.avail_out = uncompLength - strm.total_out;

		// Inflate another chunk.  
		int err = inflate(&strm, Z_SYNC_FLUSH);
		if (err == Z_STREAM_END) done = true;
		else if (err != Z_OK)  {
			break;
		}
	}

	if (inflateEnd(&strm) != Z_OK) {
		free(uncomp);
		return false;
	}

	for (size_t i = 0; i<strm.total_out; ++i) {
		uncompressedBytes += uncomp[i];
	}
	free(uncomp);
	return true;
}

/// from https://panthema.net/2007/0328-ZLibString.html
std::string decompress_string(const std::string& str)
{
	z_stream zs;                        // z_stream is zlib's control structure
	memset(&zs, 0, sizeof(zs));

	if (inflateInit(&zs) != Z_OK)
		throw(std::runtime_error("inflateInit failed while decompressing."));

	zs.next_in = (Bytef*)str.data();
	zs.avail_in = str.size();

	int ret;
	char outbuffer[32768];
	std::string outstring;

	// get the decompressed bytes blockwise using repeated calls to inflate
	do {
		zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
		zs.avail_out = sizeof(outbuffer);

		ret = inflate(&zs, 0);

		if (outstring.size() < zs.total_out) {
			outstring.append(outbuffer,
				zs.total_out - outstring.size());
		}

	} while (ret == Z_OK);

	inflateEnd(&zs);

	if (ret != Z_STREAM_END) {          // an error occurred that was not EOF
		std::ostringstream oss;
		oss << "Exception during zlib decompression: (" << ret << ") "
			<< zs.msg;
		throw(std::runtime_error(oss.str()));
	}

	return outstring;
}

struct BPEntity
{
	string name;
	double x;
	double y;

	// 0 = -y, 2 = +x, 4 = +y, 6 = -x
	int direction;
};

vector<BPEntity> parseVanillaJSON(string jsonString)
{
	JSONObject* json = new JSONObject(jsonString);
	vector<BPEntity> out;
#ifdef _DEBUG
	cout << json->ToString(2) << endl;
#endif // _DEBUG
	vector<JSONObject*>* entities = json->GetPath("blueprint.entities")->GetArray();

	for (int i = 0; i < entities->size(); i++)
	{
		JSONObject* j = entities->at(i);
		BPEntity e;
		e.name = j->GetString("name") + ((j->Get("type") != nullptr) ? "-" + j->GetString("type") : "");
		e.direction = (j->Get("direction") != nullptr) ? j->GetNumber("direction") : 0;
		e.x = j->Get("position")->GetNumber("x");
		e.y = j->Get("position")->GetNumber("y");
		out.push_back(e);
	}

	delete json;
	return out;
}

BPEntity parseEntity(const string& s, unsigned int& pos)
{
	BPEntity e;
	e.name = "";
	e.direction = 0;
	e.x = 0;
	e.y = 0;

	if (s[pos++] != '{')
	{
		goto error;
	}

	while (s[pos] != '}')
	{
		if (s[pos] == ',')
		{
			pos++;
		}
		string key;
		while (s[pos] != '=')
		{
			key += s[pos++];
		}
		pos++; // consume =

		if (key == "name")
		{
			pos++; // consume "
			while (s[pos] != '\"')
			{
				e.name += s[pos++];
			}
			pos++; // consume "
		}
		else if (key == "position")
		{
			pos += 3; // consume {x=
			size_t comma = s.find(",", pos);
			e.x = stod(s.substr(pos, comma - pos));
			pos = comma + 3; // consume ,y=
			comma = s.find("}", pos);
			e.y = stod(s.substr(pos, comma - pos));
			pos = comma + 1; // consume }
		}
		else if (key == "direction")
		{
			e.direction = stoi(s.substr(pos, 1));
			pos++; // consume number
		}
		else if (key == "type")
		{
			pos++; // consume "
			e.name += "-";
			while (s[pos] != '\"')
			{
				e.name += s[pos++];
			}
			pos++; // consume "
		}
		else
		{
			goto error;
		}
	}
	pos++;

	return e;

error:
	e.name = "undefined";
	return e;
}

vector<BPEntity> parseString(const string& s)
{
	vector<BPEntity> out;
	unsigned int pos = 0;

	while (s[pos++] == ',')
	{
		BPEntity e = parseEntity(s, pos);
		if (e.name != "undefined")
		{
			out.push_back(e);
		}
	}

	return out;
}

#ifdef _DEBUG
void displayMap(int** map, int width, int height)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (map[x][y] >= 0 && map[x][y] <= 9)
			{
				cout << " ";
			}
			cout << map[x][y] << " ";
		}
		cout << endl;
	}
}
#endif

BeltEntity* parseBlueprintString(string blueprint, size_t* outputSize, bool optimize)
{
	const int maxUndergroundDistance = 4;

	vector<BeltEntity> output;

	string base64Blueprint;
	for (unsigned int i = 0; i < blueprint.length(); i++)
	{
		char c = blueprint[i];
		if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || c == '+' || c == '/' || c == '=')
		{
			base64Blueprint += c;
		}
	}
	bool vanillaFlag = false;
	if (base64Blueprint.length() % 4 == 1)
	{
#ifdef _DEBUG
		cout << "base64Blueprint.length() % 4 == 1" << endl;
#endif
		base64Blueprint = base64Blueprint.substr(1);
		vanillaFlag = true;
	}

#ifdef _DEBUG
	cout << base64Blueprint << endl << endl;
#endif

	string compressedBlueprint = base64_decode(base64Blueprint);

	if (vanillaFlag)
	{
		compressedBlueprint = compressedBlueprint;
	}

#ifdef _DEBUG
	cout << compressedBlueprint << endl << endl;
#endif

	string decompressed;
	vector<BPEntity> entities;

	if (vanillaFlag)
	{
		decompressed = decompress_string(compressedBlueprint);
		entities = parseVanillaJSON(decompressed);
	}
	else
	{
		if (!gzipInflate(compressedBlueprint, decompressed))
		{
			// bad
			*outputSize = 0;
			return 0;
		}

#ifdef _DEBUG
		cout << "decompressed.length() = " << decompressed.length() << endl;
		cout << decompressed << endl << endl;
#endif

		string startString("entities={");
		int start = decompressed.find(startString);
		if (start == string::npos)
		{
			cerr << "did not find the start string \"" << startString << "\"" << endl;
			cerr << decompressed << endl;
			*outputSize = 0;
			return 0;
		}
		start += startString.length();
		decompressed = "," + decompressed.substr(start, decompressed.size() - start);

		entities = parseString(decompressed);
	}

	double minx = 0;
	double miny = 0;
	double maxx = 0;
	double maxy = 0;

	for (BPEntity e : entities)
	{
#ifdef _DEBUG
		cout << "(" << e.name << "," << e.x << "," << e.y << "," << e.direction << ")" << endl;
#endif
		minx = (minx < e.x) ? minx : e.x;
		miny = (miny < e.y) ? miny : e.y;
		maxx = (maxx > e.x) ? maxx : e.x;
		maxy = (maxy > e.y) ? maxy : e.y;
	}

	maxx = maxx - minx + 3 + maxUndergroundDistance * 2;
	maxy = maxy - miny + 3 + maxUndergroundDistance * 2;

	for (unsigned int i = 0; i < entities.size(); i++)
	{
		entities[i].x = entities[i].x - minx + 1 + maxUndergroundDistance;
		entities[i].y = entities[i].y - miny + 1 + maxUndergroundDistance;
	}

#ifdef _DEBUG
	cout << endl;
	for (BPEntity e : entities)
	{
		cout << "(" << e.name << "," << e.x << "," << e.y << "," << e.direction << ")" << endl;
	}
	cout << endl;
#endif

	int width = (int)ceil(maxx);
	int height = (int)ceil(maxy);

	int** beltIdMap = new int*[width];
	for (int x = 0; x < width; x++)
	{
		beltIdMap[x] = new int[height];
		for (int y = 0; y < height; y++)
		{
			beltIdMap[x][y] = -1;
		}
	}

#ifdef _DEBUG
	displayMap(beltIdMap, width, height);
	cout << endl;
#endif

	vector<BPEntity> listWithDualSplitter;

	// make new list with two separate entities per splitter
	for (int i = 0; i < entities.size(); i++)
	{
		if (entities[i].name.find("splitter") != string::npos)
		{
			BPEntity r = entities[i];
			entities[i].name += "-left";
			r.name += "-right";
			switch (entities[i].direction)
			{
			case 0:
				entities[i].x -= 0.5;
				r.x += 0.5;
				break;
			case 2:
				entities[i].y += 0.5;
				r.y -= 0.5;
				break;
			case 4:
				entities[i].x += 0.5;
				r.x -= 0.5;
				break;
			case 6:
				entities[i].y -= 0.5;
				r.y += 0.5;
				break;
			}
			listWithDualSplitter.push_back(entities[i]);
			listWithDualSplitter.push_back(r);
		}
		else
		{
			listWithDualSplitter.push_back(entities[i]);
		}
	}

#ifdef _DEBUG
	for (BPEntity e : listWithDualSplitter)
	{
		cout << "(" << e.name << "," << e.x << "," << e.y << "," << e.direction << ")" << endl;
	}
	cout << endl;
#endif

	// initialize actual belt entity vector and the position to id map
	for (int i = 0; i < listWithDualSplitter.size(); i++)
	{
		BPEntity e = listWithDualSplitter[i];

		BeltEntity b;
		b.type = TYPE_BELT;
		b.otherSplitterPart = -1;
		b.next = -1;
		b.maxThroughput = 1.0 / 3;
		b.buffer = 0;
		b.addToBuffer = 0;
		b.subtractFromBuffer = 0;

		if (e.name.find("fast") != string::npos)
		{
			b.maxThroughput = 2.0 / 3;
		}
		else if (e.name.find("express") != string::npos)
		{
			b.maxThroughput = 1.0;
		}

		beltIdMap[(int)round(e.x)][(int)round(e.y)] = i;

		if (e.name.find("splitter-left") != string::npos)
		{
			b.type = TYPE_LEFT_SPLITTER;
			b.otherSplitterPart = i + 1;
		}
		else if (e.name.find("splitter-right") != string::npos)
		{
			b.type = TYPE_RIGHT_SPLITTER;
			b.otherSplitterPart = i - 1;
		}
		else if (e.name.find("transport-belt") != string::npos)
		{
			b.type = TYPE_BELT;
		}
		else if (e.name.find("underground-belt-output") != string::npos)
		{
			b.type = TYPE_UNDERGROUND_EXIT;
		}
		else if (e.name.find("underground-belt-input") != string::npos)
		{
			b.type = TYPE_UNDERGROUND_ENTRANCE;
		}

		output.push_back(b);
	}

#ifdef _DEBUG
	displayMap(beltIdMap, width, height);
	cout << endl;
#endif

	// set the correct next id and set void belts
	for (int i = 0; i < listWithDualSplitter.size(); i++)
	{
		BPEntity& e = listWithDualSplitter[i];
		int x = (int)round(e.x);
		int y = (int)round(e.y);
		BeltEntity& b = output[beltIdMap[x][y]];

		int dx = 0;
		int dy = 0;

		switch (e.direction)
		{
		case 0:
			dy = -1;
			break;
		case 2:
			dx = 1;
			break;
		case 4:
			dy = 1;
			break;
		case 6:
			dx = -1;
			break;
		}

		if (b.type == TYPE_UNDERGROUND_ENTRANCE)
		{
			for (int i = 1; i <= maxUndergroundDistance + 1; i++)
			{
				int id = beltIdMap[x + dx * i][y + dy * i];
				if (id == -1 || output[id].type != TYPE_UNDERGROUND_EXIT)
				{
					continue;
				}
				if (listWithDualSplitter[id].direction == e.direction)
				{
					b.next = id;
					break;
				}
			}
		}
		else
		{
			b.next = beltIdMap[x + dx][y + dy];
			if (b.next != -1 && output[b.next].type == TYPE_UNDERGROUND_EXIT)
			{
				int nd = listWithDualSplitter[b.next].direction;
				if ((8 + nd - e.direction) % 4 == 2)
				{
					cerr << "[Warning] Detected sideload on underground belt" << endl;
					// side loading on underground belt exit is bad!
				}
				if (nd == e.direction)
				{
					// don't fill the back side of an underground belt exit
					b.next = -1;
				}
			}
			else if (b.next != -1 && (output[b.next].type == TYPE_LEFT_SPLITTER || output[b.next].type == TYPE_RIGHT_SPLITTER))
			{
				int nd = listWithDualSplitter[b.next].direction;
				if ((8 + nd - e.direction) % 4 == 2)
				{
					// side loading on splitter does not work
					b.next = -1;
				}
			}
		}

		if (b.type == TYPE_BELT && b.next == -1)
		{
			b.type = TYPE_VOID;
		}
	}

	// detecting output splitter
	for (int i = 0; i < output.size(); i++)
	{
		if (output[i].type == TYPE_LEFT_SPLITTER && output[output[i].otherSplitterPart].next == -1 && output[i].next == -1)
		{
#ifdef _DEBUG
			cerr << "splitter pointing in nirvana" << endl;
#endif
			BeltEntity* lsplitter = &output[i];
			BeltEntity* rsplitter = &output[lsplitter->otherSplitterPart];
			lsplitter->next = output.size();
			rsplitter->next = output.size() + 1;
			BeltEntity b;
			b.type = TYPE_VOID;
			b.maxThroughput = lsplitter->maxThroughput;
			b.next = -1;
			b.otherSplitterPart = -1;
			b.buffer = 0;
			b.addToBuffer = 0;
			b.subtractFromBuffer = 0;
			output.push_back(b);
			output.push_back(b);
		}
	}

	// search for spawn belts and splitter
	set<int> hasPrevious;
	for (BeltEntity& b : output)
	{
		hasPrevious.insert(b.next);
	}
	for (int i = 0; i < output.size(); i++)
	{
		if (output[i].type == TYPE_BELT && hasPrevious.find(i) == hasPrevious.end())
		{
			output[i].type = TYPE_SPAWN;
		}
	}
	for (int i = 0; i < output.size(); i++)
	{
		if (output[i].type == TYPE_LEFT_SPLITTER && hasPrevious.find(i) == hasPrevious.end() && hasPrevious.find(output[i].otherSplitterPart) == hasPrevious.end())
		{
#ifdef _DEBUG
			cerr << "splitter getting nothing" << endl;
#endif
			BeltEntity b;
			b.type = TYPE_SPAWN;
			b.maxThroughput = output[i].maxThroughput;
			b.next = i;
			b.otherSplitterPart = -1;
			b.buffer = 0;
			b.addToBuffer = 0;
			b.subtractFromBuffer = 0;
			output.push_back(b);
			b.next = output[i].otherSplitterPart;
			output.push_back(b);
		}
	}

	// find dimensions without counting spawn and void belts
	{
		int minx = width;
		int miny = height;
		int maxx = 0;
		int maxy = 0;
		int inputBeltCount = 0;
		int outputBeltCount = 0;
		for (BeltEntity& b : output)
		{
			inputBeltCount += b.type == TYPE_SPAWN;
			outputBeltCount += b.type == TYPE_VOID;
		}
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				if (beltIdMap[x][y] == -1)
				{
					continue;
				}
				if (output[beltIdMap[x][y]].type == TYPE_SPAWN)
				{
					continue;
				}
				if (output[beltIdMap[x][y]].type == TYPE_VOID)
				{
					continue;
				}
				minx = minx < x ? minx : x;
				miny = miny < y ? miny : y;
				maxx = maxx > x ? maxx : x;
				maxy = maxy > y ? maxy : y;
			}
		}
		cout << "Loading a " << inputBeltCount << " to " << outputBeltCount << " balancer with dimensions " << (1 + maxx - minx) << "x" << (1 + maxy - miny) << endl;
	}

	if (optimize)
	{
		// replace underground belt entrance and exit with one belt piece
		for (unsigned int i = 0; i < output.size(); i++)
		{
			BeltEntity& o = output[i];
			if (o.type == TYPE_UNDERGROUND_ENTRANCE && o.next != -1 && output[o.next].type == TYPE_UNDERGROUND_EXIT)
			{
				output[o.next].type = TYPE_PLS_DELETE;
				o.type = TYPE_BELT;
				o.next = output[o.next].next;
			}
		}

		// bridge over belts with same throughput
		bool didSomethingChange = false;
		do
		{
			didSomethingChange = false;
			for (unsigned int i = 0; i < output.size(); i++)
			{
				BeltEntity& o = output[i];
				if (o.type == TYPE_LEFT_SPLITTER || o.type == TYPE_RIGHT_SPLITTER || o.type == TYPE_BELT || o.type == TYPE_SPAWN)
				{
					if (o.next != -1 && output[o.next].next != -1 && output[o.next].type == TYPE_BELT && o.maxThroughput == output[o.next].maxThroughput)
					{
						output[o.next].type = TYPE_PLS_DELETE;
						o.next = output[o.next].next;
						didSomethingChange = true;
					}
				}
			}
		} while (didSomethingChange);

		// make vector with new ids after deleting all TYPE_PLS_DELETE
		vector<int> subFromId;
		int deletedUntilNow = 0;
		for (unsigned int i = 0; i < output.size(); i++)
		{
			if (output[i].type == TYPE_PLS_DELETE)
			{
				deletedUntilNow++;
			}
			subFromId.push_back(deletedUntilNow);
		}

		// arrange new ids in fields next and otherSplitterPart and copy non TYPE_PLS_DELETE to actual output vector
		vector<BeltEntity> smallerOutput;
		for (unsigned int i = 0; i < output.size(); i++)
		{
			if (output[i].next != -1)
			{
				output[i].next -= subFromId[output[i].next];
			}
			if (output[i].otherSplitterPart != -1)
			{
				output[i].otherSplitterPart -= subFromId[output[i].otherSplitterPart];
			}
			if (output[i].type != TYPE_PLS_DELETE)
			{
				smallerOutput.push_back(output[i]);
			}
		}

		output.clear();
		output = smallerOutput;
	}

	*outputSize = output.size() + 1;
	BeltEntity* o = new BeltEntity[*outputSize];
	for (unsigned int i = 1; i < *outputSize; i++)
	{
		o[i] = output[i - 1];
	}

	BeltEntity block;
	block.type = TYPE_BLOCK;
	block.buffer = 0;
	block.addToBuffer = 0;
	block.subtractFromBuffer = 0;
	block.next = -1;
	block.otherSplitterPart = -1;
	block.maxThroughput = 0;
	o[0] = block;

	return o;
}