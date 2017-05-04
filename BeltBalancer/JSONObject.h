#ifndef JSONOBJECT_H
#define JSONOBJECT_H

#include<string>
#include<vector>
#include<map>

class JSONObject
{
public:
	JSONObject();
	JSONObject(const std::string& jsonString);
	JSONObject(JSONObject* makeCopyFrom);
	virtual ~JSONObject();

	bool IsNull();
	bool IsArray();
	bool IsObject();
	bool IsBool();
	bool IsString();
	bool IsNumber();

	bool ParseFile(const std::string& filePath);

	bool Parse(const std::string& jsonString);
	bool Parse(const std::string& jsonString, int& pos);

	JSONObject* SetBool(bool value);
	JSONObject* SetString(const std::string& value);
	JSONObject* SetNumber(double value);
	JSONObject* SetArray();
	JSONObject* SetNull();
	JSONObject* SetJSONObject();

	JSONObject* SetNameValue(const std::string& name, JSONObject* value);
	void Remove(const std::string& name);
	void AddToArray(JSONObject* value);

	JSONObject* GetPath(const std::string& path, char delim = '.');
	JSONObject* Get(const std::string& name);

	std::vector<JSONObject*>* GetArray(const std::string& name);
	double GetNumber(const std::string& name);
	std::string GetString(const std::string& name);
	bool GetBool(const std::string& name);

	std::vector<JSONObject*>* GetArray();
	double GetNumber();
	std::string GetString();
	bool GetBool();

	bool IsOk();

	std::vector<std::string> GetChildNames();

	std::string ToString(int indentationDelta = 0, int indentation = 0);

protected:
private:
	bool isNull;
	bool isArray;
	bool isObject;
	bool isBool;
	bool isString;
	bool isNumber;

	void* data;
	std::map<std::string, int>* indexMap;
	std::vector<std::string>* nameVector;

	void ClearData();
};

#endif // JSONOBJECT_H
