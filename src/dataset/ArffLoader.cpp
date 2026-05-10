#include "ArffLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <cctype>



namespace ml {

    static std::string toUpper(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
                    [](unsigned char c) { return std::toupper(c); });
        return result;
    }

    std::string ArffLoader::trim(const std::string& str) {

        size_t first = str.find_first_not_of(" \t\n\r");

        if (first == std::string::npos) return "";
            size_t last = str.find_last_not_of(" \t\n\r");

        return str.substr(first, last - first + 1);
    }

    std::vector<std::string> ArffLoader::split(const std::string& str, char delimiter) {

        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter)) {

            tokens.push_back(trim(token));

        }

        return tokens;
    }

   void ArffLoader::parseHeader(std::ifstream& file, std::vector<Attribute>& attributes, std::string& relationName) {
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '%')
            continue;

        std::string upperLine = line;
        std::transform(upperLine.begin(), upperLine.end(), upperLine.begin(), ::toupper);

        if (upperLine.substr(0, 9) == "@RELATION") {
            relationName = trim(line.substr(9));
        }
        else if (upperLine.substr(0, 10) == "@ATTRIBUTE") {
            // format: @ATTRIBUTE nombre tipo
            std::string rest = trim(line.substr(10));
            size_t spacePos = rest.find(' ');
            if (spacePos == std::string::npos)
                throw std::runtime_error("Formato @ATTRIBUTE inválido");
            std::string attName = trim(rest.substr(0, spacePos));
            std::string attType = trim(rest.substr(spacePos + 1));

            Attribute attr;
            attr.name = attName;

            // Convert the type to uppercase for case-insensitive comparison
            std::string attTypeUpper = attType;
            std::transform(attTypeUpper.begin(), attTypeUpper.end(), attTypeUpper.begin(), ::toupper);

            if (attTypeUpper == "NUMERIC" || attTypeUpper == "REAL" || attTypeUpper == "INTEGER") {
                attr.isNumeric = true;
            }
            else if (attType.front() == '{' && attType.back() == '}') {
                attr.isNumeric = false;
                // extract the list of values between curly brackets
                std::string vals = attType.substr(1, attType.size() - 2);
                attr.nominalValues = split(vals, ',');
                for (auto& v : attr.nominalValues)
                    v = trim(v);
            }
            else {
                throw std::runtime_error("Unsupported attribute type: " + attType);
            }
            attributes.push_back(attr);
        }
        else if (upperLine == "@DATA") {
            break;
        }
    }
}


    void ArffLoader::parseData(std::ifstream& file, const std::vector<Attribute>& attributes,
                           int classIndex, Dataset& dataset) {
    std::string line;
    std::vector<Instance> instances;

    // Mapping of nominal values to integers (for undeclared values)
    std::vector<std::map<std::string, int>> nominalMaps(attributes.size());

    // Initialise maps with the values declared in the header
    for (size_t i = 0; i < attributes.size(); ++i) {
        if (!attributes[i].isNumeric) {
            for (size_t v = 0; v < attributes[i].nominalValues.size(); ++v) {
                nominalMaps[i][attributes[i].nominalValues[v]] = static_cast<int>(v);
            }
        }
    }

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '%') continue;

        std::vector<std::string> values = split(line, ',');
        if (values.size() != attributes.size()) {
            std::cerr << "Warning: the number of values differs from the inline attributes: " << line << std::endl;
            continue;
        }

        std::vector<double> features;
        int label = -1;

        for (size_t idx = 0; idx < attributes.size(); ++idx) {
            const auto& attr = attributes[idx];
            std::string valStr = values[idx];

            if (valStr == "?") {
                // Missing value: set to 0.0 for numeric values, “error” for categorical values
                if (attr.isNumeric) {
                    double missingVal = 0.0;
                    if (idx == static_cast<size_t>(classIndex))
                        label = static_cast<int>(missingVal);
                    else
                        features.push_back(missingVal);
                } else {
                    throw std::runtime_error("Missing value not supported in a nominal attribute");
                }
                continue;
            }

            if (attr.isNumeric) {
                double num = std::stod(valStr);
                if (idx == static_cast<size_t>(classIndex))
                    label = static_cast<int>(num);
                else
                    features.push_back(num);
            } else {
                // Nominal attribute
                auto& nomMap = nominalMaps[idx];
                int intVal;
                auto it = nomMap.find(valStr);
                if (it != nomMap.end()) {
                    intVal = it->second;
                } else {
                    // New undeclared value: add to the map
                    intVal = static_cast<int>(nomMap.size());
                    nomMap[valStr] = intVal;
                    std::cerr << "Warning: nominal value '" << valStr
                              << "' If not declared in @ATTRIBUTE, a code is assigned " << intVal << std::endl;
                }
                if (idx == static_cast<size_t>(classIndex))
                    label = intVal;
                else
                    features.push_back(static_cast<double>(intVal));
            }
        }

        if (label == -1)
            throw std::runtime_error("The class was not found in the instance");
        instances.push_back({features, label});
    }

    // Assign the instances to the dataset using the setInstances method
    if (!instances.empty()) {
        dataset.setInstances(instances);
    } else {
        throw std::runtime_error("No instances of the ARFF file were loaded");
    }
}


    void ArffLoader::load(const std::string& filename, Dataset& dataset, int classIndex) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("The file could not be opened ARFF: " + filename);
        }

        std::vector<Attribute> attributes;
        std::string relationName;
        parseHeader(file, attributes, relationName);

        if (attributes.empty()) {
            throw std::runtime_error("No attributes were found in the file ARFF");
        }

        // If classIndex is -1, use the last attribute
        if (classIndex == -1) {
            classIndex = static_cast<int>(attributes.size()) - 1;
        }
        if (classIndex < 0 || classIndex >= (int)attributes.size()) {
            throw std::runtime_error("Invalid class index");
        }

        parseData(file, attributes, classIndex, dataset);

        std::cout << "Dataset loaded ARFF: " << relationName << " | Instances: " << dataset.getInstances().size()
                << " | Attributes: " << dataset.getNumFeatures() << " + class\n";
    }

} // namespace ml
